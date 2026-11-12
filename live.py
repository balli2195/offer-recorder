"""Live price executor for the repricer. OFF unless ALL of these hold for a SKU:
  1. Render env LIVE_MODE=on                        (Nick sets this himself)
  2. live config enabled=true and the SKU is listed  (POST /live with the recorder key)
  3. Seller Snap repricing is OFF for that SKU        (latest Seller Snap snapshot; otherwise the two would fight)
Guardrails on every change: within the SKU's Seller Snap min/max, at most MAX_STEP_PCT per change (moves toward the target in steps),
at least MIN_INTERVAL_S between changes per SKU, at most MAX_PER_DAY changes per SKU per day (Central-time day), never on a stale decision.
Brakes on runs of price cuts (a loop with another repricer walks the price down; a bounce up and down costs little):
  chased     a cut counts as answered when a rival moves to or below the new price within CHASE_WINDOW_MIN; after CHASE_CUTS answered cuts
             in a row, cuts on that SKU stop for FREEZE_H hours (raises and holds still allowed) and the SKU is listed as paused
  drop limit a cut never takes the price more than MAX_DROP_PCT below its highest point in the last 24 h
Prices are written with the Listings Items API (patchListingsItem, purchasable_offer.our_price), keeping the offer's other fields.
Every attempt is logged to live/<date>.jsonl.gz. POST /live/stop turns it off immediately."""
import collections, copy, datetime as dt, gzip, json, math, os, threading, time
from zoneinfo import ZoneInfo
import requests

SP = 'https://sellingpartnerapi-na.amazon.com'
MP = 'ATVPDKIKX0DER'
ME = 'A2UYRB3AYUMGAJ'
CT = ZoneInfo('America/Chicago')
DEFAULTS = {'enabled': False, 'skus': [], 'max_step_pct': 5.0, 'min_interval_s': 600, 'max_per_day': 96,
            'chase_window_min': 30, 'chase_cuts': 2, 'freeze_h': 6.0, 'max_drop_pct': 10.0,
            # decision tuning, used by shadow and live alike; decay_auto = cost of waiting from the store's own return on inventory
            'decay_per_day': 0.003, 'decay_auto': True, 'min_gain_abs': 0.50, 'min_gain_rel': 0.05}


def ct_day(t):
    return dt.datetime.fromtimestamp(t, CT).strftime('%Y-%m-%d')


class Live:
    def __init__(self, data_dir, append_gz, iso, sp_token):
        self.data, self.append_gz, self.iso, self.sp_token = data_dir, append_gz, iso, sp_token
        self.lock = threading.Lock()
        self.ptype = {}                      # sku -> productType
        self.last_change = {}                # sku -> time of last applied change
        self.day_count = {}                  # (sku, date) -> changes applied
        self.ss_repricing = {}               # sku -> Seller Snap repricing flag
        self.seen = collections.defaultdict(collections.deque)   # sku -> (t, my price) over the last 24 h, live SKUs only
        self.cuts = {}                       # sku -> last applied cut {t, price, low_before, answered}
        self.streak = {}                     # sku -> answered cuts in a row
        self.frozen = {}                     # sku -> {until (epoch), why}
        self.stats = {'env_live_mode': os.environ.get('LIVE_MODE', 'off'), 'enabled': False, 'skus': 0, 'applied': 0,
                      'blocked': {}, 'last_applied': None, 'last_error': None}
        self.cfg = self.load_cfg()
        try:
            self.load_state()
        except Exception as e:  # never stop the recorder over brake state
            self.stats['last_error'] = f'state load: {type(e).__name__}: {str(e)[:200]}'

    # ---- config
    def cfg_path(self):
        return os.path.join(self.data, 'live', 'config.json')

    def load_cfg(self):
        c = dict(DEFAULTS)
        try:
            with open(self.cfg_path()) as f:
                c.update(json.load(f))
            c = {**DEFAULTS, **c}
        except FileNotFoundError:
            pass
        self.stats.update(enabled=bool(c['enabled']), skus=len(c['skus']))
        return c

    def set_cfg(self, upd):
        c = dict(self.cfg)
        for k in DEFAULTS:
            if k in upd:
                c[k] = upd[k]
        c['skus'] = sorted({str(s) for s in c['skus']})
        c['max_step_pct'] = float(min(max(c['max_step_pct'], 0.5), 10.0))
        c['min_interval_s'] = int(max(c['min_interval_s'], 120))
        c['max_per_day'] = int(min(max(c['max_per_day'], 1), 144))
        c['chase_window_min'] = int(min(max(c['chase_window_min'], 5), 240))
        c['chase_cuts'] = int(min(max(c['chase_cuts'], 1), 5))
        c['freeze_h'] = float(min(max(c['freeze_h'], 0.5), 48.0))
        c['max_drop_pct'] = float(min(max(c['max_drop_pct'], 1.0), 50.0))
        c['decay_auto'] = bool(c['decay_auto'])
        c['decay_per_day'] = float(min(max(c['decay_per_day'], 0.0), 0.02))
        c['min_gain_abs'] = float(min(max(c['min_gain_abs'], 0.0), 50.0))
        c['min_gain_rel'] = float(min(max(c['min_gain_rel'], 0.0), 1.0))
        os.makedirs(os.path.dirname(self.cfg_path()), exist_ok=True)
        tmp = self.cfg_path() + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(c, f)
        os.replace(tmp, self.cfg_path())
        self.cfg = c
        self.stats.update(enabled=bool(c['enabled']), skus=len(c['skus']))
        return c

    def stop(self):
        return self.set_cfg({'enabled': False})

    # ---- brake state: frozen SKUs and streaks are saved; changes, cuts and 24 h prices are rebuilt from the live log at start
    def state_path(self):
        return os.path.join(self.data, 'live', 'state.json')

    def save_state(self):
        try:
            os.makedirs(os.path.dirname(self.state_path()), exist_ok=True)
            tmp = self.state_path() + '.tmp'
            with open(tmp, 'w') as f:
                json.dump({'frozen': self.frozen, 'streak': self.streak}, f)
            os.replace(tmp, self.state_path())
        except OSError:
            pass

    def load_state(self):
        try:
            with open(self.state_path()) as f:
                st = json.load(f)
            self.frozen = {k: v for k, v in (st.get('frozen') or {}).items() if v.get('until', 0) > time.time()}
            self.streak = st.get('streak') or {}
        except (FileNotFoundError, ValueError):
            pass
        cutoff = time.time() - 86400
        for r in sorted(self.recent_log(2), key=lambda r: r.get('t') or ''):
            if r.get('status_code', 999) >= 300 or r.get('error'):
                continue
            try:
                t = dt.datetime.fromisoformat(r['t'].replace('Z', '+00:00')).timestamp()
            except (KeyError, ValueError):
                continue
            sku = r['sku']
            self.last_change[sku] = max(self.last_change.get(sku, 0), t)
            self.day_count[(sku, ct_day(t))] = self.day_count.get((sku, ct_day(t)), 0) + 1
            if t >= cutoff:
                self.seen[sku].append((t, r['from']))
                self.seen[sku].append((t, r['to']))
            if r['to'] < r['from']:
                self.cuts[sku] = {'t': t, 'price': r['to'], 'low_before': r.get('low_rival'), 'answered': False}
        self._frozen_stats()

    def _frozen_stats(self):
        now = time.time()
        for k in [k for k, v in self.frozen.items() if v['until'] <= now]:
            del self.frozen[k]
        self.stats['frozen'] = {k: {'until': dt.datetime.fromtimestamp(v['until'], dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'), 'why': v['why']}
                                for k, v in self.frozen.items()}

    def unfreeze(self, sku):
        self.frozen.pop(sku, None)
        self.streak[sku] = 0
        self._frozen_stats()
        self.save_state()
        return self.stats['frozen']

    def observe(self, d, now):
        """Track the live SKU's price over 24 h and whether rivals answered our last cut."""
        sku = d['sku']
        q = self.seen[sku]
        q.append((now, d['my_price']))
        while q and q[0][0] < now - 86400:
            q.popleft()
        c = self.cuts.get(sku)
        if not c or c['answered'] is not None:
            return
        win = self.cfg['chase_window_min'] * 60
        low = d.get('low_rival')
        moved = low is not None and (c['low_before'] is None or low < c['low_before'] - 0.005)
        if moved and low <= c['price'] + 0.005 and now - c['t'] <= win:
            c['answered'] = True
            self.streak[sku] = self.streak.get(sku, 0) + 1
            if self.streak[sku] >= self.cfg['chase_cuts']:
                self.frozen[sku] = {'until': now + self.cfg['freeze_h'] * 3600,
                                    'why': f"a rival answered {self.streak[sku]} cuts in a row within {self.cfg['chase_window_min']} min"}
                self._frozen_stats()
            self.save_state()
        elif now - c['t'] > win:
            c['answered'] = False
            self.streak[sku] = 0
            self.save_state()

    def on_listings(self, rows):
        with self.lock:
            self.ss_repricing = {r['sku']: bool(r.get('repricing')) for r in rows if r.get('sku')}

    def env_on(self):
        return os.environ.get('LIVE_MODE', 'off').lower() == 'on'

    # ---- decisions
    def _block(self, why):
        self.stats['blocked'][why] = self.stats['blocked'].get(why, 0) + 1
        return None

    def plan(self, d, now=None):
        """Returns the price to set for decision d, or None (with the reason counted). Pure checks, no API calls."""
        now = now or time.time()
        sku = d.get('sku')
        if not self.env_on():
            return None
        if not self.cfg.get('enabled') or sku not in set(self.cfg['skus']):
            return None
        self.observe(d, now)
        if d.get('action') == 'hold':
            return None
        with self.lock:
            ss = self.ss_repricing.get(sku)
        if ss is None:
            return self._block('no Seller Snap snapshot for SKU')
        if ss:
            return self._block('Seller Snap repricing still on')
        me, rec, lo, hi = d['my_price'], d['rec_price'], d['min'], d['max']
        if not (lo and hi and lo <= rec <= hi):
            return self._block('outside min/max')
        step = self.cfg['max_step_pct'] / 100.0
        new = round(min(max(rec, me * (1 - step)), me * (1 + step)), 2)
        new = round(min(max(new, lo), hi), 2)
        if new < me - 0.005:
            self._frozen_stats()
            if sku in self.frozen:
                return self._block('cuts paused: chased by a rival')
            high = max([p for _, p in self.seen[sku]] + [me])
            floor = math.ceil(high * (1 - self.cfg['max_drop_pct'] / 100.0) * 100 - 1e-6) / 100.0
            if new < floor:
                new = floor
                if new >= me - 0.005:
                    return self._block('24 h drop limit')
        if abs(new - me) < 0.01:
            return None
        if now - self.last_change.get(sku, 0) < self.cfg['min_interval_s']:
            return self._block('too soon since last change')
        if self.day_count.get((sku, ct_day(now)), 0) >= self.cfg['max_per_day']:
            return self._block('daily change cap')
        return new

    # ---- Amazon calls
    def _get_item(self, sku, token):
        r = requests.get(f'{SP}/listings/2021-08-01/items/{ME}/{requests.utils.quote(sku, safe="")}', headers={'x-amz-access-token': token},
                         params={'marketplaceIds': MP, 'includedData': 'summaries,attributes'}, timeout=30)
        r.raise_for_status()
        return r.json()

    def build_patch(self, item, price):
        pt = (item.get('summaries') or [{}])[0].get('productType')
        offers = copy.deepcopy((item.get('attributes') or {}).get('purchasable_offer') or [])
        mine = [o for o in offers if o.get('marketplace_id') == MP] or [{'marketplace_id': MP, 'currency': 'USD'}]
        o = mine[0]
        o['our_price'] = [{'schedule': [{'value_with_tax': round(price, 2)}]}]
        return pt, {'productType': pt, 'patches': [{'op': 'replace', 'path': '/attributes/purchasable_offer', 'value': [o]}]}

    def send(self, sku, price, preview=False):
        token = self.sp_token()
        item = self._get_item(sku, token)
        pt, body = self.build_patch(item, price)
        params = {'marketplaceIds': MP}
        if preview:
            params['mode'] = 'VALIDATION_PREVIEW'   # Amazon validates the change but does not apply it
        r = requests.patch(f'{SP}/listings/2021-08-01/items/{ME}/{requests.utils.quote(sku, safe="")}', headers={'x-amz-access-token': token, 'content-type': 'application/json'},
                           params=params, json=body, timeout=30)
        return r.status_code, r.json() if r.content else {}, body

    def maybe_apply(self, d):
        new = self.plan(d)
        if new is None:
            return None
        sku = d['sku']
        rec = {'t': self.iso(), 'sku': sku, 'asin': d['asin'], 'from': d['my_price'], 'to': new, 'target_rec': d['rec_price'], 'action': d['action'],
               'low_rival': d.get('low_rival'), 'gain_24h': d.get('gain_24h'), 'profit_day_now': d.get('profit_day_now'),
               'profit_day_rec': d.get('profit_day_rec'), 'reasons': d.get('reasons')}
        try:
            code, resp, body = self.send(sku, new)
            rec.update(status_code=code, response_status=resp.get('status'), issues=resp.get('issues'))
            if code < 300 and resp.get('status') in ('ACCEPTED', None):
                now = time.time()
                self.last_change[sku] = now
                self.day_count[(sku, ct_day(now))] = self.day_count.get((sku, ct_day(now)), 0) + 1
                self.seen[sku].append((now, new))
                if new < d['my_price']:
                    self.cuts[sku] = {'t': now, 'price': new, 'low_before': d.get('low_rival'), 'answered': None}
                else:
                    self.cuts.pop(sku, None)
                    self.streak[sku] = 0
                    self.save_state()
                self.stats['applied'] += 1
                self.stats['last_applied'] = rec['t']
            else:
                self._block('Amazon rejected')
        except Exception as e:
            rec['error'] = f'{type(e).__name__}: {str(e)[:200]}'
            self.stats['last_error'] = rec['t'] + ' ' + rec['error']
        t = dt.datetime.now(dt.timezone.utc)
        self.append_gz(os.path.join(self.data, 'live', t.strftime('%Y-%m-%d') + '.jsonl.gz'), [json.dumps(rec)])
        return rec

    def recent_log(self, days=3):
        out = []
        for i in range(days):
            d = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=i)).strftime('%Y-%m-%d')
            p = os.path.join(self.data, 'live', d + '.jsonl.gz')
            if os.path.exists(p):
                with gzip.open(p, 'rt') as f:
                    out += [json.loads(l) for l in f if l.strip()]
        return sorted(out, key=lambda r: r.get('t') or '', reverse=True)
