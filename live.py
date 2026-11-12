"""Live price executor for the repricer. OFF unless ALL of these hold for a SKU:
  1. Render env LIVE_MODE=on                        (Nick sets this himself)
  2. live config enabled=true and the SKU is listed  (POST /live with the recorder key)
  3. Seller Snap repricing is OFF for that SKU        (latest Seller Snap snapshot; otherwise the two would fight)
Guardrails on every change: within the SKU's Seller Snap min/max, at most MAX_STEP_PCT per change (moves toward the target in steps),
at least MIN_INTERVAL_S between changes per SKU, at most MAX_PER_DAY changes per SKU per day, never on a stale decision.
Prices are written with the Listings Items API (patchListingsItem, purchasable_offer.our_price), keeping the offer's other fields.
Every attempt is logged to live/<date>.jsonl.gz. POST /live/stop turns it off immediately."""
import copy, datetime as dt, json, os, threading, time
import requests

SP = 'https://sellingpartnerapi-na.amazon.com'
MP = 'ATVPDKIKX0DER'
ME = 'A2UYRB3AYUMGAJ'
DEFAULTS = {'enabled': False, 'skus': [], 'max_step_pct': 5.0, 'min_interval_s': 600, 'max_per_day': 24}


class Live:
    def __init__(self, data_dir, append_gz, iso, sp_token):
        self.data, self.append_gz, self.iso, self.sp_token = data_dir, append_gz, iso, sp_token
        self.lock = threading.Lock()
        self.ptype = {}                      # sku -> productType
        self.last_change = {}                # sku -> time of last applied change
        self.day_count = {}                  # (sku, date) -> changes applied
        self.ss_repricing = {}               # sku -> Seller Snap repricing flag
        self.stats = {'env_live_mode': os.environ.get('LIVE_MODE', 'off'), 'enabled': False, 'skus': 0, 'applied': 0,
                      'blocked': {}, 'last_applied': None, 'last_error': None}
        self.cfg = self.load_cfg()

    # ---- config
    def cfg_path(self):
        return os.path.join(self.data, 'live', 'config.json')

    def load_cfg(self):
        c = dict(DEFAULTS)
        try:
            with open(self.cfg_path()) as f:
                c.update(json.load(f))
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
        c['max_per_day'] = int(min(max(c['max_per_day'], 1), 96))
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
        if d.get('action') == 'hold':
            return None
        if not self.env_on():
            return None
        if not self.cfg.get('enabled') or sku not in set(self.cfg['skus']):
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
        if abs(new - me) < 0.01:
            return None
        if now - self.last_change.get(sku, 0) < self.cfg['min_interval_s']:
            return self._block('too soon since last change')
        day = dt.datetime.fromtimestamp(now, dt.timezone.utc).strftime('%Y-%m-%d')
        if self.day_count.get((sku, day), 0) >= self.cfg['max_per_day']:
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
               'gain_24h': d.get('gain_24h'), 'reasons': d.get('reasons')}
        try:
            code, resp, body = self.send(sku, new)
            rec.update(status_code=code, response_status=resp.get('status'), issues=resp.get('issues'))
            if code < 300 and resp.get('status') in ('ACCEPTED', None):
                now = time.time()
                self.last_change[sku] = now
                day = dt.datetime.fromtimestamp(now, dt.timezone.utc).strftime('%Y-%m-%d')
                self.day_count[(sku, day)] = self.day_count.get((sku, day), 0) + 1
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
