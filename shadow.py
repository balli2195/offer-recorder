"""Shadow-mode repricer. READ-ONLY: it computes the price it would set and logs the decision; it never calls any API
that changes a price or a Seller Snap setting.

Inputs (kept in memory, fed by the recorder's existing loops):
  offers     latest FBA new offers per ASIN from ANY_OFFER_CHANGED messages (seller -> landed price, Buy Box winner)
  listings   Seller Snap snapshot rows (sku, asin, min, max, cost, roi, v7, v30, avg_bb_share, repricing)
  inventory  fulfillable quantity per SKU
  foep       Amazon's featured offer expected price per SKU, with its time
  keepa      rival FBA stock per ASIN (latest Keepa pull)
Model: model.json written by tools/train.py (Buy Box logistic models, per-rival reaction rates, price-war penalty table).
"""
import json, math, os, threading, time, collections, datetime as dt

ME = 'A2UYRB3AYUMGAJ'
HORIZON_H = 24.0
FOEP_MAX_AGE = 45 * 60
MIN_GAIN_ABS = 0.50      # $/day of expected profit before a change is recommended
MIN_GAIN_REL = 0.05      # or 5 %
DEBOUNCE_S = 120         # at most one decision per ASIN every 2 minutes
DECAY_PER_DAY = float(os.environ.get('SHADOW_DECAY_PER_DAY', '0.003'))  # cost of waiting to sell, as a share of price per day


# ---------------------------------------------------------------- features (shared with tools/train.py)
def clip(x, lo, hi):
    return max(lo, min(hi, x))


def feats(me, low_rival, n_rivals, foep=None, share=None):
    """Feature dict for the Buy Box logistic model. Prices are landed prices; foep is Amazon's target for Nick (or None)."""
    f = clip(100.0 * (me / low_rival - 1.0), -10, 20)
    x = {'b': 1.0, 'low_neg': max(min(f, 0.0), -1.0), 'low_pos': max(f, 0.0), 'tied': 1.0 if abs(f) <= 0.5 else 0.0,
         'above1': 1.0 if f > 1.0 else 0.0, 'lnriv': math.log(max(n_rivals, 1))}
    if share is None:
        x['share'] = 0.0; x['share_na'] = 1.0
    else:
        x['share'] = clip(share, 0, 1); x['share_na'] = 0.0
    if foep is not None and foep > 0:
        g = clip(100.0 * (me / foep - 1.0), -10, 20)
        x.update({'t_neg': max(min(g, 0.0), -1.0), 't_pos': max(g, 0.0), 't_ok': 1.0 if g <= 0.01 else 0.0})
    return x


def p_win(model, x):
    key = 'bb_foep' if 't_ok' in x else 'bb_nofoep'
    m = model[key]
    z = sum(m['coef'].get(k, 0.0) * v for k, v in x.items())
    return 1.0 / (1.0 + math.exp(-clip(z, -30, 30)))


# ---------------------------------------------------------------- economics
def fee(p, sku):
    ref = 0.08 if p <= 10 else 0.15
    return ref * p + sku['fixed_fee'], ref


def margin(p, sku):
    f, _ = fee(p, sku)
    return p - f - sku['cost']


def rival_info(model, seller):
    r = model['rivals'].get(seller)
    d = model['rival_default']
    if not r:
        return d['rate'], d['lag_min'], False
    return r['rate'], r.get('lag_min') or d['lag_min'], False


def war_penalty_pct(model, rate):
    """Expected % drop in the listing low 24 h after an undercut, given the undercut rival's reactive rate."""
    tab = model['war_table']  # list of [rate_mid, pct_drop]
    if rate <= tab[0][0]:
        return tab[0][1]
    for (r0, d0), (r1, d1) in zip(tab, tab[1:]):
        if rate <= r1:
            return d0 + (d1 - d0) * (rate - r0) / (r1 - r0)
    return tab[-1][1]


def decide(model, asin, offers, winner, sku, foep=None, foep_age=None, my_stock=None, rival_stock=None, rival_floor_hit=None, tune=None):
    """offers: {seller: landed price} for FBA new offers on the listing; winner: Buy Box seller id or None.
    sku: dict with sku, min, max, cost, fixed_fee, lam (units per Buy Box hour). Returns a decision dict or None."""
    if ME not in offers or len(offers) < 2:
        return None
    me = offers[ME]
    rivals = {s: p for s, p in offers.items() if s != ME}
    low_s = min(rivals, key=rivals.get)
    low = rivals[low_s]
    near = [s for s, p in rivals.items() if p <= low * 1.005]
    n_riv = len(rivals)
    use_foep = foep if (foep and foep_age is not None and foep_age <= FOEP_MAX_AGE) else None
    share = None
    if my_stock is not None and rival_stock is not None and (my_stock + rival_stock) > 0:
        share = my_stock / (my_stock + rival_stock)
    tune = tune or {}
    decay = tune.get('decay_per_day', DECAY_PER_DAY)
    g_abs = tune.get('min_gain_abs', MIN_GAIN_ABS)
    g_rel = tune.get('min_gain_rel', MIN_GAIN_REL)
    lo, hi = sku['min'], sku['max']
    if not (lo and hi) or lo > hi:
        return None
    cands = {round(me, 2), round(low, 2), round(low - 0.01, 2)}
    if use_foep:  # small steps only when Amazon's target is known; without it the Buy Box model is much weaker (AUC ~0.72 vs ~0.89)
        cands |= {math.floor(use_foep * 100) / 100.0, round(me * 1.01, 2), round(me * 1.02, 2), round(me * 0.99, 2)}
    if me <= low - 0.005:  # Nick alone at the low: the next offer minus a cent is a raise candidate
        cands |= {round(low - 0.01, 2)}
    cands = sorted({round(clip(c, lo, hi), 2) for c in cands})
    lam = sku['lam']
    rate = max((rival_info(model, s)[0] for s in near), default=model['rival_default']['rate'])
    lag = min((rival_info(model, s)[1] for s in near), default=model['rival_default']['lag_min'])
    p_now = p_win(model, feats(me, low, n_riv, use_foep, share))
    qty = my_stock if my_stock is not None else (sku.get('qty') or 1)
    qty = max(qty, 1)
    daily_now = lam * 24.0 * p_now
    days_to_sell = min(qty / max(daily_now, 0.05), 180.0)
    # What is already paid for is sunk: a unit sold now is worth its net proceeds; a unit kept is worth selling it later
    # near today's market low, less the cost of waiting (capital, storage, price erosion) over about half the sell-through time.
    ref_low = 0.08 if low <= 10 else 0.15
    v_future = (low - (ref_low * low + sku['fixed_fee'])) * (1.0 - decay * days_to_sell / 2.0)
    exposed = min(qty, max(daily_now, 0.05) * 30.0)   # units whose future value a price war would lower
    rows = []
    for p in cands:
        m = margin(p, sku)
        f_p, ref = fee(p, sku)
        pw_alone = p_win(model, feats(p, low, n_riv, use_foep, share))
        pen = 0.0
        if p < low - 0.005 and p < me - 0.005:  # a new cut below the rivals: they may follow after their lag; then we are tied at p
            pw_tied = p_win(model, feats(p, p, n_riv, use_foep, share))
            fl = min(lag / 60.0 / HORIZON_H, 1.0)
            pw = (1 - rate) * pw_alone + rate * (fl * pw_alone + (1 - fl) * pw_tied)
            pen = war_penalty_pct(model, rate) / 100.0 * low * (1 - ref_low) * exposed
        else:
            pw = pw_alone
        units = min(lam * HORIZON_H * pw, qty)
        ev = units * ((p - f_p) - v_future) - pen
        rows.append({'price': p, 'p_win': round(pw, 4), 'units_24h': round(units, 3), 'margin': round(m, 2),
                     'roi': round(m / sku['cost'], 4) if sku['cost'] else None, 'penalty': round(pen, 2), 'exp_value_24h': round(ev, 2)})
    cur = min(rows, key=lambda r: abs(r['price'] - me))
    best = max(rows, key=lambda r: r['exp_value_24h'])
    gain = best['exp_value_24h'] - cur['exp_value_24h']
    change = gain >= max(g_abs, g_rel * abs(cur['exp_value_24h'])) and abs(best['price'] - me) >= 0.01
    rec = best['price'] if change else me
    if not change:
        action = 'hold'
    elif rec > me:
        action = 'raise_to_target' if (use_foep and abs(rec - math.floor(use_foep * 100) / 100.0) < 0.005) else ('raise_to_next' if abs(rec - (low - 0.01)) < 0.005 else 'raise')
    elif rec < low - 0.005:
        action = 'undercut'
    elif abs(rec - low) < 0.005:
        action = 'match_low'
    else:
        action = 'lower'
    reasons = []
    if winner == ME:
        reasons.append('holding Buy Box')
    if use_foep:
        reasons.append(f'target {use_foep:.2f} ({(me / use_foep - 1) * 100:+.1f}% vs mine)')
    else:
        reasons.append('no fresh target price')
    reasons.append(f'lowest rival {low:.2f} ({low_s}), reactive rate {rate:.0%}')
    if share is not None:
        reasons.append(f'stock share {share:.0%}')
    if lo > low:
        reasons.append('floor above lowest rival')
    return {'asin': asin, 'sku': sku['sku'], 'my_price': me, 'low_rival': low, 'n_rivals': n_riv, 'winner': winner,
            'foep': use_foep, 'rec_price': rec, 'qty': qty, 'days_to_sell': round(days_to_sell, 1), 'v_future': round(v_future, 2), 'action': action, 'gain_24h': round(gain if change else 0.0, 2),
            'cur': cur, 'best': best, 'reasons': reasons, 'min': lo, 'max': hi, 'lam': round(lam, 4)}


# ---------------------------------------------------------------- live state and loop (used by app.py)
class Shadow:
    def __init__(self, data_dir, append_gz, iso):
        self.data, self.append_gz, self.iso = data_dir, append_gz, iso
        self.lock = threading.Lock()
        self.offers = {}            # asin -> (t, {seller: price}, winner)
        self.listings = {}          # asin -> sku dict
        self.stock = {}             # sku -> fulfillable
        self.foep = {}              # sku -> (t, value)
        self.rival_stock = {}       # asin -> (t, units of competitive rivals)
        self.pending = collections.OrderedDict()
        self.last_decided = {}
        self.model = None
        self.on_decision = None
        self.tune_fn = None      # returns the tuning dict (live config), so settings live in one place
        self.recent = collections.deque(maxlen=2000)
        self.latest = {}         # asin -> latest decision
        self.stats = {'model_trained_at': None, 'decisions': 0, 'changes_suggested': 0, 'last_decision': None, 'last_error': None, 'by_action': {}}
        self.load_model()

    def model_path(self):
        return os.path.join(self.data, 'model', 'model.json')

    def load_model(self):
        try:
            with open(self.model_path()) as f:
                m = json.load(f)
            self.model = m
            self.stats['model_trained_at'] = m.get('trained_at')
            self.stats['model_info'] = {'validation': m.get('validation'), 'bb_rows': m.get('bb_rows'), 'messages': m.get('messages'),
                                        'window_days': m.get('window_days'), 'rivals': len(m.get('rivals') or {}), 'rival_default': m.get('rival_default'),
                                        'war_table': m.get('war_table')}
        except FileNotFoundError:
            self.model = None
        except Exception as e:
            self.stats['last_error'] = f'model load: {e}'

    def save_model(self, raw):
        m = json.loads(raw)
        for k in ('bb_foep', 'bb_nofoep', 'rivals', 'rival_default', 'war_table', 'trained_at'):
            if k not in m:
                raise ValueError('model missing ' + k)
        os.makedirs(os.path.dirname(self.model_path()), exist_ok=True)
        tmp = self.model_path() + '.tmp'
        with open(tmp, 'w') as f:
            f.write(raw if isinstance(raw, str) else raw.decode())
        os.replace(tmp, self.model_path())
        self.load_model()

    # --- feeds
    def on_message_bodies(self, bodies):
        for b in bodies:
            try:
                p = json.loads(b)['Payload']['AnyOfferChangedNotification']
            except Exception:
                continue
            a = p['OfferChangeTrigger']['ASIN']
            offs = {}
            win = None
            for o in p.get('Offers') or []:
                if o.get('SubCondition') != 'new' or not o.get('IsFulfilledByAmazon'):
                    continue
                pr = (o.get('ListingPrice') or {}).get('Amount', 0) + (o.get('Shipping') or {}).get('Amount', 0)
                offs[o['SellerId']] = pr
                if o.get('IsBuyBoxWinner'):
                    win = o['SellerId']
            with self.lock:
                self.offers[a] = (time.time(), offs, win)
                if ME in offs:
                    self.pending[a] = True
                    self.pending.move_to_end(a)

    def on_listings(self, rows):
        by = {}
        for r in rows:
            a = r.get('asin')
            if not a or not r.get('cost') or not r.get('listed_price'):
                continue
            if a in by and (by[a].get('qty') or 0) >= (r.get('qty') or 0):
                continue
            by[a] = r
        out = {}
        for a, r in by.items():
            lp, cost, roi = r['listed_price'], r['cost'], r.get('roi')
            ref = 0.08 if lp <= 10 else 0.15
            fees = lp - cost - (roi / 100.0 * cost) if roi is not None else 0.35 * lp
            v = r.get('v30') or 0.0
            bbs = r.get('avg_bb_share') if r.get('avg_bb_share') is not None else r.get('bb_share')
            bbs = clip((bbs or 25.0) / 100.0, 0.25, 1.0)  # conservative: at most 4x current sales if Nick won every Buy Box
            out[a] = {'sku': r['sku'], 'title': (r.get('title') or '')[:120], 'min': r.get('min_price'), 'max': r.get('max_price'), 'cost': cost,
                      'fixed_fee': max(0.0, fees - ref * lp), 'lam': max(v, 0.01) / 24.0 / bbs, 'qty': r.get('qty'), 'ss_repricing': bool(r.get('repricing'))}
        with self.lock:
            self.listings = out

    def on_inventory(self, rows):
        m = {x['sellerSku']: ((x.get('inventoryDetails') or {}).get('fulfillableQuantity') or 0) for x in rows}
        with self.lock:
            self.stock = m

    def on_foep(self, sku, body):
        try:
            r = (body.get('featuredOfferExpectedPriceResults') or [{}])[0]
            if r.get('resultStatus') != 'VALID_FOEP':
                return
            v = r['featuredOfferExpectedPrice']['listingPrice']['amount']
        except Exception:
            return
        with self.lock:
            self.foep[sku] = (time.time(), v)

    def on_keepa_products(self, products):
        for p in products or []:
            offs = []
            for o in p.get('offers') or []:
                if not o.get('isFBA') or o.get('condition') != 1:
                    continue
                q = o.get('offerCSV') or []
                s = o.get('stockCSV') or []
                if len(q) < 3 or q[-2] <= 0:
                    continue
                offs.append((o['sellerId'], (q[-2] + max(q[-1], 0)) / 100.0, s[-1] if s else None))
            if not offs:
                continue
            lo = min(x[1] for x in offs)
            riv = [x for x in offs if x[0] != ME and x[1] <= lo * 1.02]
            if riv and all(x[2] is not None for x in riv):
                with self.lock:
                    self.rival_stock[p['asin']] = (time.time(), float(sum(x[2] for x in riv)))

    # --- decisions
    def decide_one(self, asin):
        with self.lock:
            o = self.offers.get(asin)
            sku = self.listings.get(asin)
            if not o or not sku or not self.model:
                return None
            fo = self.foep.get(sku['sku'])
            st = self.stock.get(sku['sku'])
            rs = self.rival_stock.get(asin)
            model = self.model
        t, offs, win = o
        foep, age = (fo[1], time.time() - fo[0]) if fo else (None, None)
        rstock = rs[1] if (rs and time.time() - rs[0] < 8 * 3600) else None
        tune = self.tune_fn() if self.tune_fn else None
        return decide(model, asin, offs, win, sku, foep, age, st, rstock, tune=tune)

    def step(self):
        """Decide for ASINs that changed, at most once per DEBOUNCE_S each. Returns the number of decisions logged."""
        now = time.time()
        todo = []
        with self.lock:
            for a in list(self.pending.keys()):
                if now - self.last_decided.get(a, 0) >= DEBOUNCE_S:
                    todo.append(a)
                    del self.pending[a]
        lines = []
        for a in todo:
            d = self.decide_one(a)
            self.last_decided[a] = now
            if d is None:
                continue
            d['t'] = self.iso()
            d['ss_repricing'] = (self.listings.get(a) or {}).get('ss_repricing')
            if self.on_decision:
                try:
                    d['live'] = self.on_decision(d)
                except Exception as e:
                    d['live'] = {'error': f'{type(e).__name__}: {str(e)[:200]}'}
            self.recent.append(d)
            self.latest[a] = d
            lines.append(json.dumps(d))
            self.stats['decisions'] += 1
            self.stats['by_action'][d['action']] = self.stats['by_action'].get(d['action'], 0) + 1
            if d['action'] != 'hold':
                self.stats['changes_suggested'] += 1
        if lines:
            t = dt.datetime.now(dt.timezone.utc)
            self.append_gz(os.path.join(self.data, 'shadow', t.strftime('%Y-%m-%d'), t.strftime('%H') + '.jsonl.gz'), lines)
            self.stats['last_decision'] = self.iso()
        return len(lines)

    def loop(self):
        while True:
            try:
                self.step()
                self.stats['last_error'] = None
            except Exception as e:  # never let shadow mode disturb recording
                self.stats['last_error'] = f'{self.iso()} {type(e).__name__}: {str(e)[:200]}'
            time.sleep(10)
