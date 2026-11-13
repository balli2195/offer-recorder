"""Shadow-mode repricer. READ-ONLY: it computes the price it would set and logs the decision; it never calls any API
that changes a price or a Seller Snap setting.

Inputs (kept in memory, fed by the recorder's existing loops):
  offers     latest FBA new offers per ASIN from ANY_OFFER_CHANGED messages (seller -> landed price, Buy Box winner)
  listings   Seller Snap snapshot rows (sku, asin, min, max, cost, roi, v7, v30, avg_bb_share, repricing)
  inventory  fulfillable quantity per SKU
  foep       Amazon's featured offer expected price per SKU, with its time
  keepa      rival FBA stock per ASIN (latest Keepa pull)
  fees       Amazon's own fee estimate per SKU at a price in each fee band (fees.py)
Model: model.json written by tools/train.py (Buy Box logistic models, per-rival reaction rates, price-war penalty table).
"""
import json, math, os, statistics, threading, time, collections, datetime as dt
import fees as _fees
import demand as DM
import limits as L

ME = 'A2UYRB3AYUMGAJ'
HORIZON_H = 24.0
FOEP_MAX_AGE = 45 * 60
MIN_GAIN_ABS = 0.50      # $/day of expected profit before a change is recommended
MIN_GAIN_REL = 0.05      # or 5 %
DEBOUNCE_S = 120         # at most one decision per ASIN every 2 minutes
DECAY_PER_DAY = float(os.environ.get('SHADOW_DECAY_PER_DAY', '0.003'))  # cost of waiting to sell, as a share of price per day
PREP = float(os.environ.get('PREP_FEE', '1.40'))   # Nick's prep cost per unit, on every SKU; the Hub's limit settings override it
LIMIT_DEFAULTS = {'min_roi': 0.12, 'prep': 1.40, 'max_floor_mult': 1.25, 'max_new_mult': 2.0, 'max_history_days': 90}
FALLBACK_FBA = 4.11  # only until Amazon's fee for a SKU is loaded (median FBA fee in the Nov 12 sample)


# ---------------------------------------------------------------- features (shared with tools/train.py)
def clip(x, lo, hi):
    return max(lo, min(hi, x))


def feats(me, low_rival, n_rivals, foep=None, share=None, holder=None):
    """Feature dict for the Buy Box logistic model. Prices are landed prices; foep is Amazon's target for Nick (or None).
    holder (optional, Nov 13 test): who holds the Buy Box now, {'kind': 'me' | 'low' (a rival at the lowest price) | 'above' (a rival
    priced above the lowest) | 'other' (an FBM or backordered offer), 'price': the holder's landed price when kind is 'above'}."""
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
    if holder is not None:
        k = holder.get('kind')
        x.update({'h_me': 1.0 if k == 'me' else 0.0, 'h_other': 1.0 if k == 'other' else 0.0, 'h_above': 0.0, 'h_gap_neg': 0.0, 'h_gap_pos': 0.0})
        if k == 'above' and holder.get('price'):
            h = clip(100.0 * (me / holder['price'] - 1.0), -10, 20)
            x.update({'h_above': 1.0, 'h_gap_neg': max(min(h, 0.0), -5.0), 'h_gap_pos': max(h, 0.0)})
    return x


def p_win(model, x):
    key = 'bb_foep' if 't_ok' in x else 'bb_nofoep'
    m = model[key]
    z = sum(m['coef'].get(k, 0.0) * v for k, v in x.items())
    return 1.0 / (1.0 + math.exp(-clip(z, -30, 30)))


# ---------------------------------------------------------------- economics
def fee(p, sku):
    """Amazon's fee at price p: from Amazon's own estimates when loaded (sku['fees'] = bands), else 8%/15% referral + a typical FBA fee."""
    x = _fees.fee_at(sku.get('fees'), p)
    if x:
        return x
    ref = 0.08 if p <= 10 else 0.15
    return ref * p + (sku.get('fixed_fee') or FALLBACK_FBA), ref


def margin(p, sku):
    """Profit per unit sold at p: price - Amazon fees - cost - prep (Seller Snap's profit; ROI = this / cost)."""
    f, _ = fee(p, sku)
    return p - f - sku['cost'] - PREP


def rival_info(model, seller, asin=None):
    """(answer rate, response minutes, listing-specific?) for a rival. The seller's rate over all listings, or, when the model has the rival's
    record on this listing, that record pulled toward the seller's rate with the weight of `listing_weight` events (Nov 12 check)."""
    r = model['rivals'].get(seller)
    d = model['rival_default']
    rate, lag = (r['rate'], r.get('lag_min') or d['lag_min']) if r else (d['rate'], d['lag_min'])
    pl = (model.get('rival_listing') or {}).get(f'{seller}|{asin}') if asin else None
    if not pl or not pl.get('n'):
        return rate, lag, False
    m = model.get('listing_weight', 10)
    n = pl['n']
    rate = (pl['k'] + m * rate) / (n + m)
    if pl.get('lag_min'):
        lag = (pl['lag_min'] * n + lag * m) / (n + m)
    return rate, lag, True


def war_penalty_pct(model, rate):
    """Expected % drop in the listing low 24 h after an undercut, given the undercut rival's reactive rate."""
    tab = model['war_table']  # list of [rate_mid, pct_drop]
    if rate <= tab[0][0]:
        return tab[0][1]
    for (r0, d0), (r1, d1) in zip(tab, tab[1:]):
        if rate <= r1:
            return d0 + (d1 - d0) * (rate - r0) / (r1 - r0)
    return tab[-1][1]


def backordered(o):
    """An FBA offer Amazon shows as not yet available ("FUTURE_WITH_DATE": units inbound, in transfer or reserved).
    Nov 12 check: at the listing low, such offers held the Buy Box 13.1% of the time vs 32.7% for offers available now;
    Nick's own backordered offers 1% vs 25%."""
    return ((o.get('ShippingTime') or {}).get('AvailabilityType') or 'NOW') != 'NOW'


def fba_offers(n):
    """({seller: landed price} for FBA new offers available now, plus Nick's own offer whatever its availability,
    Buy Box winner of any fulfillment, Nick backordered?, number of backordered rival FBA offers left out).
    Backordered rivals are left out of the competition (and the 'lowest rival') because they rarely win the Buy Box:
    matching them would only push the price down. The same rule is used in tools/train.py."""
    offs, win, me_bo, n_bo = {}, None, False, 0
    for o in n.get('Offers') or []:
        if o.get('SubCondition') != 'new':
            continue
        if o.get('IsBuyBoxWinner'):
            win = o['SellerId']          # any fulfillment: a merchant-fulfilled offer can hold the Buy Box too
        if not o.get('IsFulfilledByAmazon'):
            continue
        bo = backordered(o)
        if o.get('SellerId') == ME:
            me_bo = bo
        elif bo:
            n_bo += 1
            continue
        offs[o['SellerId']] = (o.get('ListingPrice') or {}).get('Amount', 0) + (o.get('Shipping') or {}).get('Amount', 0)
    return offs, win, me_bo, n_bo


def suppression(n):
    """(suppressed, lowest new offer landed price, its channel) from one ANY_OFFER_CHANGED notification.
    Suppressed = Amazon shows no Buy Box: no Buy Box price in the summary and no offer marked winner (Nov 12 check: 211 of 214
    messages on listings the competitive summary showed without a featured offer; 3 of 5,154 on listings with one)."""
    s = n.get('Summary') or {}
    win = any(o.get('IsBuyBoxWinner') for o in n.get('Offers') or [])
    sup = not s.get('BuyBoxPrices') and not win
    lows = [(x['LandedPrice']['Amount'], x.get('FulfillmentChannel')) for x in s.get('LowestPrices') or []
            if x.get('Condition') == 'new' and (x.get('LandedPrice') or {}).get('Amount')]
    if not lows:
        lows = [((o.get('ListingPrice') or {}).get('Amount', 0) + (o.get('Shipping') or {}).get('Amount', 0), 'Amazon' if o.get('IsFulfilledByAmazon') else 'Merchant')
                for o in n.get('Offers') or [] if o.get('SubCondition') == 'new' and o.get('SellerId') != ME]
    lo = min(lows) if lows else (None, None)
    return sup, lo[0], lo[1]


def decide_suppressed(asin, offers, sku, low_any, low_ch, qty, tune=None, me_backordered=False):
    """Buy Box suppressed (Nick, Nov 12 2026): don't chase Amazon's target price; match the lowest offer on the listing
    (any fulfillment, landed price), within min/max, and let the usual step limits and brakes work from there."""
    me = offers[ME]
    lo, hi = sku['min'], sku['max']
    if not (lo and hi) or lo > hi or not low_any:
        return None
    rec = round(clip(low_any, lo, hi), 2)
    if me_backordered and rec < me:   # like Seller Snap: no cuts until the stock is available
        rec = me
    change = abs(rec - me) >= 0.01
    action = 'hold' if not change else ('match_low' if rec < me else 'raise')
    daily = min(float(qty or 0), max(sku.get('v30') or 0.0, 0.0))   # recent sales pace; the Buy Box model doesn't apply here
    def row(p):
        m = margin(p, sku)
        f_p, _ = fee(p, sku)
        return {'price': p, 'p_win': None, 'units_24h': round(daily, 3), 'fees': round(f_p, 2), 'margin': round(m, 2),
                'roi': round(m / sku['cost'], 4) if sku['cost'] else None, 'profit_24h': round(daily * m, 2), 'penalty': 0.0, 'exp_value_24h': None}
    cur, best = row(me), row(rec)
    ch = 'FBA' if low_ch == 'Amazon' else 'merchant-fulfilled' if low_ch == 'Merchant' else ''
    reasons = [f'Buy Box suppressed: matching the lowest offer {low_any:.2f}' + (f' ({ch})' if ch else '')]
    if me_backordered:
        reasons.append('your offer is backordered: no cuts until the stock is available')
    if low_any < lo - 0.005:
        reasons.append(f'lowest offer is below your min {lo:.2f}')
    if low_any > hi + 0.005:
        reasons.append(f'lowest offer is above your max {hi:.2f}')
    fees_src = 'amazon' if sku.get('fees') and _fees.fee_at(sku['fees'], me) else 'estimate'
    if fees_src != 'amazon':
        reasons.append("fees estimated (Amazon's fee for this SKU not loaded yet)")
    rivals = {s_: p for s_, p in offers.items() if s_ != ME}
    return {'asin': asin, 'sku': sku['sku'], 'my_price': me, 'low_rival': min(rivals.values()) if rivals else None, 'low_any': low_any,
            'n_rivals': len(rivals), 'winner': None, 'suppressed': True, 'foep': None, 'rec_price': rec if change else me, 'qty': qty,
            'days_to_sell': round(qty / daily, 1) if daily > 0 and qty else None, 'v_future': None, 'action': action, 'gain_24h': None,
            'cur': cur, 'best': best if change else cur, 'reasons': reasons, 'min': lo, 'max': hi, 'lam': round(sku['lam'], 4), 'backordered': me_backordered,
            'profit_day_now': cur['profit_24h'], 'profit_day_rec': (best if change else cur)['profit_24h'], 'roi_now': cur['roi'],
            'roi_rec': (best if change else cur)['roi'], 'fees_src': fees_src, 'decay': None}


def decide(model, asin, offers, winner, sku, foep=None, foep_age=None, my_stock=None, rival_stock=None, rival_floor_hit=None, tune=None,
           suppressed=False, low_any=None, low_ch=None, me_backordered=False, rivals_backordered=0, bb_price=None):
    """offers: {seller: landed price} for FBA new offers on the listing; winner: Buy Box seller id or None.
    sku: dict with sku, min, max, cost, fixed_fee, lam (units per Buy Box hour). Returns a decision dict or None.
    suppressed: Amazon shows no Buy Box on the listing; then the price matches the lowest offer instead (decide_suppressed)."""
    if ME not in offers:
        return None
    if suppressed:
        qty = my_stock if my_stock is not None else (sku.get('qty') or 0)
        return decide_suppressed(asin, offers, sku, low_any, low_ch, qty, tune, me_backordered)
    if len(offers) < 2:
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
    raised = None
    if hi and winner and winner != ME and bb_price and bb_price > hi + 0.005:   # a rival holds the Buy Box above the max: raise the max to it (Nick, Nov 12)
        raised, hi = hi, round(bb_price, 2)
    if not (lo and hi) or lo > hi:
        return None
    cands = {round(me, 2), round(low, 2), round(low - 0.01, 2)}
    if raised:
        cands.add(hi)
    if use_foep:  # small steps only when Amazon's target is known; without it the Buy Box model is much weaker (AUC ~0.72 vs ~0.89)
        cands |= {math.floor(use_foep * 100) / 100.0, round(me * 1.01, 2), round(me * 1.02, 2), round(me * 0.99, 2)}
    # who holds the Buy Box now (used only by a model trained with holder inputs: tools/train.py --holder)
    use_h = any(k.startswith('h_') for k in ((model.get('bb_foep' if use_foep else 'bb_nofoep') or {}).get('coef') or {}))
    if winner == ME:
        h_kind, h_price = 'me', None
    elif winner in rivals:
        h_price = rivals[winner]; h_kind = 'above' if h_price > low + 0.005 else 'low'
    else:
        h_kind, h_price = 'other', None
    def hold_at(p):
        if not use_h:
            return None
        if h_kind == 'me':
            return {'kind': 'me'} if p <= me + 0.005 else {'kind': 'low'}   # a raise isn't credited with keeping the Buy Box
        return {'kind': h_kind, 'price': h_price}
    if use_h and h_kind == 'above':   # a rival above the lowest holds it: its price (and a cent under) are worth scoring
        cands |= {round(h_price, 2), round(h_price - 0.01, 2)}
    if me <= low - 0.005:  # Nick alone at the low: the next offer minus a cent is a raise candidate
        cands |= {round(low - 0.01, 2)}
    cands = sorted({round(clip(c, lo, hi), 2) for c in cands})
    if me_backordered:   # Nick's units aren't available yet: don't compete on price (no cuts), as Seller Snap does; raises still allowed
        cands = [c for c in cands if c >= me - 0.005] or [round(me, 2)]
    lam = sku['lam']
    if sku.get('pace_day'):   # sales-model pace (demand.py): the units/day when priced level with the lowest rival; scale by the Buy Box chance at each price
        pw_ref = p_win(model, feats(low, low, n_riv, use_foep, share, hold_at(low)))
        lam = sku['pace_day'] / 24.0 / max(pw_ref, 0.25)
    rate = max((rival_info(model, s, asin)[0] for s in near), default=model['rival_default']['rate'])
    lag = min((rival_info(model, s, asin)[1] for s in near), default=model['rival_default']['lag_min'])
    p_now = p_win(model, feats(me, low, n_riv, use_foep, share, hold_at(me)))
    qty = my_stock if my_stock is not None else (sku.get('qty') or 1)
    qty = max(qty, 1)
    daily_now = lam * 24.0 * p_now
    days_to_sell = min(qty / max(daily_now, 0.05), 180.0)
    # What is already paid for is sunk: a unit sold now is worth its net proceeds; a unit kept is worth selling it later
    # near today's market low, less the cost of waiting (capital, storage, price erosion) over about half the sell-through time.
    f_low, ref_low = fee(low, sku)
    v_future = (low - f_low) * (1.0 - decay * days_to_sell / 2.0)
    exposed = min(qty, max(daily_now, 0.05) * 30.0)   # units whose future value a price war would lower
    rows = []
    for p in cands:
        m = margin(p, sku)
        f_p, ref = fee(p, sku)
        pw_alone = p_win(model, feats(p, low, n_riv, use_foep, share, hold_at(p)))
        pen = 0.0
        if p < low - 0.005 and p < me - 0.005:  # a new cut below the rivals: they may follow after their lag; then we are tied at p
            pw_tied = p_win(model, feats(p, p, n_riv, use_foep, share, hold_at(p)))
            fl = min(lag / 60.0 / HORIZON_H, 1.0)
            pw = (1 - rate) * pw_alone + rate * (fl * pw_alone + (1 - fl) * pw_tied)
            pen = war_penalty_pct(model, rate) / 100.0 * low * (1 - ref_low) * exposed
        else:
            pw = pw_alone
        units = min(lam * HORIZON_H * pw, qty)
        ev = units * ((p - f_p) - v_future) - pen
        rows.append({'price': p, 'p_win': round(pw, 4), 'units_24h': round(units, 3), 'fees': round(f_p, 2), 'margin': round(m, 2),
                     'roi': round(m / sku['cost'], 4) if sku['cost'] else None, 'profit_24h': round(units * m, 2),
                     'penalty': round(pen, 2), 'exp_value_24h': round(ev, 2)})
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
    elif h_kind == 'above':
        reasons.append(f'Buy Box held by {winner} at {h_price:.2f}, above the lowest rival')
    if use_foep:
        reasons.append(f'target {use_foep:.2f} ({(me / use_foep - 1) * 100:+.1f}% vs mine)')
    else:
        reasons.append('no fresh target price')
    reasons.append(f'lowest rival {low:.2f} ({low_s}), reactive rate {rate:.0%}')
    if share is not None:
        reasons.append(f'stock share {share:.0%}')
    if sku.get('pace_day'):
        reasons.append(f"sales pace {sku['pace_day']:.1f}/day level with the lowest rival (sales model, listing demand {sku.get('d2', 0):g}/day)")
    if lo > low:
        reasons.append('floor above lowest rival')
    if raised:
        reasons.append(f'max raised from {raised:.2f} to the Buy Box price {hi:.2f} held by a rival')
    if me_backordered:
        reasons.append('your offer is backordered: no cuts until the stock is available')
    if rivals_backordered:
        reasons.append(f'{rivals_backordered} backordered rival offer{"s" if rivals_backordered > 1 else ""} ignored')
    fees_src = 'amazon' if sku.get('fees') and _fees.fee_at(sku['fees'], me) else 'estimate'
    if fees_src != 'amazon':
        reasons.append("fees estimated (Amazon's fee for this SKU not loaded yet)")
    rec_row = best if change else cur
    return {'asin': asin, 'sku': sku['sku'], 'my_price': me, 'low_rival': low, 'n_rivals': n_riv, 'winner': winner, 'suppressed': False,
            'foep': use_foep, 'rec_price': rec, 'qty': qty, 'days_to_sell': round(days_to_sell, 1), 'v_future': round(v_future, 2), 'action': action, 'gain_24h': round(gain if change else 0.0, 2),
            'cur': cur, 'best': best, 'reasons': reasons, 'min': lo, 'max': hi, 'lam': round(lam, 4),
            'profit_day_now': cur['profit_24h'], 'profit_day_rec': rec_row['profit_24h'], 'roi_now': cur['roi'], 'roi_rec': rec_row['roi'],
            'fees_src': fees_src, 'decay': round(decay, 5), 'backordered': me_backordered, 'rivals_backordered': rivals_backordered,
            'pace_day': sku.get('pace_day'), 'd2': sku.get('d2')}


def _num(v):
    try:
        return float(v) if v not in (None, '') else None
    except (TypeError, ValueError):
        return None


def capital_return(rows):
    """Cost of waiting from Nick's own numbers: what a dollar tied up in inventory earns per day across the store.
    profit/day = sum over SKUs of 30-day units/day x Seller Snap profit per unit (ROI x cost; it already takes off fees and prep)
    capital    = sum of (on hand + inbound) x (cost + prep)
    A sale delayed by a day costs about this share of its proceeds, because that cash would otherwise be back in inventory earning it."""
    last = {}
    for r in rows:
        if r.get('sku'):
            last[r['sku']] = r
    prof = cap = cogs = 0.0
    for r in last.values():
        c = r.get('cost') or 0.0
        if c <= 0:
            continue
        v = r.get('v30') or 0.0
        if r.get('roi') is not None:
            prof += v * r['roi'] / 100.0 * c
        cap += ((r.get('qty') or 0) + (r.get('inbound') or 0)) * (c + PREP)
        cogs += v * (c + PREP)
    if cap <= 0 or cogs <= 0:
        return None
    return {'per_day': round(clip(prof / cap, 0.0005, 0.02), 5), 'profit_per_day': round(prof, 2), 'capital': round(cap, 2),
            'days_of_stock': round(cap / cogs, 1), 'roi_per_turn': round(prof / cogs, 4)}


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
        self.fees = {}              # sku -> {'t', 'bands'} from Amazon's fee estimates
        self.fba_median = FALLBACK_FBA
        self.capital_rate = None    # profit per day / money tied up in inventory (cost + prep, on hand and inbound)
        self.refs = {}              # asin -> (t, {CompetitivePrice, WasPrice, ...}) from the competitive summary sweep
        self.featured = {}          # asin -> (t, has a featured offer) from the same sweep
        self.sup_since = {}         # asin -> first time seen suppressed in this run
        self.ss_rows = {}           # sku -> latest Seller Snap row (every SKU, not one per ASIN)
        self.inv = {}               # sku -> {fulfillable, inbound, reserved, unfulfillable}
        self.catalog = {}           # sku -> Amazon all-listings report row {asin, title, status, channel, price, qty}
        self.catalog_t = None
        self.fyp = {}               # sku -> (status, reason) from Amazon's suppressed-listings report
        self.limits = {}            # from the Hub (POST /limits): settings, costs, overrides
        self.fba_high = {}          # asin -> (highest lowest-FBA price in the past year, days of FBA history), from Keepa
        self.traffic = {}           # asin -> {units, bb, pv} over the last 30 days (Amazon Sales & Traffic)
        self.demand = {}            # asin -> {d2 (listing units/day), bracket, rank, root, nvar, t} from Keepa (demand.py)
        self.orders = {}            # sku -> [(epoch, units)] over the last 8 days, from the recorder's 30-minute order pulls
        self.k_pace = None          # pooled scale of the sales-model pace, fitted on Nick's recent sales
        self.traffic_t = None
        self.load_state()
        self.pending = collections.OrderedDict()
        self.last_decided = {}
        self.model = None
        self.on_decision = None
        self.tune_fn = None      # returns the tuning dict (live config), so settings live in one place
        self.ab = None           # abtest.ABTest while the alternating-day test runs (set by app.py)
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
            self._overlay_rivals()
            self.stats['model_info'] = {'validation': m.get('validation'), 'bb_rows': m.get('bb_rows'), 'messages': m.get('messages'),
                                        'window_days': m.get('window_days'), 'rivals': len(m.get('rivals') or {}), 'rival_default': m.get('rival_default'),
                                        'war_table': m.get('war_table')}
        except FileNotFoundError:
            self.model = None
        except Exception as e:
            self.stats['last_error'] = f'model load: {e}'

    RIVAL_KEYS = ('rivals', 'rival_default', 'rival_listing', 'listing_weight', 'war_table')

    def rivals_path(self):
        return os.path.join(self.data, 'model', 'rivals.json')

    def _overlay_rivals(self):
        """The hourly rival refresh (rivals.json) replaces the model's rival tables when it is newer than the model's training."""
        try:
            with open(self.rivals_path()) as f:
                r = json.load(f)
        except (OSError, ValueError):
            return
        if self.model is None or (r.get('updated_at') or '') <= (self.model.get('trained_at') or ''):
            return
        for k in self.RIVAL_KEYS:
            if k in r:
                self.model[k] = r[k]
        self.stats['rivals_updated_at'] = r.get('updated_at')
        if self.stats.get('model_info'):
            self.stats['model_info'].update(rivals=len(r.get('rivals') or {}), rival_default=r.get('rival_default'), war_table=r.get('war_table'),
                                            rivals_updated_at=r.get('updated_at'))

    def on_rivals(self):
        """Called after the hourly refresh wrote rivals.json."""
        with self.lock:
            self._overlay_rivals()

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
            offs, win, me_bo, n_bo = fba_offers(p)
            sup, low_any, low_ch = suppression(p)
            bbp = [x for x in (p.get('Summary') or {}).get('BuyBoxPrices') or [] if (x.get('condition') or x.get('Condition') or '').lower() == 'new']
            bb_price = ((bbp[0].get('LandedPrice') or {}).get('Amount')) if bbp else None
            with self.lock:
                self.offers[a] = (time.time(), offs, win, {'suppressed': sup, 'low_any': low_any, 'low_ch': low_ch,
                                                           'me_backordered': me_bo, 'rivals_backordered': n_bo, 'bb_price': bb_price})
                if ME in offs:
                    self.pending[a] = True
                    self.pending.move_to_end(a)

    def on_listings(self, rows):
        """Seller Snap snapshot: kept only as a fallback (cost, min, sales) while the SKU has no data of Nick's own, and for its on/off flag."""
        ss = {r['sku']: r for r in rows if r.get('sku')}
        with self.lock:
            self.ss_rows = ss
        self.rebuild()

    # --- own data (no Seller Snap)
    def state_path(self, name):
        return os.path.join(self.data, 'limits', name)

    def load_state(self):
        for name, attr in (('limits.json', 'limits'), ('fba_high.json', 'fba_high'), ('traffic.json', 'traffic'), ('demand.json', 'demand')):
            try:
                with open(self.state_path(name)) as f:
                    v = json.load(f)
                if attr == 'fba_high':
                    v = {k: tuple(x) for k, x in v.items()}
                setattr(self, attr, v)
            except (FileNotFoundError, ValueError, OSError):
                pass

    def save_state(self, name, obj):
        try:
            os.makedirs(os.path.dirname(self.state_path(name)), exist_ok=True)
            tmp = self.state_path(name) + '.tmp'
            with open(tmp, 'w') as f:
                json.dump(obj, f)
            os.replace(tmp, self.state_path(name))
        except OSError:
            pass

    def on_limits(self, body):
        """From the Hub: limit settings, unit costs and min/max overrides."""
        lim = {'sent_at': body.get('sent_at'), 'settings': {**LIMIT_DEFAULTS, **(body.get('settings') or {})},
               'costs': body.get('costs') or {}, 'overrides': body.get('overrides') or {}}
        with self.lock:
            self.limits = lim
        self.save_state('limits.json', lim)
        self.rebuild()
        return {'costs': len(lim['costs'].get('sku') or {}), 'overrides': len(lim['overrides'])}

    def on_orders(self, by_sku):
        """by_sku: {sku: [(epoch, units)]} for the last 8 days (cancelled orders left out)."""
        with self.lock:
            self.orders = {k: list(v) for k, v in by_sku.items()}
        self.rebuild()

    def units_since(self, sku, t):
        with self.lock:
            return sum(q for tt, q in self.orders.get(sku) or [] if tt >= t)

    def on_traffic(self, by_asin):
        with self.lock:
            self.traffic = by_asin
            self.traffic_t = time.time()
        self.save_state('traffic.json', by_asin)
        self.rebuild()

    def rebuild(self):
        """The priced listings, one SKU per ASIN, from Nick's own data: Amazon's listing report and FBA inventory for the SKUs,
        the Hub for cost and overrides, Amazon's fees for the 12% ROI min, Keepa's year of FBA prices for the max, and Amazon's
        Sales & Traffic for sales speed and Buy Box share. Seller Snap only fills gaps (and says so in each field's source)."""
        global PREP
        with self.lock:
            cat, inv, ss, lim = dict(self.catalog), dict(self.inv), dict(self.ss_rows), dict(self.limits)
            fees_by, fba_med, fh, tr = dict(self.fees), self.fba_median, dict(self.fba_high), dict(self.traffic)
            dmd, orders = dict(self.demand), dict(self.orders)
        st = {**LIMIT_DEFAULTS, **(lim.get('settings') or {})}
        PREP = float(st['prep'])
        ov, costs = lim.get('overrides') or {}, lim.get('costs') or {}
        skus = (set(cat) | set(inv)) if cat else set(ss)
        best = {}
        for sku in skus:
            c, iv, r = cat.get(sku), inv.get(sku), ss.get(sku)
            if c and c.get('channel') == 'MFN':
                continue
            asin = (c or {}).get('asin') or (r or {}).get('asin')
            if not asin:
                continue
            fu = (iv or {}).get('fulfillable')
            if fu is None:   # Seller Snap's count only before Amazon's inventory has loaded: it lags, and kept 7 empty inactive SKUs priced on Nov 12
                fu = 0 if (inv and cat) else ((r or {}).get('qty') or 0)
            tot = fu + (iv or {}).get('reserved', 0) + (iv or {}).get('inbound', 0)
            if tot <= 0:
                continue
            if asin not in best or (fu, tot) > best[asin][0]:
                best[asin] = ((fu, tot), sku)
        out, miss, src = {}, collections.Counter(), collections.Counter()
        for asin, (_, sku) in best.items():
            c, iv, r = cat.get(sku) or {}, inv.get(sku) or {}, ss.get(sku) or {}
            cost, csrc = L.resolve_cost(sku, asin, costs, r.get('cost'))
            bands = (fees_by.get(sku) or {}).get('bands')
            o = ov.get(sku) or {}
            if o.get('min') is not None:
                mn, nsrc = float(o['min']), (o.get('source') or 'override') + ' min'
            elif cost:
                mn, nsrc = L.rule_min(cost, bands, PREP, st['min_roi'], fba_med), f"{st['min_roi']:.0%} ROI"
            elif r.get('min_price'):
                mn, nsrc = r['min_price'], 'Seller Snap'
            else:
                mn, nsrc = None, None
            if o.get('max') is not None:
                mx, xsrc = float(o['max']), (o.get('source') or 'override') + ' max'
            else:
                mx, xsrc = L.resolve_max(mn, fh.get(asin), st)
            if mn and mx and mx < mn:
                mx, xsrc = mn, 'min (max was below it)'
            if tr:
                t = tr.get(asin) or {}
                v30, bbs, dsrc = (t.get('units') or 0) / 30.0, t.get('bb'), 'Amazon sales & traffic'
            else:
                v30 = r.get('v30') or 0.0
                bbs = r.get('avg_bb_share') if r.get('avg_bb_share') is not None else r.get('bb_share')
                dsrc = 'Seller Snap' if r else None
            bbs = clip((bbs if bbs is not None else 25.0) / 100.0, 0.25, 1.0)  # conservative: at most 4x current sales if Nick won every Buy Box
            if not (cost and mn and mx):
                miss['no cost' if not cost else 'no min/max'] += 1
                continue
            for k, v in (('cost', csrc), ('min', nsrc), ('max', xsrc), ('sales', dsrc)):
                src[f'{k}: {v}'] += 1
            out[asin] = {'sku': sku, 'title': (c.get('title') or r.get('title') or '')[:120], 'min': round(mn, 2), 'max': round(mx, 2), 'cost': round(cost, 4),
                         'listed': c.get('price') or r.get('listed_price'), 'v30': round(v30, 4), 'lam': max(v30, 0.01) / 24.0 / bbs,
                         'qty': iv.get('fulfillable', r.get('qty')), 'ss_repricing': bool(r.get('repricing')) if r else False,
                         'src': {'cost': csrc, 'min': nsrc, 'max': xsrc, 'sales': dsrc},
                         'ss_min': r.get('min_price'), 'ss_max': r.get('max_price'), 'ss_cost': r.get('cost'),
                         'year_high': (fh.get(asin) or (None, None))[0], 'year_days': (fh.get(asin) or (None, None))[1],
                         'year_fba_high': (tuple(fh.get(asin) or ()) + (None,) * 4)[2], 'year_bb_high': (tuple(fh.get(asin) or ()) + (None,) * 4)[3]}
        self.set_pace(out, inv, dmd, orders)
        rate = self.own_capital_return(out, inv, cat, costs, fees_by, fba_med) if (tr and costs) else capital_return(list(ss.values()))
        with self.lock:
            self.listings = out
            if rate:
                self.capital_rate = rate['per_day']
                self.stats['capital'] = rate
            self.stats['limits'] = {'priced': len(out), 'skipped': dict(miss), 'sources': dict(src), 'hub_sent_at': lim.get('sent_at'),
                                    'year_highs': len(fh), 'traffic_asins': len(tr), 'settings': st}

    def set_pace(self, out, inv, dmd, orders, now=None):
        """Sales pace per listing the way Nick's sales model does it (demand.py): k x g(stock / D2) x sqrt(D2) x tier, with k fitted on
        his own recent sales across SKUs, then each SKU's own sales blended in. Own sales: the last 7 days of orders once the 30-minute
        order pulls have loaded, else Amazon's 30-day Sales & Traffic units. Listings without a Keepa D2 keep the old pace (lam)."""
        now = now or time.time()
        use_orders = bool(orders)
        rows, own = [], {}
        for asin, x in out.items():
            d = (dmd.get(asin) or {}).get('d2')
            iv = inv.get(x['sku']) or {}
            S = (iv.get('fulfillable') or 0) + (iv.get('reserved') or 0) + (iv.get('inbound') or 0)
            if not S:
                S = x.get('qty') or 0
            if use_orders:
                u, days = sum(q for t, q in orders.get(x['sku']) or [] if t >= now - 7 * 86400), 7.0
            else:
                u, days = x['v30'] * 30.0, 30.0
            own[asin] = (d, S, u, days)
            if d and S > 0 and u > 0:
                rows.append((u / days, S, d))
        k = DM.fit_k(rows) if len(rows) >= 20 else None
        with self.lock:
            self.k_pace = k
            self.stats['pace'] = {'k': round(k, 4) if k else None, 'fit_skus': len(rows), 'own_sales': '7-day orders' if use_orders else '30-day Sales & Traffic',
                                  'with_d2': sum(1 for v in own.values() if v[0])}
        if not k:
            return
        for asin, x in out.items():
            d, S, u, days = own[asin]
            if not d:
                continue
            prior = k * DM.shape(max(S, 1), d)
            x['pace_day'] = round(DM.pace(k, S, d, u, days), 4)
            x['pace_prior'] = round(prior, 4)
            x['d2'] = d
            x['src']['pace'] = f"sales model (D2 {d:g}/day) + {'7-day orders' if use_orders else '30-day sales'}"

    def own_capital_return(self, listings, inv, cat, costs, fees_by, fba_med):
        """Cost of waiting from Nick's own numbers: profit a day (30-day units x profit per unit at the current price) over the
        money in inventory (on hand + reserved + inbound, at cost + prep)."""
        prof = cap = cogs = 0.0
        for a, x in listings.items():
            p = x.get('listed') or x['min']
            sk = {'cost': x['cost'], 'fees': (fees_by.get(x['sku']) or {}).get('bands'), 'fixed_fee': fba_med}
            prof += x['v30'] * margin(p, sk)
            cogs += x['v30'] * (x['cost'] + PREP)
        for sku, iv in inv.items():
            q = iv.get('fulfillable', 0) + iv.get('reserved', 0) + iv.get('inbound', 0)
            if q <= 0:
                continue
            cost, _ = L.resolve_cost(sku, (cat.get(sku) or {}).get('asin'), costs)
            if cost:
                cap += q * (cost + PREP)
        if cap <= 0 or cogs <= 0:
            return None
        return {'per_day': round(clip(prof / cap, 0.0005, 0.02), 5), 'profit_per_day': round(prof, 2), 'capital': round(cap, 2),
                'days_of_stock': round(cap / cogs, 1), 'roi_per_turn': round(prof / cogs, 4), 'source': 'own data'}

    def limits_check(self):
        """Own min/max next to Seller Snap's, for the switch-over check."""
        with self.lock:
            L_ = dict(self.listings)
        rows, d_min, d_max = [], [], []
        for a, x in L_.items():
            if not (x.get('ss_min') and x.get('ss_max')):
                continue
            dm = x['min'] / x['ss_min'] - 1
            dx = x['max'] / x['ss_max'] - 1
            d_min.append(dm); d_max.append(dx)
            rows.append({'asin': a, 'sku': x['sku'], 'title': x['title'], 'min': x['min'], 'ss_min': x['ss_min'], 'min_diff': round(dm, 4),
                         'max': x['max'], 'ss_max': x['ss_max'], 'max_diff': round(dx, 4), 'cost': x['cost'], 'ss_cost': x.get('ss_cost'),
                         'src': x['src'], 'year_high': x.get('year_high'), 'year_days': x.get('year_days'),
                         'year_fba_high': x.get('year_fba_high'), 'year_bb_high': x.get('year_bb_high')})
        q = lambda v, k: sorted(v)[int(k * (len(v) - 1))] if v else None
        summ = {'compared': len(rows), 'min_within_5pct': sum(1 for v in d_min if abs(v) <= 0.05), 'max_within_5pct': sum(1 for v in d_max if abs(v) <= 0.05),
                'min_diff_median': q(d_min, .5), 'max_diff_median': q(d_max, .5)}
        return {'summary': summ, 'rows': sorted(rows, key=lambda r: -abs(r['min_diff']))}

    def on_fees(self, by_sku):
        f = [b['bands']['mid']['fba'] for b in by_sku.values() if (b.get('bands') or {}).get('mid')]
        with self.lock:
            self.fees = by_sku
            if f:
                self.fba_median = statistics.median(f)
            self.stats['fees_known'] = len(by_sku)
        self.rebuild()

    def on_competitive_summary(self, asin, body):
        """One ASIN's competitive summary (segments sweep): reference prices and whether any featured offer exists."""
        refs = {x.get('name'): (x.get('price') or {}).get('amount') for x in body.get('referencePrices') or [] if x.get('name')}
        has = any(s_.get('sellerId') for o in body.get('featuredBuyingOptions') or [] for s_ in o.get('segmentedFeaturedOffers') or [])
        now = time.time()
        with self.lock:
            if refs:
                self.refs[asin] = (now, refs)
            if 'featuredBuyingOptions' in body:
                self.featured[asin] = (now, has)
                if not has:
                    self.sup_since.setdefault(asin, now)
                else:
                    self.sup_since.pop(asin, None)

    def breakeven(self, sku):
        """Lowest price (to the cent) at which a unit sold makes no loss after Amazon fees, cost and prep."""
        p = round(max(sku['cost'] * 0.5, 0.5), 2)
        hi = sku['cost'] * 4 + 40
        while p < hi:
            if margin(p, sku) >= 0:
                return round(p, 2)
            p = round(p + 0.01, 2) if p < 12 else round(p * 1.002 + 0.01, 2)
        return None

    def suppressed_list(self):
        """Listings with the Buy Box suppressed now (latest sweep shows no featured offer, or the latest offer-change
        message says so), with what it would take to get it back."""
        now = time.time()
        with self.lock:
            ids = {a for a, (t, has) in self.featured.items() if not has and now - t < 3 * 3600}
            ids |= {a for a, d in self.latest.items() if d.get('suppressed') and (a not in self.featured or not self.featured[a][1])}
            items = [(a, self.listings.get(a), self.latest.get(a), self.refs.get(a), self.sup_since.get(a)) for a in ids]
            fees_by, stock, foep, fba_med = dict(self.fees), dict(self.stock), dict(self.foep), self.fba_median
        out = []
        for a, sku, d, rf, since in items:
            if not sku:
                continue
            sk = dict(sku, fees=(fees_by.get(sku['sku']) or {}).get('bands'), fixed_fee=fba_med)
            qty = stock.get(sku['sku'])
            qty = qty if qty is not None else (sku.get('qty') or 0)
            if not qty:
                continue
            refs = (rf or (None, {}))[1]
            cp = refs.get('CompetitivePrice')
            fo = foep.get(sku['sku'])
            target = fo[1] if fo and now - fo[0] < 3 * 3600 else None
            be = self.breakeven(sk)
            bar = cp if cp is not None else target
            if bar is None:
                status = 'no competitive price from Amazon yet'
            elif sku['min'] and bar >= sku['min'] - 0.005:
                status = 'at or above your min: the repricer can win it back'
            elif be is not None and bar >= be:
                status = 'below your min, still profitable: lower the min to win it back'
            else:
                status = 'below your break-even: wait for it to rise, or liquidate'
            out.append({'asin': a, 'sku': sku['sku'], 'title': sku.get('title'), 'my_price': (d or {}).get('my_price') or sku.get('listed'),
                        'low_any': (d or {}).get('low_any'), 'competitive_price': cp, 'was_price': refs.get('WasPrice'), 'target': target,
                        'min': sku['min'], 'max': sku['max'], 'cost': sku['cost'], 'breakeven': be, 'qty': qty,
                        'inventory': round(qty * (sku['cost'] + PREP), 2), 'sales_per_day': sku.get('v30'), 'status': status,
                        'since': dt.datetime.fromtimestamp(since, dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ') if since else None,
                        'decision': {k: (d or {}).get(k) for k in ('t', 'action', 'rec_price', 'backordered')} if d else None})
        return sorted(out, key=lambda r: -r['inventory'])

    def on_inventory(self, rows):
        m, det = {}, {}
        for x in rows:
            d = x.get('inventoryDetails') or {}
            m[x['sellerSku']] = d.get('fulfillableQuantity') or 0
            det[x['sellerSku']] = {'fulfillable': d.get('fulfillableQuantity') or 0,
                                   'inbound': (d.get('inboundWorkingQuantity') or 0) + (d.get('inboundShippedQuantity') or 0) + (d.get('inboundReceivingQuantity') or 0),
                                   'reserved': (d.get('reservedQuantity') or {}).get('totalReservedQuantity') or 0,
                                   'unfulfillable': (d.get('unfulfillableQuantity') or {}).get('totalUnfulfillableQuantity') or 0}
        with self.lock:
            self.stock = m
            self.inv = det
        self.rebuild()

    def on_catalog(self, rows):
        """Amazon's all-listings report (every SKU on the account, with Active / Inactive / Incomplete)."""
        c = {}
        for r in rows:
            sku = (r.get('seller-sku') or '').strip()
            if not sku:
                continue
            c[sku] = {'asin': (r.get('asin1') or '').strip(), 'title': (r.get('item-name') or r.get('\ufeffitem-name') or '').strip()[:160],
                      'status': (r.get('status') or '').strip(), 'channel': 'FBA' if (r.get('fulfillment-channel') or '').startswith('AMAZON') else 'MFN',
                      'price': _num(r.get('price')), 'qty': _num(r.get('quantity'))}
        with self.lock:
            self.catalog = c
            self.catalog_t = time.time()
        self.rebuild()

    def on_fyp(self, rows):
        """Amazon's suppressed-listings report: listings Amazon hides from search (missing info, etc.)."""
        with self.lock:
            self.fyp = {(r.get('SKU') or r.get('sku') or '').strip(): ((r.get('Status') or '').strip(), (r.get('Reason') or '').strip())
                        for r in rows if (r.get('SKU') or r.get('sku'))}

    def listing_state(self, sku, ss=None, cat=None, inv=None):
        """Seller Snap-style listing state: active, out_of_stock, deactivated or archived.
        archived     no longer tracked on one side: on Amazon's listing report but not in Seller Snap, or the other way round
        out_of_stock no sellable units
        deactivated  sellable units, but Amazon shows the listing Inactive / Incomplete or hides it from search
        active       sellable units and Amazon shows the listing Active"""
        if (self.catalog and cat is None) or (self.ss_rows and ss is None):
            return 'archived'
        fu = (inv or {}).get('fulfillable')
        if fu is None:
            fu = (ss or {}).get('qty') or (cat or {}).get('qty') or 0
        if fu <= 0:
            return 'out_of_stock'
        st = (cat or {}).get('status')
        if (st and st != 'Active') or sku in self.fyp:
            return 'deactivated'
        return 'active'

    def table(self, live_skus=()):
        """One row per SKU (Seller Snap and Amazon's listing report together), with state, stock, Seller Snap settings and
        the repricer's latest decision, for the Hub's Repricer page."""
        with self.lock:
            ss_rows, cat, inv, latest, listings = dict(self.ss_rows), dict(self.catalog), dict(self.inv), dict(self.latest), dict(self.listings)
            refs = dict(self.refs)
            tr_ = dict(self.traffic)
        live = set(live_skus)
        priced = {v['sku']: a for a, v in listings.items()}
        out = []
        for sku in sorted(set(ss_rows) | set(cat)):
            r, c, iv = ss_rows.get(sku), cat.get(sku), inv.get(sku)
            asin = (r or {}).get('asin') or (c or {}).get('asin')
            a_pr = priced.get(sku)
            d = latest.get(a_pr) if a_pr else None
            pl = listings.get(a_pr) if a_pr else None
            bb = (r or {}).get('avg_bb_share')
            out.append({'sku': sku, 'asin': asin, 'title': (r or {}).get('title') or (c or {}).get('title'),
                        'state': self.listing_state(sku, r, c, iv), 'amazon_status': (c or {}).get('status'), 'channel': (c or {}).get('channel'),
                        'search_suppressed': (self.fyp.get(sku) or (None, None))[1] or None,
                        'in_seller_snap': r is not None, 'ss_repricing': (r or {}).get('repricing'), 'preset': (r or {}).get('preset'),
                        'price': (c or {}).get('price') or (r or {}).get('listed_price'),
                        'min': (pl or {}).get('min'), 'max': (pl or {}).get('max'), 'cost': (pl or {}).get('cost'), 'limit_src': (pl or {}).get('src'),
                        'ss_min': (r or {}).get('min_price'), 'ss_max': (r or {}).get('max_price'),
                        'roi': (r or {}).get('roi'), 'bb_share': (tr_.get(asin) or {}).get('bb') if tr_ else (bb if bb is not None else (r or {}).get('bb_share')),
                        'v30': round(((tr_.get(asin) or {}).get('units') or 0) / 30.0, 2) if tr_ else (r or {}).get('v30'), 'fulfillable': (iv or {}).get('fulfillable', 0), 'inbound': (iv or {}).get('inbound', 0),
                        'reserved': (iv or {}).get('reserved', 0), 'live': sku in live, 'priced': a_pr is not None,
                        'priced_sku': (listings.get(asin) or {}).get('sku') if asin else None,
                        'competitive_price': ((refs.get(asin) or (None, {}))[1] or {}).get('CompetitivePrice') if asin else None,
                        'decision': {k: d.get(k) for k in ('t', 'action', 'my_price', 'rec_price', 'low_rival', 'low_any', 'foep', 'gain_24h',
                                                          'profit_day_now', 'profit_day_rec', 'suppressed', 'backordered')} if d else None})
        # sparse: leave out empty fields (6,000 rows; most are out of stock with little to show)
        zero_ok = ('fulfillable', 'inbound', 'reserved')   # 0 is the page default for these; elsewhere 0 means something (0% Buy Box)
        return [{k: v for k, v in r.items() if not (v is None or v is False or v == '' or (k in zero_ok and v == 0))} for r in out]

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
        dm = {}
        for p in products or []:
            x = DM.from_keepa(p)
            if x:
                dm[p['asin']] = dict(x, t=time.time())
        if dm:
            with self.lock:
                self.demand.update(dm)
                d = dict(self.demand)
            self.save_state('demand.json', d)
        highs = {}
        for p in products or []:
            cs = p.get('csv') or []
            f, b = (cs[10] if len(cs) > 10 else None), (cs[18] if len(cs) > 18 else None)
            long_f = bool(f and len(f) >= 4 and f[-2] - f[0] > 30 * 1440)
            long_b = bool(b and len(b) >= 6 and b[-3] - b[0] > 30 * 1440)
            if long_f or long_b:   # only a pull with a long history (the daily year pull)
                h = L.fba_year_high(p, now_min=int(time.time() / 60) - L.KEEPA_EPOCH_MIN)
                if h:
                    highs[p['asin']] = h
        if highs:
            with self.lock:
                self.fba_high.update(highs)
                fh = dict(self.fba_high)
            self.save_state('fba_high.json', fh)
            self.rebuild()
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
            sku = dict(sku, fees=(self.fees.get(sku['sku']) or {}).get('bands'), fixed_fee=self.fba_median)
            ab = self.ab if (self.ab and self.ab.in_test(sku['sku'])) else None
            if ab and ab.cfg.get('limits') == 'sellersnap' and sku.get('ss_min') and sku.get('ss_max'):
                # alternating-day test: both repricers use Seller Snap's min/max (Nick, Nov 12)
                sku.update(min=round(float(sku['ss_min']), 2), max=round(float(sku['ss_max']), 2))
            fo = self.foep.get(sku['sku'])
            crate = self.capital_rate
            st = self.stock.get(sku['sku'])
            rs = self.rival_stock.get(asin)
            model = self.model
        t, offs, win, ex = o if len(o) == 4 else (*o, {})
        foep, age = (fo[1], time.time() - fo[0]) if fo else (None, None)
        rstock = rs[1] if (rs and time.time() - rs[0] < 8 * 3600) else None
        tune = dict(self.tune_fn() or {}) if self.tune_fn else {}
        if tune.get('decay_auto', True) and crate:
            tune['decay_per_day'] = crate
        return decide(model, asin, offs, win, sku, foep, age, st, rstock, tune=tune,
                      suppressed=ex.get('suppressed', False), low_any=ex.get('low_any'), low_ch=ex.get('low_ch'),
                      me_backordered=ex.get('me_backordered', False), rivals_backordered=ex.get('rivals_backordered', 0),
                      bb_price=None if ab else ex.get('bb_price'))   # in the test, no raising the max to a rival's Buy Box (Seller Snap doesn't)

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
            if self.ab:
                arm = self.ab.arm(d['sku'], now)
                if arm:
                    d['arm'] = arm; d['ab_group'] = self.ab.cfg['groups'].get(d['sku']); d['ab_day'] = self.ab.day_index(now)
                    d['washout'] = self.ab.washout(now); d['ab_dry_run'] = self.ab.cfg['dry_run']
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
