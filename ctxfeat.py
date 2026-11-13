"""Context inputs for the Buy Box boosted trees (Nick, Nov 13 2026): everything about a moment except Nick's own price.

The trees start from the logistic models' log-odds and add a correction from these inputs only, so the predicted share still moves
smoothly with price (tested Nov 13: BUYBOX_BOOSTED_TREE_TEST and BUYBOX_HEAD_TO_HEAD docs). Training (tools/train.py) and the live
repricer (shadow.py) both build them with this module, from the same offer-change messages.

From the latest offer message on the listing in which Nick has an offer (MSG_COLS), plus rolling and timing inputs (ROLL_COLS),
plus who holds the Buy Box (HOLD_COLS). Missing values are NaN (the trees handle them).
"""
import collections, datetime as dt, math
from zoneinfo import ZoneInfo
import numpy as np

ME = 'A2UYRB3AYUMGAJ'; AMAZON = 'ATVPDKIKX0DER'
CT = ZoneInfo('America/Chicago')
MSG_COLS = ['log_rank', 'cat_rank_log', 'n_offers', 'n_bb_eligible', 'amazon_on', 'cpt_ratio', 'list_ratio',
            'low_fb_pct', 'low_fb_log', 'low_featured', 'near_n', 'near_min_fb_log', 'near_featured_share',
            'fbm_ratio', 'fbm_prime', 'hold_fb_pct', 'hold_fb_log', 'hold_fba', 'hold_is_low', 'n_winners']
ROLL_COLS = ['msgs_6h', 'low_var_24h', 'hour_ct', 'weekday', 'n_rivals']
HOLD_COLS = ['hold_me', 'hold_above', 'hold_other']
FEATURES = MSG_COLS + ROLL_COLS + HOLD_COLS
NAN = float('nan')


def _land(o):
    return (o.get('ListingPrice') or {}).get('Amount', 0) + (o.get('Shipping') or {}).get('Amount', 0)


def _bo(o):
    return ((o.get('ShippingTime') or {}).get('AvailabilityType') or '') == 'FUTURE_WITH_DATE'


def _fb(o):
    return o.get('SellerFeedbackRating') or {}


def message_ctx(p):
    """(MSG_COLS values, lowest available FBA rival landed price) from one AnyOfferChangedNotification payload, or None when Nick has
    no offer on it or no available FBA rival competes."""
    offs = [o for o in p.get('Offers') or [] if (o.get('SubCondition') or '').lower() == 'new']
    if not any(o.get('SellerId') == ME for o in offs):
        return None
    fba = [o for o in offs if o.get('IsFulfilledByAmazon') and o.get('SellerId') != ME and not _bo(o)]
    if not fba:
        return None
    L = min(_land(o) for o in fba)
    if L <= 0:
        return None
    low = min(fba, key=_land)
    near = [o for o in fba if _land(o) <= L * 1.01]
    fbm = [o for o in offs if not o.get('IsFulfilledByAmazon') and o.get('SellerId') != ME]
    sm = p.get('Summary') or {}
    rr = [r.get('Rank') for r in sm.get('SalesRankings') or [] if r.get('Rank')]
    nof = sum(x.get('OfferCount', 0) for x in sm.get('NumberOfOffers') or [] if (x.get('Condition') or '').lower() == 'new')
    nbb = sum(x.get('OfferCount', 0) for x in sm.get('NumberOfBuyBoxEligibleOffers') or [] if (x.get('Condition') or '').lower() == 'new')
    cpt = (sm.get('CompetitivePriceThreshold') or {}).get('Amount')
    lp = (sm.get('ListPrice') or {}).get('Amount')
    wins = [o for o in offs if o.get('IsBuyBoxWinner')]
    hold = next((o for o in wins if o.get('SellerId') != ME), None) if not any(o.get('SellerId') == ME for o in wins) else None
    row = [math.log10(rr[0]) if rr else NAN, math.log10(rr[-1]) if len(rr) > 1 else NAN, float(nof), float(nbb),
           1.0 if any(o.get('SellerId') == AMAZON for o in offs) else 0.0, cpt / L if cpt else NAN, lp / L if lp else NAN,
           float(_fb(low).get('SellerPositiveFeedbackRating', NAN)), math.log10(1 + (_fb(low).get('FeedbackCount') or 0)),
           1.0 if low.get('IsFeaturedMerchant') else 0.0, float(len(near)),
           min(math.log10(1 + (_fb(o).get('FeedbackCount') or 0)) for o in near), sum(1 for o in near if o.get('IsFeaturedMerchant')) / len(near),
           (min(_land(o) for o in fbm) / L) if fbm else NAN, 1.0 if any((o.get('PrimeInformation') or {}).get('IsOfferPrime') for o in fbm) else 0.0,
           float(_fb(hold).get('SellerPositiveFeedbackRating', NAN)) if hold else NAN,
           math.log10(1 + (_fb(hold).get('FeedbackCount') or 0)) if hold else NAN,
           (1.0 if hold.get('IsFulfilledByAmazon') else 0.0) if hold else NAN,
           (1.0 if hold.get('SellerId') == low.get('SellerId') else 0.0) if hold else NAN, float(len(wins))]
    return row, float(L)


def hold_cols(hk):
    return [1.0 if hk == 'me' else 0.0, 1.0 if hk == 'above' else 0.0, 1.0 if hk == 'other' else 0.0]


def vector(msg_row, n_msgs_6h, lows_24h, t, n_rivals, hk):
    """The full FEATURES vector: the latest message's values, messages on the listing in the last 6 h, the variation of the lowest
    rival price over 24 h (std / mean), hour and weekday in Central time at t, number of rivals, and who holds the Buy Box."""
    h = dt.datetime.fromtimestamp(t, CT)
    w = np.asarray(lows_24h, dtype=float)
    var = float(np.std(w) / np.mean(w)) if len(w) > 1 and np.mean(w) > 0 else 0.0
    return list(msg_row) + [float(n_msgs_6h), var, float(h.hour), float(h.weekday()), float(n_rivals)] + hold_cols(hk)


def vector_at(tt, M, L, i, t, n_rivals, hk):
    """vector() from one listing's sorted arrays (times tt, MSG rows M, lowest rival L) at message index i (training)."""
    if i < 0:
        return None
    j6 = int(np.searchsorted(tt, t - 6 * 3600, side='left')); j24 = int(np.searchsorted(tt, t - 86400, side='left'))
    return vector(M[i], i - j6 + 1, L[j24:i + 1], t, n_rivals, hk)


class Live:
    """Per-listing context for the repricer: the latest message's values and 24 h of (time, lowest rival) per ASIN."""
    def __init__(self):
        self.last = {}
        self.hist = collections.defaultdict(collections.deque)

    def add(self, asin, t, payload):
        x = message_ctx(payload)
        if x is None:
            return
        self.last[asin] = x[0]
        q = self.hist[asin]
        q.append((t, x[1]))
        while q and q[0][0] < t - 86400:
            q.popleft()

    def vector(self, asin, t, n_rivals, hk):
        row = self.last.get(asin)
        if row is None:
            return None
        q = self.hist[asin]
        return vector(row, sum(1 for a, _ in q if a >= t - 6 * 3600), [l for a, l in q if a >= t - 86400], t, n_rivals, hk)
