"""How fast a SKU sells when priced competitively, the same way Nick's buy-quantity sales model does it (Nick, Nov 13 2026).

From the sales model v4 candidate (claude/sales-model-v2/CANDIDATE_V4_AND_CONFIRMATION.md, bq_v4_candidate.json on the Lead Score server):
  D2    continuous listing demand (units/day): Keepa's "bought in past month" bracket (a floor), placed inside
        [bracket, min(next bracket, 2 x bracket)] by a per-root-category sales-rank -> monthly-sold line (variations adjust it);
        the rank estimate, capped at 147/month, when Keepa shows no bracket.
  g     stock-depth curve: pace rises with Nick's stock relative to demand, g(d) = (d/7)^0.7 / (1 + (d/60)^0.7), = 1 at 7 days.
  rate  k x g(S / D2) x D2^0.5 x tier factor. Nick's pace grows with the square root of listing demand (alpha 0.5 fit best;
        "share x demand" over-predicted big listings). The sales model's trees L(x) aren't ported; k is fitted on Nick's own
        recent sales across his SKUs (pooled), then each SKU's own sales are blended in (Gamma-Poisson: the prior counts as
        PRIOR_DAYS days of evidence).
The repricer then scales this "competitive" pace by the Buy Box chance at a price relative to the chance when tied with the
lowest rival (shadow.decide).
"""
import math

KEEPA_EPOCH_MIN = 21564000
BRACKETS = [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000,
            20000, 30000, 40000, 50000, 1e9]
RANK_FIT = {'Beauty & Personal Care': (-0.5380135120852971, 10.956661417157044),
            'Health & Household': (-0.5823099080843296, 11.42741718067572),
            'Home & Kitchen': (-0.02485230232096271, 5.157434000674831),
            '_all': (-0.5223751771502675, 10.77613041915754)}
TIER = {'<=100': 1.0829462032565957, '101-300': 1.028993534932575, '301-999': 1.0123249101367147, '1000+': 1.1788672015562782}
NO_BRACKET_CAP = 147.0
PRIOR_DAYS = 7.0


def nxt(b):
    for x in BRACKETS:
        if x > b:
            return x
    return BRACKETS[-1]


def d_rank(rank, root, nvar=1):
    """Monthly units from sales rank, per root category; variations lower it (exp(-0.4 x ln nvar))."""
    if not rank or rank <= 0:
        return None
    s, c = RANK_FIT.get(root) or RANK_FIT['_all']
    return math.exp(c + s * math.log(rank) - 0.4 * math.log(max(nvar or 1, 1)))


def d2_monthly(bracket, rank, root, nvar=1):
    dr = d_rank(rank, root, nvar)
    if bracket and bracket > 0:
        lo, hi = float(bracket), min(nxt(bracket), 2.0 * bracket)
        return min(max(dr if dr else lo, lo), hi)
    return min(dr, NO_BRACKET_CAP) if dr else None


def from_keepa(p):
    """{'d2': units/day, 'bracket', 'rank', 'root', 'nvar'} from a Keepa product, or None."""
    csv = p.get('csv') or []
    r = csv[3] if len(csv) > 3 and csv[3] else None
    rank = r[-1] if r and len(r) >= 2 and r[-1] > 0 else None
    tree = p.get('categoryTree') or []
    root = tree[0].get('name') if tree and isinstance(tree[0], dict) else None
    nvar = len(p.get('variations') or []) or 1
    bracket = p.get('monthlySold') if (p.get('monthlySold') or 0) > 0 else None
    m = d2_monthly(bracket, rank, root, nvar)
    if not m:
        return None
    return {'d2': round(max(m / 30.0, 0.05), 3), 'bracket': bracket, 'rank': rank, 'root': root, 'nvar': nvar}


def g(days):
    d = max(days, 0.5)
    return (d / 7.0) ** 0.7 / (1.0 + (d / 60.0) ** 0.7) * (1.0 + (7.0 / 60.0) ** 0.7)


def tier(d2):
    m = d2 * 30.0
    return TIER['<=100'] if m <= 100 else TIER['101-300'] if m <= 300 else TIER['301-999'] if m <= 999 else TIER['1000+']


def shape(stock, d2):
    """g(S/D2) x D2^0.5 x tier: everything in the sales model's rate except the scale k."""
    return g(stock / d2) * math.sqrt(d2) * tier(d2)


def fit_k(rows):
    """Pooled scale from (units/day, stock, d2) of Nick's SKUs with sales: the median of rate / shape."""
    v = sorted(u / shape(s, d) for u, s, d in rows if u and u > 0 and s and s > 0 and d)
    return v[len(v) // 2] if v else None


def pace(k, stock, d2, own_units=None, own_days=None):
    """Units/day when priced competitively: the prior k x shape blended with the SKU's own sales."""
    prior = k * shape(max(stock, 1), d2)
    if own_units is None or not own_days:
        return prior
    return (prior * PRIOR_DAYS + own_units) / (PRIOR_DAYS + own_days)
