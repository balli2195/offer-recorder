"""Price limits and inputs for the repricer from Nick's own data, so nothing depends on Seller Snap (Nick, Nov 12 2026).

  cost  Hub cost (POST /limits from the Hub, hourly): InventoryLab snapshot -> InventoryLab month (by SKU) ->
        order-ledger average (last 12 months) -> latest lead cost (by ASIN). Seller Snap's cost only if none of those has it.
  min   a Hub override (liquidation min Nick approved, or a manual min), else the lowest price where
        (price - Amazon fees - cost - prep) / cost >= min ROI (12%), with Amazon's exact fees for the SKU.
  max   a Hub override, else the higher of the year high (Keepa: the higher of the lowest-FBA price and the Buy Box price over
        the past year) and 1.25 x min;
        with under 90 days of FBA history, 2 x min. At decision time it is raised to a rival's Buy Box price above it.
  sales Amazon Sales & Traffic: units ordered and Buy Box % over the last 30 days, by ASIN.
"""
import math

import fees as _fees

KEEPA_EPOCH_MIN = 21564000   # Keepa time is minutes since 2011-01-01


def resolve_cost(sku, asin, costs, ss_cost=None):
    """(cost, source) using the Hub's order of sources."""
    costs = costs or {}
    c = (costs.get('sku') or {}).get(sku)
    if c and c[0] and c[0] > 0.01:
        return float(c[0]), 'InventoryLab' if c[1] == 'inventorylab' else 'InventoryLab month'
    a = (asin or '').upper()
    if (costs.get('asin_ledger') or {}).get(a):
        return float(costs['asin_ledger'][a]), 'order ledger'
    if (costs.get('asin_lead') or {}).get(a):
        return float(costs['asin_lead'][a]), 'lead cost'
    if ss_cost:
        return float(ss_cost), 'Seller Snap'
    return None, None


def fee_fn(bands, fallback_fba):
    def f(p):
        x = _fees.fee_at(bands, p) if bands else None
        if x:
            return x[0]
        return (0.08 if p <= 10 else 0.15) * p + fallback_fba
    return f


def rule_min(cost, bands, prep, min_roi, fallback_fba=4.11):
    """Lowest price (to the cent) with (p - fees(p) - cost - prep) / cost >= min_roi."""
    if not cost or cost <= 0:
        return None
    fee = fee_fn(bands, fallback_fba)
    need = cost * (1 + min_roi) + prep          # p - fees(p) must reach this
    ok = lambda p: p - fee(p) >= need - 1e-9
    # Profit rises with price inside each fee band (Amazon's fee steps at $10 and $50), so binary-search each band in order.
    for lo, hi in ((0.01, 9.99), (10.00, 10.00), (10.01, 50.00), (50.01, max(60.0, need * 3 + 50))):
        a, b = int(round(lo * 100)), int(round(hi * 100))
        if not ok(b / 100.0):
            continue
        while a < b:
            m = (a + b) // 2
            if ok(m / 100.0):
                b = m
            else:
                a = m + 1
        return round(a / 100.0, 2)
    return None


def _year_high(c, step, start, now_min):
    """(highest price, minutes with a price) over [start, now] in a Keepa price series. step 2 = (time, price);
    step 3 = (time, price, shipping), as in the Buy Box series."""
    hi, span, prev_t, prev_v = None, 0, None, None
    pts = [(c[i], (c[i + 1] + max(c[i + 2], 0)) if step == 3 and c[i + 1] > 0 else c[i + 1]) for i in range(0, len(c) - step + 1, step)]
    for t, v in pts + [(now_min, None)]:
        if prev_t is not None and prev_v is not None and prev_v > 0:
            a, b = max(prev_t, start), min(t, now_min)
            if b > a:
                span += b - a
                hi = max(hi or 0, prev_v)
        prev_t, prev_v = t, v
    return hi, span


def fba_year_high(product, now_min=None):
    """(year high, days of price history, highest lowest-FBA price, highest Buy Box price) over the past 365 days, from a Keepa
    product pulled with offers and a year of history (csv[10] = lowest new FBA price, csv[18] = Buy Box price + shipping).
    The year high is the higher of the two: the lowest-FBA price misses times an FBM seller or Amazon held the Buy Box above
    every FBA offer (Nov 12 check: 7 of 28 restocking SKUs). None when the product has neither."""
    csv = product.get('csv') or []
    f = csv[10] if len(csv) > 10 and csv[10] and len(csv[10]) >= 4 else None
    b = csv[18] if len(csv) > 18 and csv[18] and len(csv[18]) >= 6 else None
    if not f and not b:
        return None
    now_min = now_min or max(x[-3 if x is b else -2] for x in (f, b) if x)
    start = now_min - 365 * 1440
    fh, fs = _year_high(f, 2, start, now_min) if f else (None, 0)
    bh, bs = _year_high(b, 3, start, now_min) if b else (None, 0)
    if not fh and not bh:
        return None
    r = lambda v: round(v / 100.0, 2) if v else None
    return r(max(fh or 0, bh or 0)), round(max(fs, bs) / 1440.0, 1), r(fh), r(bh)


def resolve_max(min_price, year_high, settings):
    """(max, source)."""
    if not min_price:
        return None, None
    yh = tuple(year_high or (None, 0))
    hi, days = yh[0], yh[1]
    fh, bh = (yh[2], yh[3]) if len(yh) >= 4 else (hi, None)
    floor = settings.get('max_floor_mult', 1.25) * min_price
    if hi and days >= settings.get('max_history_days', 90):
        if hi < floor:
            return round(floor, 2), '1.25 x min'
        return round(hi, 2), 'year-high Buy Box price' if bh and bh >= hi and (not fh or bh > fh) else 'year-high FBA price'
    return round(settings.get('max_new_mult', 2.0) * min_price, 2), '2 x min (short FBA history)'
