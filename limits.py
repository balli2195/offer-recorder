"""Price limits and inputs for the repricer from Nick's own data, so nothing depends on Seller Snap (Nick, Nov 12 2026).

  cost  Hub cost (POST /limits from the Hub, hourly): InventoryLab snapshot -> InventoryLab month (by SKU) ->
        order-ledger average (last 12 months) -> latest lead cost (by ASIN). Seller Snap's cost only if none of those has it.
  min   a Hub override (liquidation min Nick approved, or a manual min), else the lowest price where
        (price - Amazon fees - cost - prep) / cost >= min ROI (12%), with Amazon's exact fees for the SKU.
  max   a Hub override, else the higher of the highest lowest-FBA price in the past year (Keepa) and 1.25 x min;
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


def fba_year_high(product, now_min=None):
    """(highest lowest-FBA price in the past 365 days, days of FBA history in that year) from a Keepa product with
    offers and a year of history (csv[10] = lowest new FBA price). None when the product has no FBA history."""
    csv = product.get('csv') or []
    c = csv[10] if len(csv) > 10 else None
    if not c or len(c) < 4:
        return None
    t_last = c[-2]
    now_min = now_min or t_last
    start = now_min - 365 * 1440
    hi, span = None, 0
    prev_t, prev_v = None, None
    for i in range(0, len(c) - 1, 2):
        t, v = c[i], c[i + 1]
        if prev_t is not None and prev_v is not None and prev_v > 0:
            a, b = max(prev_t, start), min(t, now_min)
            if b > a:
                span += b - a
                hi = max(hi or 0, prev_v)
        prev_t, prev_v = t, v
    if prev_v is not None and prev_v > 0 and now_min > max(prev_t, start):
        span += now_min - max(prev_t, start)
        hi = max(hi or 0, prev_v)
    if not hi:
        return None
    return round(hi / 100.0, 2), round(span / 1440.0, 1)


def resolve_max(min_price, year_high, settings):
    """(max, source)."""
    if not min_price:
        return None, None
    hi, days = (year_high or (None, 0))
    if hi and days >= settings.get('max_history_days', 90):
        return round(max(hi, settings.get('max_floor_mult', 1.25) * min_price), 2), 'year-high FBA price' if hi >= settings.get('max_floor_mult', 1.25) * min_price else '1.25 x min'
    return round(settings.get('max_new_mult', 2.0) * min_price, 2), '2 x min (short FBA history)'
