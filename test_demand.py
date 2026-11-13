import time
import demand as DM
import shadow as S


def test_d2_bracket_rank_and_cap():
    # inside [bracket, min(next, 2 x bracket)]: a strong rank lifts a 50,000+ listing toward its cap, a weak one stays at the floor
    assert 50000 < DM.d2_monthly(50000, 1, 'Beauty & Personal Care') <= 100000
    assert DM.d2_monthly(100, 10 ** 6, 'Beauty & Personal Care') == 100
    assert DM.d2_monthly(None, 10 ** 6, 'Beauty & Personal Care') < 147 and DM.d2_monthly(None, 1, 'x') == 147   # no bracket: capped
    p = {'monthlySold': 50000, 'csv': [None, None, None, [1, 50, 2, 40]], 'categoryTree': [{'name': 'Beauty & Personal Care'}], 'variations': [{}] * 4}
    x = DM.from_keepa(p)
    assert x['bracket'] == 50000 and x['rank'] == 40 and x['nvar'] == 4 and 1666 < x['d2'] <= 3334


def test_stock_curve_and_blend():
    assert abs(DM.g(7) - 1) < 1e-9 and DM.g(0.5) < DM.g(7) < DM.g(60)
    k = DM.fit_k([(u, 100, 10) for u in (1, 2, 3)])
    assert abs(k - 2 / DM.shape(100, 10)) < 1e-9
    assert DM.pace(k, 100, 10) == k * DM.shape(100, 10)
    assert abs(DM.pace(k, 100, 10, own_units=70, own_days=7) - (DM.pace(k, 100, 10) * 7 + 70) / 14) < 1e-9   # own sales blended in


def test_shadow_pace_from_demand_and_orders(tmp_path):
    sh = S.Shadow(str(tmp_path), lambda p, l: None, lambda *a: 'now')
    now = time.time()
    out = {f'A{i}': {'sku': f'S{i}', 'v30': 1.0, 'qty': 50, 'src': {}} for i in range(25)}
    inv = {f'S{i}': {'fulfillable': 40, 'reserved': 5, 'inbound': 5} for i in range(25)}
    dmd = {f'A{i}': {'d2': 10.0} for i in range(25)}
    dmd['A0'] = {'d2': 1000.0}                                    # a big listing, stocked 10 days deep
    inv['S0'] = {'fulfillable': 10000, 'reserved': 0, 'inbound': 0}
    orders = {f'S{i}': [(now - 3600, 7)] for i in range(25)}      # 1/day each over 7 days
    sh.set_pace(out, inv, dmd, orders, now)
    assert sh.stats['pace']['fit_skus'] == 25 and sh.stats['pace']['own_sales'] == '7-day orders'
    assert out['A0']['pace_prior'] > out['A1']['pace_prior'] * 5     # the square root of demand (10x at equal stock depth)
    assert abs(out['A1']['pace_day'] - 1.0) < 0.3                    # a typical SKU's pace stays near its own sales
    del dmd['A2']
    out2 = {k: dict(v, src={}) for k, v in out.items()}
    for v in out2.values():
        v.pop('pace_day', None)
    sh.set_pace(out2, inv, dmd, orders, now)
    assert 'pace_day' not in out2['A2']                              # no Keepa demand: the old pace (lam) is used


def test_decide_uses_pace_scaled_by_buy_box_chance():
    from test_shadow import MODEL, SKU
    base = S.decide(MODEL, 'A1', {S.ME: 30.5, 'CALM': 30.0}, 'CALM', SKU)
    fast = S.decide(MODEL, 'A1', {S.ME: 30.5, 'CALM': 30.0}, 'CALM', dict(SKU, pace_day=20.0, d2=500.0))
    assert fast['lam'] > base['lam'] and fast['pace_day'] == 20.0
    tied = S.p_win(MODEL, S.feats(30.0, 30.0, 1))
    assert abs(fast['lam'] - 20.0 / 24 / max(tied, 0.25)) < 1e-3
    assert any('sales pace 20.0/day' in r for r in fast['reasons'])
