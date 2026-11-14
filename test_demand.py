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
    w = tied + S.SALES_BETA * (1 - tied)   # sales at a share s scale with s + beta (1 - s)
    assert abs(fast['lam'] - 20.0 / 24 / max(w, 0.25)) < 1e-3
    assert any('sales pace 20.0/day' in r for r in fast['reasons'])


def test_train_stand_in_takes_keepa_products():
    # tools/train.py builds a bare Shadow (no __init__) and feeds it Keepa products: every attribute on_keepa_products touches must exist
    import ast, os
    src = open(os.path.join(os.path.dirname(__file__), 'tools', 'train.py')).read()
    line = next(l for l in src.splitlines() if 'sh.fba_high = {}' in l)
    sh = S.Shadow.__new__(S.Shadow); sh.lock = __import__('threading').Lock(); sh.rival_stock = {}
    exec(line, {'sh': sh})
    sh.on_keepa_products([{'asin': 'A1', 'monthlySold': 100, 'csv': [None, None, None, [1, 500]], 'categoryTree': [{'name': 'x'}]}])
    assert 'A1' in sh.demand


def test_sales_without_the_buy_box():
    # units = rate x (s + beta (1 - s)): with beta > 0 a price with almost no Buy Box chance still sells, and a fitted beta in the model wins
    from test_shadow import MODEL, SKU
    sk = dict(SKU, pace_day=10.0, d2=100.0)
    d0 = S.decide(dict(MODEL, sales_link={'beta_used': 0.0}), 'A1', {S.ME: 30.5, 'CALM': 30.0}, 'CALM', sk, my_stock=50)
    d1 = S.decide(dict(MODEL, sales_link={'beta_used': 0.3}), 'A1', {S.ME: 30.5, 'CALM': 30.0}, 'CALM', sk, my_stock=50)
    assert d1['sales_beta'] == 0.3 and d0['sales_beta'] == 0.0
    pw = d0['cur']['p_win']
    assert d1['cur']['units_24h'] > d0['cur']['units_24h']
    tied = S.p_win(MODEL, S.feats(30.0, 30.0, 1))
    want = 10.0 * (pw + 0.3 * (1 - pw)) / max(tied + 0.3 * (1 - tied), 0.25)
    assert abs(d1['cur']['units_24h'] - min(want, d1['qty'])) < 0.01


def test_saleslink_fit_recovers_beta():
    import numpy as np, saleslink as SL
    rng = np.random.default_rng(1); seg, sales = {}, {}
    t0, b = 0.0, 10800
    for i in range(120):
        a = f'A{i}'; rate = rng.uniform(0.2, 2.0); tt, ss, sl = [], [], []
        for k in range(56):
            s = float(rng.choice([0.0, 0.1, 0.5, 1.0])); tt.append(k * b + 60); ss.append(s)
            q = rng.poisson(rate * (s + 0.25 * (1 - s)))
            if q: sl.append((k * b + 120, float(q)))
        seg[a] = (tt, ss); sales[a] = sl
    r = SL.fit(seg, sales, t0, 56 * b, test_from=40 * b)
    assert abs(r['beta'] - 0.25) <= 0.06 and r['test_dev'] < r['test_dev_beta0']
