import json, math, time
import numpy as np
import ctxfeat as CX
import shadow as S
from test_shadow import MODEL, SKU


def msg(offers, t='2026-11-13T12:00:00Z', ranks=((None, 1500), (None, 40))):
    return {'OfferChangeTrigger': {'ASIN': 'A1', 'TimeOfOfferChange': t},
            'Summary': {'SalesRankings': [{'ProductCategoryId': 'x', 'Rank': r} for _, r in ranks],
                        'NumberOfOffers': [{'Condition': 'new', 'OfferCount': len(offers)}], 'NumberOfBuyBoxEligibleOffers': [{'Condition': 'new', 'OfferCount': 2}],
                        'CompetitivePriceThreshold': {'Amount': 28.0}, 'ListPrice': {'Amount': 40.0}},
            'Offers': [{'SellerId': s, 'SubCondition': 'new', 'IsFulfilledByAmazon': f, 'IsBuyBoxWinner': w, 'ListingPrice': {'Amount': p}, 'Shipping': {'Amount': 0},
                        'SellerFeedbackRating': {'FeedbackCount': fc, 'SellerPositiveFeedbackRating': 95}, 'IsFeaturedMerchant': True,
                        'ShippingTime': {'AvailabilityType': 'NOW'}, 'PrimeInformation': {'IsOfferPrime': f}} for s, p, f, w, fc in offers]}


def test_message_ctx_and_vector():
    p = msg([(S.ME, 30.5, True, False, 100), ('R1', 30.0, True, True, 999), ('R2', 31.0, True, False, 9), ('F1', 29.0, False, False, 50)])
    row, L = CX.message_ctx(p)
    assert L == 30.0 and len(row) == len(CX.MSG_COLS)
    d = dict(zip(CX.MSG_COLS, row))
    assert abs(d['log_rank'] - math.log10(1500)) < 1e-9 and d['cpt_ratio'] == 28 / 30 and d['fbm_ratio'] == 29 / 30 and d['hold_is_low'] == 1.0 and d['n_winners'] == 1
    assert CX.message_ctx(msg([('R1', 30.0, True, True, 9)])) is None          # Nick not on the listing
    lv = CX.Live(); now = time.time()
    lv.add('A1', now - 3600, p); lv.add('A1', now - 60, p)
    v = lv.vector('A1', now, 2, 'low')
    assert len(v) == len(CX.FEATURES) and dict(zip(CX.FEATURES, v))['msgs_6h'] == 2 and v[-3:] == [0.0, 0.0, 0.0]
    # training's array form gives the same vector
    tt = np.array([now - 3600, now - 60]); M = np.array([row, row]); Ls = np.array([L, L])
    assert np.allclose(CX.vector_at(tt, M, Ls, 1, now, 2, 'low'), v, equal_nan=True)


class Tree:   # stand-in booster: adds a fixed log-odds
    def __init__(self, z): self.z = z
    def predict(self, X, raw_score=True): return np.array([self.z] * len(X))


def test_decide_uses_trees_and_change_model():
    offs = {S.ME: 30.5, 'CALM': 30.0}
    base = S.decide(MODEL, 'A1', offs, 'CALM', SKU, foep=30.2, foep_age=60)
    ctx = lambda n, hk: [0.0] * len(CX.FEATURES)
    up = S.decide(MODEL, 'A1', offs, 'CALM', SKU, foep=30.2, foep_age=60, boosters={'tree_foep': Tree(1.0)}, ctx_fn=ctx)
    assert up['cur']['p_win'] > base['cur']['p_win'] and 'tree' in up['bb_parts']
    chg = dict(MODEL, bb_change_foep={'coef': {'b': -5.0}})                       # post-change model: every change scores very low
    d = S.decide(chg, 'A1', offs, 'CALM', SKU, foep=30.2, foep_age=60)
    assert d['cur']['p_win'] == base['cur']['p_win']                            # current price: ordinary model
    others = [r for r in [d['best']] if abs(r['price'] - 30.5) >= 0.005]
    assert d['action'] == 'hold' or all(r['p_win'] < 0.01 for r in others)      # other prices: the post-change model
    assert 'post-change model' in d['bb_parts']
    assert S.change_terms(31.0, 30.0, 1.0) == {'c_raise': 1.0, 'c_raise_me': 1.0, 'c_step': 100 * (31 / 30 - 1), 'c_step_raise': 100 * (31 / 30 - 1)}
    assert S.decide(MODEL, 'A1', offs, 'CALM', SKU, boosters={'tree_foep': Tree(1.0)}, ctx_fn=lambda n, hk: None)['bb_parts'] == []   # no context yet


def test_shadow_loads_boosters_from_model(tmp_path):
    import lightgbm as lgb, os
    X = np.random.default_rng(1).normal(size=(400, len(CX.FEATURES))); y = (X[:, 0] > 0).astype(float)
    b = lgb.train({'objective': 'cross_entropy', 'verbose': -1, 'num_leaves': 4}, lgb.Dataset(X, y, feature_name=CX.FEATURES), 5)
    m = dict(MODEL, tree_foep={'booster': b.model_to_string(), 'features': CX.FEATURES, 'rounds': 5},
             tree_nofoep={'booster': b.model_to_string(), 'features': ['wrong'], 'rounds': 5})
    os.makedirs(tmp_path / 'model'); json.dump(m, open(tmp_path / 'model' / 'model.json', 'w'))
    sh = S.Shadow(str(tmp_path), lambda p, l: None, lambda *a: 'now')
    assert list(sh.boosters) == ['tree_foep'] and 'tree_foep' in sh.stats['model_parts']


def test_decide_uses_unified_model_when_a_target_is_known():
    # Nick, Nov 14: one model scores the current price (is_change 0) and every other price (is_change 1); no-target decisions don't use it
    offs = {S.ME: 30.5, 'CALM': 30.0}
    uni = dict(MODEL, bb_unified={'coef': {'b': 0.0, 'is_change': -5.0}, 'platt': {'hold': [1.0, 0.0], 'change': [1.0, 0.0]}})
    ctx = lambda n, hk: [0.0] * len(CX.FEATURES)
    d = S.decide(uni, 'A1', offs, 'CALM', SKU, foep=30.2, foep_age=60, boosters={'tree_unified': Tree(0.5)}, ctx_fn=ctx)
    assert d['bb_parts'] == ['unified model', 'unified tree', 'calibrated']
    assert abs(d['cur']['p_win'] - S.sigmoid(0.5)) < 1e-3                      # holding: b + tree
    assert d['best']['price'] == 30.5 or d['best']['p_win'] < 0.02             # other prices: is_change -5 + tree 0.5
    x = S.unified_feats(30.0, 30.5, 30.0, 1, 30.2, {'kind': 'low', 'price': 30.0})
    assert x['is_change'] == 1.0 and x['chg_x_t_ok'] == x['t_ok'] and x['share'] == 0.0 and 'chg_x_is_change' not in x
    h = S.unified_feats(30.5, 30.5, 30.0, 1, 30.2, {'kind': 'me'})
    assert h['is_change'] == 0.0 and h['c_step'] == 0.0 and all(v == 0.0 for k, v in h.items() if k.startswith('chg_x_'))
    nt = S.decide(uni, 'A1', offs, 'CALM', SKU)                                # no target: the old no-target path
    assert 'unified model' not in nt['bb_parts']
    assert abs(S.platt(0.7, [0.9, -0.12]) - S.sigmoid(0.9 * math.log(0.7 / 0.3) - 0.12)) < 1e-9
