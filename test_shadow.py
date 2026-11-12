import json, gzip, os, time
import shadow as S

MODEL = {
    'trained_at': 'test', 'rivals': {'REACT': {'rate': 0.9, 'lag_min': 5}, 'CALM': {'rate': 0.05, 'lag_min': 20}},
    'rival_default': {'rate': 0.2, 'lag_min': 12},
    'war_table': [[0.05, 0.4], [0.9, 4.0]],
    # Buy Box: strong reward for being at/below target, penalty for being above the lowest rival
    'bb_foep': {'coef': {'b': -1.0, 't_ok': 3.0, 't_pos': -0.5, 'low_pos': -0.3, 'tied': 0.0, 'low_neg': 0.0, 'above1': -0.5, 'lnriv': -0.2, 'share': 0.5, 'share_na': 0.0, 't_neg': 0.0}},
    'bb_nofoep': {'coef': {'b': 0.0, 'low_neg': -0.2, 'low_pos': -0.6, 'tied': -0.5, 'above1': -1.0, 'lnriv': -0.2, 'share': 0.5, 'share_na': 0.0}},
}
SKU = {'sku': 'S1', 'min': 25.0, 'max': 40.0, 'cost': 15.0, 'fixed_fee': 4.5, 'lam': 0.2}


def test_feats_and_pwin_monotone():
    a = S.p_win(MODEL, S.feats(30.0, 30.0, 2, foep=30.5))
    b = S.p_win(MODEL, S.feats(31.0, 30.0, 2, foep=30.5))
    assert a > b and 0 < b < 1


def test_raises_to_target_when_holding_and_target_above():
    d = S.decide(MODEL, 'A1', {S.ME: 29.50, 'CALM': 30.40}, S.ME, SKU, foep=30.10, foep_age=60)
    assert d['action'] in ('raise_to_target', 'raise', 'raise_to_next') and d['rec_price'] > 29.50
    assert SKU['min'] <= d['rec_price'] <= SKU['max']


def test_never_outside_min_max():
    sku = dict(SKU, min=31.0, max=33.0)
    d = S.decide(MODEL, 'A1', {S.ME: 31.5, 'CALM': 29.0}, 'CALM', sku, foep=28.0, foep_age=60)
    assert 31.0 <= d['rec_price'] <= 33.0 and d['best']['price'] >= 31.0
    assert 'floor above lowest rival' in d['reasons']


def test_undercut_penalised_more_against_reactive_rival():
    calm = S.decide(MODEL, 'A1', {S.ME: 30.5, 'CALM': 30.0}, 'CALM', SKU)
    react = S.decide(MODEL, 'A1', {S.ME: 30.5, 'REACT': 30.0}, 'REACT', SKU)
    pc = {r['price']: r for r in [calm['best']]}
    # the undercut candidate 29.99 carries a bigger price-war penalty against the reactive rival
    def pen(d, model_sellers):
        return S.war_penalty_pct(MODEL, S.rival_info(MODEL, model_sellers)[0])
    assert pen(react, 'REACT') > pen(calm, 'CALM')
    assert not (react['action'] == 'undercut' and calm['action'] != 'undercut')


def test_stale_target_ignored():
    d = S.decide(MODEL, 'A1', {S.ME: 30.0, 'CALM': 30.0}, 'CALM', SKU, foep=35.0, foep_age=3600)
    assert d['foep'] is None and 'no fresh target price' in d['reasons']


def test_hold_when_gain_small():
    sku = dict(SKU, lam=0.0001)
    d = S.decide(MODEL, 'A1', {S.ME: 30.0, 'CALM': 30.0}, 'CALM', sku, foep=30.0, foep_age=10)
    assert d['action'] == 'hold' and d['rec_price'] == 30.0


def test_needs_me_and_a_rival():
    assert S.decide(MODEL, 'A1', {'CALM': 30.0}, 'CALM', SKU) is None
    assert S.decide(MODEL, 'A1', {S.ME: 30.0}, S.ME, SKU) is None


def body(asin, offers):
    return json.dumps({'Payload': {'AnyOfferChangedNotification': {'OfferChangeTrigger': {'ASIN': asin}, 'Offers': [
        {'SellerId': s, 'SubCondition': 'new', 'IsFulfilledByAmazon': True, 'ListingPrice': {'Amount': p}, 'Shipping': {'Amount': 0}, 'IsBuyBoxWinner': w}
        for s, p, w in offers]}}})


def test_shadow_feeds_and_logs(tmp_path):
    written = []
    sh = S.Shadow(str(tmp_path), lambda path, lines: written.append((path, lines)), lambda: '2026-11-12T00:00:00Z')
    assert sh.model is None
    sh.save_model(json.dumps(MODEL))
    assert sh.model['trained_at'] == 'test' and sh.stats['model_info']['rivals'] == 2
    sh.on_listings([{'sku': 'S1', 'asin': 'A1', 'min_price': 25, 'max_price': 40, 'listed_price': 30, 'cost': 15, 'roi': 30, 'v30': 2, 'avg_bb_share': 50, 'repricing': True, 'qty': 5}])
    sh.on_inventory([{'sellerSku': 'S1', 'inventoryDetails': {'fulfillableQuantity': 5}}])
    sh.on_foep('S1', {'featuredOfferExpectedPriceResults': [{'resultStatus': 'VALID_FOEP', 'featuredOfferExpectedPrice': {'listingPrice': {'amount': 30.1}}}]})
    sh.on_message_bodies([body('A1', [(S.ME, 29.5, True), ('CALM', 30.4, False)]), body('A2', [('X', 10, True), ('Y', 11, False)])])
    assert list(sh.pending) == ['A1']        # only listings where Nick has an offer are queued
    assert sh.step() == 1
    path, lines = written[0]
    d = json.loads(lines[0])
    assert '/shadow/' in path and d['asin'] == 'A1' and d['foep'] == 30.1 and sh.stats['decisions'] == 1
    sh.on_message_bodies([body('A1', [(S.ME, 29.5, True), ('CALM', 30.3, False)])])
    assert sh.step() == 0                     # debounced


def test_no_penalty_for_staying_where_already_below():
    d = S.decide(MODEL, 'A1', {S.ME: 29.0, 'REACT': 30.0}, S.ME, SKU, foep=29.5, foep_age=60)
    cur = d['cur']
    assert cur['price'] == 29.0 and cur['penalty'] == 0.0
