import json, os, time
import live as L

def mk(tmp_path, env='on'):
    os.environ['LIVE_MODE'] = env
    logs = []
    lv = L.Live(str(tmp_path), lambda p, lines: logs.append((p, lines)), lambda: '2026-11-12T00:00:00Z', lambda: 'tok')
    return lv, logs

D = {'sku': 'S1', 'asin': 'A1', 'action': 'raise_to_target', 'my_price': 30.0, 'rec_price': 33.0, 'min': 25.0, 'max': 40.0, 'gain_24h': 2.0, 'reasons': []}

def test_off_by_default(tmp_path):
    lv, _ = mk(tmp_path, env='off')
    lv.on_listings([{'sku': 'S1', 'repricing': False}])
    lv.set_cfg({'enabled': True, 'skus': ['S1']})
    assert lv.plan(D) is None                      # env switch off
    os.environ['LIVE_MODE'] = 'on'
    lv2, _ = mk(tmp_path / 'b')
    lv2.on_listings([{'sku': 'S1', 'repricing': False}])
    assert lv2.plan(D) is None                     # config not enabled / SKU not listed

def test_blocks_when_seller_snap_still_repricing(tmp_path):
    lv, _ = mk(tmp_path)
    lv.set_cfg({'enabled': True, 'skus': ['S1']})
    lv.on_listings([{'sku': 'S1', 'repricing': True}])
    assert lv.plan(D) is None and lv.stats['blocked']['Seller Snap repricing still on'] == 1

def test_step_limit_and_bounds(tmp_path):
    lv, _ = mk(tmp_path)
    lv.set_cfg({'enabled': True, 'skus': ['S1'], 'max_step_pct': 5})
    lv.on_listings([{'sku': 'S1', 'repricing': False}])
    assert lv.plan(D) == 31.5                      # 30 -> 33 requested, capped at +5 %
    assert lv.plan(dict(D, rec_price=45.0)) is None and lv.stats['blocked']['outside min/max'] == 1
    assert lv.plan(dict(D, action='hold')) is None

def test_interval_and_daily_cap(tmp_path):
    lv, _ = mk(tmp_path)
    lv.set_cfg({'enabled': True, 'skus': ['S1'], 'min_interval_s': 600, 'max_per_day': 1})
    lv.on_listings([{'sku': 'S1', 'repricing': False}])
    now = time.time()
    lv.last_change['S1'] = now - 60
    assert lv.plan(D, now) is None and lv.stats['blocked']['too soon since last change'] == 1
    lv.last_change['S1'] = now - 3600
    lv.day_count[('S1', L.ct_day(now))] = 1   # the cap counts Central-time days
    assert lv.plan(D, now) is None and lv.stats['blocked']['daily change cap'] == 1

def test_patch_keeps_other_offer_fields(tmp_path):
    lv, _ = mk(tmp_path)
    item = {'summaries': [{'productType': 'BEAUTY'}], 'attributes': {'purchasable_offer': [
        {'marketplace_id': 'ATVPDKIKX0DER', 'currency': 'USD', 'our_price': [{'schedule': [{'value_with_tax': 30.0}]}],
         'minimum_seller_allowed_price': [{'schedule': [{'value_with_tax': 20.0}]}]}]}}
    pt, body = lv.build_patch(item, 31.5)
    v = body['patches'][0]['value'][0]
    assert pt == 'BEAUTY' and v['our_price'][0]['schedule'][0]['value_with_tax'] == 31.5
    assert v['minimum_seller_allowed_price'][0]['schedule'][0]['value_with_tax'] == 20.0
    assert item['attributes']['purchasable_offer'][0]['our_price'][0]['schedule'][0]['value_with_tax'] == 30.0  # input not mutated

def test_apply_logs_and_counts(tmp_path, monkeypatch):
    lv, logs = mk(tmp_path)
    lv.set_cfg({'enabled': True, 'skus': ['S1']})
    lv.on_listings([{'sku': 'S1', 'repricing': False}])
    sent = {}
    monkeypatch.setattr(lv, '_get_item', lambda sku, tok: {'summaries': [{'productType': 'BEAUTY'}], 'attributes': {}})
    class R:
        status_code = 200; content = b'1'
        def json(self): return {'status': 'ACCEPTED'}
    def patch(url, headers, params, json, timeout):
        sent.update(params=params, body=json); return R()
    monkeypatch.setattr(L.requests, 'patch', patch)
    rec = lv.maybe_apply(D)
    assert rec['to'] == 31.5 and lv.stats['applied'] == 1 and 'mode' not in sent['params']
    assert '/live/' in logs[0][0]
    # preview never applies
    lv.send('S1', 31.0, preview=True)
    assert sent['params']['mode'] == 'VALIDATION_PREVIEW'

def test_stop(tmp_path):
    lv, _ = mk(tmp_path)
    lv.set_cfg({'enabled': True, 'skus': ['S1']})
    assert lv.stop()['enabled'] is False
    lv2, _ = mk(tmp_path)          # persisted
    assert lv2.cfg['enabled'] is False and lv2.cfg['skus'] == ['S1']


CUT = {'sku': 'S1', 'asin': 'A1', 'action': 'undercut', 'my_price': 30.0, 'rec_price': 29.0, 'low_rival': 30.0, 'min': 20.0, 'max': 40.0, 'gain_24h': 1.0, 'reasons': []}


def live_ready(tmp_path, **cfg):
    lv, logs = mk(tmp_path)
    lv.set_cfg(dict({'enabled': True, 'skus': ['S1'], 'min_interval_s': 120}, **cfg))
    lv.on_listings([{'sku': 'S1', 'repricing': False}])
    lv.send = lambda sku, price, preview=False: (200, {'status': 'ACCEPTED'}, {})
    return lv, logs


def test_chased_cuts_pause_cuts_but_not_raises(tmp_path):
    lv, _ = live_ready(tmp_path, chase_cuts=2, chase_window_min=30, max_drop_pct=50)
    t = time.time()
    lv.maybe_apply(dict(CUT))                                   # cut 30 -> 29
    lv.observe(dict(CUT, my_price=29.0, low_rival=29.0), t + 300)  # a rival matched within 5 min
    assert lv.streak['S1'] == 1 and 'S1' not in lv.frozen
    lv.last_change['S1'] = t - 1000
    lv.maybe_apply(dict(CUT, my_price=29.0, rec_price=28.0, low_rival=29.0))  # cut again 29 -> 28
    lv.observe(dict(CUT, my_price=28.0, low_rival=27.99), t + 600)           # answered again
    assert 'S1' in lv.frozen and 'S1' in lv.stats['frozen']
    lv.last_change['S1'] = t - 1000
    assert lv.plan(dict(CUT, my_price=28.0, rec_price=27.5, low_rival=27.99)) is None
    assert lv.stats['blocked']['cuts paused: chased by a rival'] == 1
    assert lv.plan(dict(CUT, action='raise', my_price=28.0, rec_price=29.0, low_rival=29.5)) == 29.0   # raises still allowed
    lv.unfreeze('S1')
    assert 'S1' not in lv.frozen and lv.streak['S1'] == 0


def test_unanswered_cut_resets_streak(tmp_path):
    lv, _ = live_ready(tmp_path)
    t = time.time()
    lv.maybe_apply(dict(CUT))
    lv.streak['S1'] = 1
    lv.observe(dict(CUT, my_price=29.0, low_rival=30.0), t + 31 * 60)   # nobody followed within 30 min
    assert lv.streak['S1'] == 0 and lv.cuts['S1']['answered'] is False


def test_matching_the_low_is_not_counted_as_chased(tmp_path):
    lv, _ = live_ready(tmp_path, chase_cuts=1)
    lv.maybe_apply(dict(CUT, action='match_low', rec_price=29.0, low_rival=29.0))   # rival already at 29
    lv.observe(dict(CUT, my_price=29.0, low_rival=29.0), time.time() + 60)
    assert 'S1' not in lv.frozen


def test_daily_drop_limit(tmp_path):
    lv, _ = live_ready(tmp_path, max_drop_pct=10, max_step_pct=10)
    now = time.time()
    lv.seen['S1'].append((now - 3600, 33.0))                    # was 33.00 an hour ago
    assert lv.plan(dict(CUT, my_price=30.0, rec_price=27.0), now) == 29.7   # floor = 33 x 0.90
    assert lv.plan(dict(CUT, my_price=29.7, rec_price=27.0), now) is None
    assert lv.stats['blocked']['24 h drop limit'] == 1


def test_brake_state_survives_restart(tmp_path):
    lv, logs = live_ready(tmp_path)
    lv.frozen['S1'] = {'until': time.time() + 3600, 'why': 'test'}
    lv.save_state()
    os.makedirs(tmp_path / 'live', exist_ok=True)
    import gzip as _gz
    with _gz.open(tmp_path / 'live' / (time.strftime('%Y-%m-%d', time.gmtime()) + '.jsonl.gz'), 'wt') as f:
        f.write(json.dumps({'t': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'sku': 'S1', 'from': 30.0, 'to': 29.0, 'status_code': 200}) + '\n')
    lv2, _ = mk(tmp_path)
    assert 'S1' in lv2.frozen and lv2.cuts['S1']['price'] == 29.0 and lv2.last_change['S1'] > 0
    assert max(p for _, p in lv2.seen['S1']) == 30.0


def test_day_is_central_time():
    t = 1762995600   # 2025-11-13 01:00 UTC = 2025-11-12 19:00 Central
    assert L.ct_day(t) == '2025-11-12'
