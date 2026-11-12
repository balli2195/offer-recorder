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
    import datetime as dt
    lv.day_count[('S1', dt.datetime.fromtimestamp(now, dt.timezone.utc).strftime('%Y-%m-%d'))] = 1
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
