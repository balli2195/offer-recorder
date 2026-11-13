import datetime as dt, json, os
from zoneinfo import ZoneInfo
import abtest as AB, live as L

CT = ZoneInfo('America/Chicago')
ts = lambda *a: dt.datetime(*a, tzinfo=CT).timestamp()


class R:
    def __init__(self, code=200, body=None, loc=None):
        self.status_code, self._b, self.headers = code, body, ({'Location': loc} if loc else {})
    def json(self): return self._b
    def raise_for_status(self):
        if self.status_code >= 400: raise RuntimeError(self.status_code)


class Http:
    def __init__(self, fail=()):
        self.calls, self.fail = [], set(fail)
    def post(self, url, json=None, allow_redirects=True, timeout=None):
        self.calls.append(json); return R(302, loc='https://x/once')
    def get(self, url, timeout=None):
        b = self.calls[-1]
        f = [s for s in b['skus'] if s in self.fail]
        return R(200, {'ok': True, 'result': {'on': b['on'], 'updated': len(b['skus']) - len(f), 'failed': f, 'skipped': []}})


def mk(tmp_path, http=None, **cfg):
    logs = []
    lv = L.Live(str(tmp_path), lambda p, lines: None, lambda: 'now', lambda: 'tok')
    ab = AB.ABTest(str(tmp_path), lambda p, lines: logs.append(lines), lambda *a: '2026-11-13T09:00:00Z', live=lv, wait=lambda s: None, http=http or Http())
    lv.ab = ab
    ab.set_cfg({'enabled': True, 'start': '2026-11-13', 'groups': {'S1': 'A', 'S2': 'B'}, **cfg})
    return ab, lv, logs


def test_schedule_alternates_at_3am_central(tmp_path):
    ab, _, _ = mk(tmp_path)
    assert ab.arm('S1', ts(2026, 11, 13, 2, 59)) is None                 # before the first switch
    assert ab.arm('S1', ts(2026, 11, 13, 3, 0)) == 'repricer' and ab.arm('S2', ts(2026, 11, 13, 3, 0)) == 'sellersnap'
    assert ab.arm('S1', ts(2026, 11, 14, 2, 59)) == 'repricer'           # day 0 runs until 3 AM the next day
    assert ab.arm('S1', ts(2026, 11, 14, 3, 1)) == 'sellersnap' and ab.arm('S2', ts(2026, 11, 14, 3, 1)) == 'repricer'
    assert ab.arm('S9', ts(2026, 11, 14, 4)) is None                      # not a test SKU
    assert ab.washout(ts(2026, 11, 14, 4, 59)) and not ab.washout(ts(2026, 11, 14, 5, 1))
    n = dt.datetime.fromtimestamp(ab.next_switch(ts(2026, 11, 14, 4)), CT)
    assert (n.day, n.hour) == (15, 3)


def test_live_gate_follows_the_arm_and_dry_run(tmp_path):
    os.environ['LIVE_MODE'] = 'on'
    ab, lv, _ = mk(tmp_path)
    lv.set_cfg({'enabled': True, 'skus': []})
    lv.on_listings([{'sku': 'S1', 'repricing': False}, {'sku': 'S2', 'repricing': False}])
    d = {'sku': 'S1', 'asin': 'A1', 'action': 'raise', 'my_price': 30.0, 'rec_price': 31.0, 'min': 25.0, 'max': 40.0}
    t = ts(2026, 11, 13, 12)
    assert lv.plan(d, t) is None                                          # dry run (default): never live
    ab.set_cfg({'dry_run': False})
    assert lv.plan(d, t) == 31.0                                          # S1's repricer day, not in the live SKU list
    assert lv.plan(dict(d, sku='S2'), t) is None                          # S2 is on Seller Snap today
    lv.stop()
    assert lv.plan(d, t) is None                                          # the Hub's Stop stops the test too
    os.environ['LIVE_MODE'] = 'off'


def test_switch_turns_seller_snap_off_then_on(tmp_path):
    http = Http(fail={'S2'})
    ab, lv, logs = mk(tmp_path, http=http, dry_run=False, box_url='https://box', box_key='k')
    lv.on_listings([{'sku': 'S1', 'repricing': True}, {'sku': 'S2', 'repricing': False}])
    r = ab.switch(ts(2026, 11, 13, 3, 0, 5))
    assert [(c['action'], c['skus'], c['on']) for c in http.calls] == [('set_repricing', ['S1'], False), ('set_repricing', ['S2'], True)]
    assert lv.ss_repricing == {'S1': False, 'S2': False}                  # S2's switch failed: left as it was
    assert not r['ok'] and r['errors'] == ['1 SKU(s) not switched: S2'] and r['repricer_group'] == 'A' and json.loads(logs[-1][0])['day'] == 0
    ab2 = AB.ABTest(str(tmp_path), lambda p, l: None, lambda *a: 'x', wait=lambda s: None)
    assert ab2.cfg['box_key'] == 'k' and ab2.last_switch['repricer_group'] == 'A'   # config kept on disk
    assert 'box_key' not in ab.public() and ab.public()['box_ready']


def test_dry_run_switch_calls_nothing_and_end_restores(tmp_path):
    http = Http()
    ab, _, _ = mk(tmp_path, http=http, box_url='https://box', box_key='k')
    r = ab.switch(ts(2026, 11, 13, 3, 1))
    assert r['ok'] and r['dry_run'] and http.calls == []
    ab.set_cfg({'dry_run': False})
    ab.end()
    assert http.calls[-1]['on'] is True and sorted(http.calls[-1]['skus']) == ['S1', 'S2'] and not ab.cfg['enabled']
