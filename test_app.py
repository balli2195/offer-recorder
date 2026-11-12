import gzip, json, os, tempfile, time, importlib, sys
os.environ.update(DATA_DIR=tempfile.mkdtemp(), RECORDER_KEY='k', AWS_ACCESS_KEY_ID='x', AWS_SECRET_ACCESS_KEY='y', AWS_DEFAULT_REGION='us-east-1', QUEUE_URL='')
import boto3
from moto import mock_aws
import app as A

def read(path):
    with gzip.open(path, 'rt') as f:
        return [json.loads(l) for l in f]

@mock_aws
def test_saves_then_deletes_and_multi_member_gzip():
    s = boto3.client('sqs', region_name='us-east-1'); q = s.create_queue(QueueName='t')['QueueUrl']; A.QUEUE = q
    for i in range(25): s.send_message(QueueUrl=q, MessageBody=json.dumps({'i': i}))
    n = 0
    for _ in range(6):
        n += A.poll_once(s)
        if n == 25: break
    assert n == 25
    assert s.get_queue_attributes(QueueUrl=q, AttributeNames=['ApproximateNumberOfMessages'])['Attributes']['ApproximateNumberOfMessages'] == '0'
    files = []
    for r, _, ns in os.walk(os.path.join(A.DATA, 'offers')): files += [os.path.join(r, x) for x in ns]
    rows = [x for f in files for x in read(f)]
    assert sorted(json.loads(x['body'])['i'] for x in rows) == list(range(25))
    assert all(x['sent_ms'] for x in rows)

@mock_aws
def test_disk_failure_keeps_message_in_queue():
    s = boto3.client('sqs', region_name='us-east-1'); q = s.create_queue(QueueName='f', Attributes={'VisibilityTimeout': '0'})['QueueUrl']; A.QUEUE = q
    s.send_message(QueueUrl=q, MessageBody='{}')
    orig = A.save_batch
    A.save_batch = lambda m: (_ for _ in ()).throw(OSError('disk'))
    try:
        try: A.poll_once(s); assert False
        except OSError: pass
    finally: A.save_batch = orig
    assert A.poll_once(s) == 1

def test_orders_strip_location():
    tsv = 'amazon-order-id\tship-city\tship-state\tship-postal-code\tsku\tquantity\nA1\tAustin\tTX\t78701\tS\t1\n'
    row = json.loads(A.strip_orders(tsv)[0])
    assert row['amazon-order-id'] == 'A1' and row['sku'] == 'S'
    assert 'ship-city' not in row and 'Austin' not in json.dumps(row) and '78701' not in json.dumps(row)

def test_keys_required_and_path_safe():
    c = A.app.test_client()
    assert c.get('/healthz').status_code == 200
    assert c.get('/status').status_code == 401 and c.get('/files').status_code == 401
    h = {'X-Recorder-Key': 'k'}
    assert c.get('/status', headers=h).status_code == 200
    assert c.get('/files', headers=h).status_code == 200
    assert c.get('/files/../../etc/passwd', headers=h).status_code in (404, 400)

def test_import_does_not_start_threads():
    assert A._started is False
    assert 'pid' not in A.STATE

class R:
    def __init__(self, code, data=None): self.status_code = code; self._d = data or {}
    def json(self): return self._d
    def raise_for_status(self):
        if self.status_code >= 300: raise RuntimeError(self.status_code)

def test_segments_sweep_batches_saves_and_backs_off(monkeypatch):
    A.IN_STOCK = [f'A{i:03d}' for i in range(45)]
    calls = []; waits = []
    def post(url, headers, json, timeout):
        calls.append([r['asin'] for r in json['requests']])
        assert all(r['includedData'] == ['featuredBuyingOptions', 'referencePrices'] for r in json['requests'])
        if len(calls) == 2: return R(429)
        return R(200, {'responses': [{'status': {'statusCode': 200}, 'body': {'asin': r['asin'], 'featuredBuyingOptions': []}} for r in json['requests']]})
    monkeypatch.setattr(A.requests, 'post', post)
    monkeypatch.setattr(A, 'sp_token', lambda: 't')
    before = A.STATE['segments']['saved']
    assert A.segments_sweep(wait=waits.append) is True
    assert [len(c) for c in calls] == [20, 20, 20, 5]          # the throttled batch was retried
    assert A.STATE['segments']['saved'] - before == 45 and A.STATE['segments']['throttled'] == 1
    assert A.SEG_GAP * 2 in waits and waits.count(A.SEG_GAP) == 3
    assert A.STATE['segments']['asins'] == 45

def test_snapshot_keeps_only_in_stock_asins(monkeypatch):
    monkeypatch.setattr(A, 'box_listings', lambda: {'rows': [{'asin': 'X', 'qty': 2}, {'asin': 'X', 'qty': 1}, {'asin': 'Y', 'qty': 0}, {'asin': 'Z', 'qty': None}, {'asin': None, 'qty': 3}]})
    A.snapshot_listings()
    assert A.IN_STOCK == ['X']

class G:
    def __init__(self, code, data=None, content=b''): self.status_code = code; self._d = data or {}; self.content = content
    def json(self): return self._d
    def raise_for_status(self):
        if self.status_code >= 300: raise RuntimeError(self.status_code)

def inv_row(sku, asin, ful=0, inbound=0):
    return {'sellerSku': sku, 'asin': asin, 'totalQuantity': ful, 'inventoryDetails': {'fulfillableQuantity': ful, 'inboundShippedQuantity': inbound,
            'reservedQuantity': {'totalReservedQuantity': 0}, 'unfulfillableQuantity': {'totalUnfulfillableQuantity': 0}}}

def test_inventory_walks_pages_keeps_rows_with_qty(monkeypatch):
    pages = [{'payload': {'inventorySummaries': [inv_row('S1', 'A1', 3), inv_row('S2', 'A2', 0)]}, 'pagination': {'nextToken': 'n1'}},
             {'payload': {'inventorySummaries': [inv_row('S3', 'A3', 0, inbound=5), inv_row('S4', 'A4', 2)]}}]
    seq = [G(429)] + [G(200, p) for p in pages]; seen = []
    def get(url, headers, params, timeout):
        seen.append(params.get('nextToken')); return seq.pop(0)
    monkeypatch.setattr(A.requests, 'get', get); monkeypatch.setattr(A, 'sp_token', lambda: 't')
    A.snapshot_inventory(wait=lambda s: None)
    assert seen == [None, None, 'n1']
    files = [os.path.join(r, x) for r, _, ns in os.walk(os.path.join(A.DATA, 'inventory')) for x in ns]
    snap = read(files[0])[-1]
    assert [x['sellerSku'] for x in snap['rows']] == ['S1', 'S3', 'S4'] and snap['skus_seen'] == 4
    assert A.SELLABLE == [('S1', 'A1'), ('S4', 'A4')] and A.STATE['inventory']['sellable_skus'] == 2

def test_foep_sweep_batches_of_40_with_backoff(monkeypatch):
    A.SELLABLE = [(f'S{i:03d}', f'A{i:03d}') for i in range(85)]
    calls = []; waits = []
    def post(url, headers, json, timeout):
        assert url.endswith('/offer/featuredOfferExpectedPrice')
        calls.append(len(json['requests']))
        if len(calls) == 2: return G(429)
        return G(200, {'responses': [{'status': {'statusCode': 200}, 'body': {'featuredOfferExpectedPriceResults': []}} for _ in json['requests']]})
    monkeypatch.setattr(A.requests, 'post', post); monkeypatch.setattr(A, 'sp_token', lambda: 't')
    before = A.STATE['foep']['saved']
    assert A.foep_sweep(wait=waits.append) is True
    assert calls == [40, 40, 40, 5] and A.STATE['foep']['saved'] - before == 85 and A.STATE['foep']['throttled'] >= 1
    assert waits.count(A.FOEP_GAP) == 3 and A.STATE['foep']['skus'] == 85

def test_run_report_and_health(monkeypatch):
    perf = json.dumps({'accountStatuses': [{'marketplaceId': 'ATVPDKIKX0DER', 'status': 'AT_RISK'}]}).encode()
    supp = 'Status\tReason\tSKU\nSearch Suppressed\tMissing info\tS1\nSearch Suppressed\tMissing info\tS2\n'.encode()
    docs = iter([perf, supp]); made = []
    def post(url, headers, timeout, json):
        made.append(json['reportType']); return G(202, {'reportId': 'r'})
    def get(url, headers=None, timeout=None):
        if '/documents/' in url: return G(200, {'url': 'u'})
        if url == 'u': return G(200, content=next(docs))
        return G(200, {'processingStatus': 'DONE', 'reportDocumentId': 'd'})
    monkeypatch.setattr(A.requests, 'post', post); monkeypatch.setattr(A.requests, 'get', get); monkeypatch.setattr(A, 'sp_token', lambda: 't')
    monkeypatch.setattr(A.time, 'sleep', lambda s: None)
    A.snapshot_health()
    assert made == ['GET_V2_SELLER_PERFORMANCE_REPORT', 'GET_MERCHANTS_LISTINGS_FYP_REPORT']
    assert A.STATE['health']['account_status'] == 'ATVPDKIKX0DER:AT_RISK' and A.STATE['health']['suppressed_rows'] == 2
    files = [os.path.join(r, x) for r, _, ns in os.walk(os.path.join(A.DATA, 'health')) for x in ns]
    rows = read(files[0]); assert rows[0]['body']['accountStatuses'][0]['status'] == 'AT_RISK' and 'S2' in rows[1]['tsv']

def test_keepa_batches_saves_and_respects_reserve(monkeypatch):
    os.environ['KEEPA_KEY'] = 'kk'
    A.IN_STOCK = [f'A{i:03d}' for i in range(30)]; A.SELLABLE = [('S', 'A029'), ('T', 'B999')]
    lefts = iter([A.KEEPA_RESERVE + 100, 9000]); calls = []; waits = []
    def get(url, timeout, params):
        assert params['stock'] == 1 and params['only_live_offers'] == 1 and params['key'] == 'kk'
        calls.append(params['asin'].split(','))
        return G(200, {'products': [{'asin': a} for a in calls[-1]], 'tokensConsumed': 167, 'tokensLeft': next(lefts)})
    monkeypatch.setattr(A.requests, 'get', get)
    A.pull_keepa(wait=waits.append)
    assert [len(c) for c in calls] == [20, 11]          # 30 in-stock ASINs + B999 from inventory
    assert 60 in waits                                   # paused because tokens were near the reserve
    files = [os.path.join(r, x) for r, _, ns in os.walk(os.path.join(A.DATA, 'keepa')) for x in ns]
    rows = read(files[0]); assert len(rows) == 2 and rows[0]['response']['products'][0]['asin'] == 'A000'
    assert A.STATE['keepa']['tokens_used'] == 334 and A.STATE['keepa']['asins'] == 31

def test_shadow_endpoints_and_feed():
    import test_shadow as TS
    c = A.app.test_client(); h = {'X-Recorder-Key': 'k'}
    assert c.post('/model', data='{}', headers=h).status_code == 400  # malformed model is rejected
    r = c.post('/model', data=json.dumps(TS.MODEL), headers=h); assert r.status_code == 200 and r.get_json()['trained_at'] == 'test'
    A.SH.on_listings([{'sku': 'S1', 'asin': 'A9', 'min_price': 25, 'max_price': 40, 'listed_price': 30, 'cost': 15, 'roi': 30, 'v30': 2, 'avg_bb_share': 50, 'repricing': True, 'qty': 5}])
    A.save_batch([{'MessageId': 'm1', 'Body': TS.body('A9', [(A._shadow.ME, 29.5, True), ('CALM', 30.4, False)])}])
    assert 'A9' in A.SH.pending
    A.SH.step()
    rows = c.get('/shadow?n=5', headers=h).get_json(); assert rows and rows[-1]['asin'] == 'A9' and rows[-1]['live'] is None
    st = c.get('/status', headers=h).get_json(); assert st['shadow']['decisions'] >= 1 and st['live']['env_live_mode'] in ('off', 'on')
    r = c.post('/live', json={'enabled': True, 'skus': ['S1']}, headers=h).get_json(); assert r['config']['enabled'] is True
    assert c.post('/live/stop', headers=h).get_json()['config']['enabled'] is False
    assert c.post('/live', json={}).status_code == 401

def test_repricer_skus_tuning_and_log():
    import test_shadow as TS
    c = A.app.test_client(); h = {'X-Recorder-Key': 'k'}
    c.post('/model', data=json.dumps(TS.MODEL), headers=h)
    A.SH.on_listings([{'sku': 'S7', 'asin': 'A7', 'title': 'Thing', 'min_price': 25, 'max_price': 40, 'listed_price': 30, 'cost': 15, 'roi': 30, 'v30': 2, 'avg_bb_share': 50, 'repricing': False, 'qty': 5}])
    r = c.post('/live', json={'skus': ['S7'], 'min_gain_abs': 99, 'decay_per_day': 0.5}, headers=h).get_json()
    assert r['config']['min_gain_abs'] == 50.0 and r['config']['decay_per_day'] == 0.02   # clamped
    A.SH.on_message_bodies([TS.body('A7', [(A._shadow.ME, 29.5, True), ('CALM', 30.4, False)])]); A.SH.last_decided.pop('A7', None); A.SH.step()
    rows = [x for x in c.get('/repricer/skus', headers=h).get_json() if x['sku'] == 'S7']
    assert rows and rows[0]['live'] is True and rows[0]['ss_repricing'] is False and rows[0]['decision']['action'] == 'hold'   # $50 minimum gain holds everything
    assert c.get('/shadow?sku=S7', headers=h).get_json()[-1]['sku'] == 'S7'
    assert c.get('/live/log', headers=h).status_code == 200
    c.post('/live', json={'min_gain_abs': 0.5, 'decay_per_day': 0.003, 'skus': []}, headers=h)
