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
        assert all(r['includedData'] == ['featuredBuyingOptions'] for r in json['requests'])
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
