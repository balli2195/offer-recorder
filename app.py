"""Offer recorder (Nick, Nov 2026). Read-only: records what Seller Snap and the market do. It cannot change prices.
Three loops write gzip JSONL to the Render disk:
  offers/   every ANY_OFFER_CHANGED message from SQS, raw (saved to disk before it is deleted from the queue)
  listings/ Seller Snap listing snapshot, hourly, through the Seller Snap box (read action only)
  orders/   Amazon all-orders report, daily, last 8 days, with customer location columns removed
Endpoints: /healthz (open); /status, /files, /files/<path> (header X-Recorder-Key).
Env: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION, QUEUE_URL, RECORDER_KEY, SS_BOX_URL, SS_BOX_KEY,
     SPAPI_CLIENT_ID, SPAPI_CLIENT_SECRET, SPAPI_REFRESH_TOKEN, optional DATA_DIR (default /var/data), MIN_FREE_FRAC.
"""
import csv, datetime as dt, gzip, hmac, io, json, logging, os, shutil, threading, time
import boto3, requests
from flask import Flask, abort, jsonify, request, send_from_directory

DATA = os.environ.get('DATA_DIR', '/var/data')
QUEUE = os.environ.get('QUEUE_URL', '')
REGION = os.environ.get('AWS_REGION', 'us-east-1')
KEY = os.environ.get('RECORDER_KEY', '')
MIN_FREE = float(os.environ.get('MIN_FREE_FRAC', '0.05'))
SNAP_EVERY = int(os.environ.get('SNAP_EVERY_S', '3600'))
ORDER_EVERY = int(os.environ.get('ORDER_EVERY_S', '86400'))
SP = 'https://sellingpartnerapi-na.amazon.com'
MP = 'ATVPDKIKX0DER'
ORDER_COLS = ['amazon-order-id', 'purchase-date', 'last-updated-date', 'order-status', 'fulfillment-channel', 'sales-channel',
              'sku', 'asin', 'item-status', 'quantity', 'currency', 'item-price', 'item-promotion-discount', 'is-business-order', 'order-item-id']
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger('recorder')
app = Flask(__name__)
STATE = {'started': None, 'offers': {'saved': 0, 'last_saved': None, 'last_sent': None, 'last_error': None, 'paused_low_disk': False},
         'listings': {'last': None, 'rows': None, 'last_error': None}, 'orders': {'last': None, 'rows': None, 'last_error': None}}
LOCK = threading.Lock()


def now():
    return dt.datetime.now(dt.timezone.utc)


def iso(d=None):
    return (d or now()).strftime('%Y-%m-%dT%H:%M:%SZ')


def free_frac():
    u = shutil.disk_usage(DATA)
    return u.free / u.total


def append_gz(path, lines):
    """Add lines as one gzip member and fsync, so the file stays valid and the data is on disk before we return."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    blob = gzip.compress(('\n'.join(lines) + '\n').encode(), compresslevel=6)
    with open(path, 'ab') as f:
        f.write(blob)
        f.flush()
        os.fsync(f.fileno())


def write_once(path, lines):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.tmp'
    with gzip.open(tmp, 'wt') as f:
        f.write('\n'.join(lines) + '\n')
    os.replace(tmp, path)


def save_batch(msgs):
    t = now()
    lines = [json.dumps({'received_at': iso(t), 'message_id': m['MessageId'], 'sent_ms': (m.get('Attributes') or {}).get('SentTimestamp'), 'body': m['Body']}) for m in msgs]
    append_gz(os.path.join(DATA, 'offers', t.strftime('%Y-%m-%d'), t.strftime('%H') + '.jsonl.gz'), lines)


def poll_once(sqs):
    """Returns the number of messages handled. Deletes only after the batch is safely on disk; a crash means duplicates, not loss."""
    r = sqs.receive_message(QueueUrl=QUEUE, MaxNumberOfMessages=10, WaitTimeSeconds=20, AttributeNames=['SentTimestamp'])
    msgs = r.get('Messages') or []
    if not msgs:
        return 0
    save_batch(msgs)
    res = sqs.delete_message_batch(QueueUrl=QUEUE, Entries=[{'Id': str(i), 'ReceiptHandle': m['ReceiptHandle']} for i, m in enumerate(msgs)])
    if res.get('Failed'):
        log.warning('delete failed for %d messages; they will come back as duplicates', len(res['Failed']))
    sent = max(int((m.get('Attributes') or {}).get('SentTimestamp') or 0) for m in msgs)
    with LOCK:
        STATE['offers']['saved'] += len(msgs)
        STATE['offers']['last_saved'] = iso()
        if sent:
            STATE['offers']['last_sent'] = iso(dt.datetime.fromtimestamp(sent / 1000, dt.timezone.utc))
    return len(msgs)


def offers_loop():
    if not QUEUE:
        log.warning('QUEUE_URL not set; offers loop off')
        return
    sqs = boto3.client('sqs', region_name=REGION)
    while True:
        try:
            low = free_frac() < MIN_FREE
            with LOCK:
                STATE['offers']['paused_low_disk'] = low
            if low:  # messages stay safe in the queue for up to 14 days
                time.sleep(300)
                continue
            poll_once(sqs)
            with LOCK:
                STATE['offers']['last_error'] = None
        except Exception as e:
            log.exception('offers loop')
            with LOCK:
                STATE['offers']['last_error'] = f'{iso()} {type(e).__name__}: {str(e)[:200]}'
            time.sleep(30)


def box_listings():
    r = requests.post(os.environ['SS_BOX_URL'], json={'key': os.environ['SS_BOX_KEY'], 'action': 'listings'}, allow_redirects=False, timeout=180)
    if r.status_code in (301, 302, 303):
        r = requests.get(r.headers['Location'], timeout=180)
    r.raise_for_status()
    j = r.json()
    if not j.get('ok'):
        raise RuntimeError('Seller Snap box error: ' + str(j.get('error'))[:200])
    return j['result']


def snapshot_listings():
    res = box_listings()
    t = now()
    rows = res['rows']
    line = json.dumps({'taken_at': iso(t), 'exportedAt': res.get('exportedAt'), 'rows': rows})
    append_gz(os.path.join(DATA, 'listings', t.strftime('%Y-%m-%d') + '.jsonl.gz'), [line])
    with LOCK:
        STATE['listings'].update(last=iso(t), rows=len(rows), last_error=None)


def sp_token():
    r = requests.post('https://api.amazon.com/auth/o2/token', data={'grant_type': 'refresh_token', 'refresh_token': os.environ['SPAPI_REFRESH_TOKEN'],
                      'client_id': os.environ['SPAPI_CLIENT_ID'], 'client_secret': os.environ['SPAPI_CLIENT_SECRET']}, timeout=30)
    r.raise_for_status()
    return r.json()['access_token']


def strip_orders(tsv):
    """Keeps only ORDER_COLS, so no customer city, state, or postal code is ever stored."""
    out = []
    for row in csv.DictReader(io.StringIO(tsv), delimiter='\t'):
        out.append(json.dumps({c: row.get(c) for c in ORDER_COLS}))
    return out


def pull_orders():
    h = {'x-amz-access-token': sp_token(), 'content-type': 'application/json'}
    end = now()
    start = end - dt.timedelta(days=8)
    r = requests.post(f'{SP}/reports/2021-06-30/reports', headers=h, timeout=60, json={'reportType': 'GET_FLAT_FILE_ALL_ORDERS_DATA_BY_ORDER_DATE_GENERAL',
                      'marketplaceIds': [MP], 'dataStartTime': iso(start), 'dataEndTime': iso(end - dt.timedelta(minutes=5))})
    r.raise_for_status()
    rid = r.json()['reportId']
    for _ in range(120):
        time.sleep(15)
        s = requests.get(f'{SP}/reports/2021-06-30/reports/{rid}', headers=h, timeout=30)
        s.raise_for_status()
        s = s.json()
        if s.get('processingStatus') in ('DONE', 'FATAL', 'CANCELLED'):
            break
    if s.get('processingStatus') != 'DONE':
        raise RuntimeError('orders report ' + str(s.get('processingStatus')))
    d = requests.get(f"{SP}/reports/2021-06-30/documents/{s['reportDocumentId']}", headers=h, timeout=30)
    d.raise_for_status()
    d = d.json()
    b = requests.get(d['url'], timeout=300).content
    if d.get('compressionAlgorithm') == 'GZIP':
        b = gzip.decompress(b)
    lines = strip_orders(b.decode('latin-1'))
    write_once(os.path.join(DATA, 'orders', end.strftime('%Y-%m-%d') + '.jsonl.gz'), lines)
    with LOCK:
        STATE['orders'].update(last=iso(), rows=len(lines), last_error=None)


def every(name, seconds, fn, needs):
    if not all(os.environ.get(n) for n in needs):
        log.warning('%s loop off: missing env', name)
        with LOCK:
            STATE[name]['last_error'] = 'not configured'
        return
    while True:
        t0 = time.time()
        try:
            if free_frac() >= MIN_FREE:
                fn()
        except Exception as e:
            log.exception(name)
            with LOCK:
                STATE[name]['last_error'] = f'{iso()} {type(e).__name__}: {str(e)[:200]}'
        time.sleep(max(60, seconds - (time.time() - t0)))


def need_key():
    if not KEY or not hmac.compare_digest(request.headers.get('X-Recorder-Key', ''), KEY):
        abort(401)


@app.get('/healthz')
def healthz():
    return 'ok'


@app.get('/status')
def status():
    need_key()
    u = shutil.disk_usage(DATA)
    with LOCK:
        s = json.loads(json.dumps(STATE))
    s['status_pid'] = os.getpid()
    s['disk'] = {'used_gb': round(u.used / 2**30, 2), 'total_gb': round(u.total / 2**30, 2), 'free_frac': round(u.free / u.total, 3)}
    return jsonify(s)


@app.get('/files')
def files():
    need_key()
    out = []
    for root, _, names in os.walk(DATA):
        for n in names:
            if n.endswith('.jsonl.gz'):
                p = os.path.join(root, n)
                out.append({'path': os.path.relpath(p, DATA), 'bytes': os.path.getsize(p)})
    return jsonify(sorted(out, key=lambda x: x['path']))


@app.get('/files/<path:p>')
def file_get(p):
    need_key()
    return send_from_directory(DATA, p, as_attachment=True)


_started = False


def start():
    """Called from gunicorn's post_fork hook (gunicorn.conf.py), never at import: if the app is preloaded, threads started
    at import would run in the master process, and the worker answering /status would never see their state."""
    global _started
    if _started:
        return
    _started = True
    os.makedirs(DATA, exist_ok=True)
    STATE['started'] = iso()
    STATE['pid'] = os.getpid()
    threading.Thread(target=offers_loop, daemon=True).start()
    threading.Thread(target=every, args=('listings', SNAP_EVERY, snapshot_listings, ['SS_BOX_URL', 'SS_BOX_KEY']), daemon=True).start()
    threading.Thread(target=every, args=('orders', ORDER_EVERY, pull_orders, ['SPAPI_CLIENT_ID', 'SPAPI_CLIENT_SECRET', 'SPAPI_REFRESH_TOKEN']), daemon=True).start()
