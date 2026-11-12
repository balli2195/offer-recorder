"""Offer recorder (Nick, Nov 2026). Read-only: records what Seller Snap and the market do. It cannot change prices.
Four loops write gzip JSONL to the Render disk:
  offers/   every ANY_OFFER_CHANGED message from SQS, raw (saved to disk before it is deleted from the queue)
  listings/ Seller Snap listing snapshot, hourly, through the Seller Snap box (read action only)
  segments/ Amazon's Featured Offer breakdown (competitive summary) for every in-stock ASIN, one call per 31 s, about 15 min a sweep
  orders/   Amazon all-orders report, daily, last 8 days, with customer location columns removed
  inventory/ Amazon FBA inventory summaries (fulfillable, reserved, inbound, unfulfillable), hourly, every SKU with any quantity
  foep/     Amazon's featured offer expected price (the Buy Box target price) for every sellable SKU, a sweep every 30 min
  health/   daily seller performance report (account status, metrics) and suppressed-listings report
  keepa/    Keepa live offers with stock for in-stock ASINs, every 6 h, full responses (needs KEEPA_KEY)
Endpoints: /healthz (open); /status, /files, /files/<path> (header X-Recorder-Key).
Env: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION, QUEUE_URL, RECORDER_KEY, SS_BOX_URL, SS_BOX_KEY,
     SPAPI_CLIENT_ID, SPAPI_CLIENT_SECRET, SPAPI_REFRESH_TOKEN, optional KEEPA_KEY, DATA_DIR (default /var/data), MIN_FREE_FRAC.
"""
import csv, datetime as dt, gzip, hmac, io, json, logging, os, shutil, threading, time
import boto3, requests
from flask import Flask, abort, jsonify, request, send_from_directory

DATA = os.environ.get('DATA_DIR', '/var/data')
QUEUE = os.environ.get('QUEUE_URL', '')
REGION = os.environ.get('AWS_REGION', 'us-east-1')
KEY = os.environ.get('RECORDER_KEY', '')
MIN_FREE = float(os.environ.get('MIN_FREE_FRAC', '0.05'))
SEG_GAP = float(os.environ.get('SEG_GAP_S', '31'))  # Amazon's limit for this call is 0.033 requests/second
SNAP_EVERY = int(os.environ.get('SNAP_EVERY_S', '3600'))
ORDER_EVERY = int(os.environ.get('ORDER_EVERY_S', '86400'))
INV_EVERY = int(os.environ.get('INV_EVERY_S', '3600'))
FOEP_EVERY = int(os.environ.get('FOEP_EVERY_S', '1800'))
FOEP_GAP = float(os.environ.get('FOEP_GAP_S', '31'))  # Amazon's limit for the FOEP batch call is 0.033 requests/second, 40 SKUs a call
HEALTH_EVERY = int(os.environ.get('HEALTH_EVERY_S', '86400'))
KEEPA_EVERY = int(os.environ.get('KEEPA_EVERY_S', '21600'))
KEEPA_RESERVE = int(os.environ.get('KEEPA_RESERVE', '2500'))  # leave this many tokens for the Lead Score server
KEEPA_BATCH = 20
SP = 'https://sellingpartnerapi-na.amazon.com'
MP = 'ATVPDKIKX0DER'
ORDER_COLS = ['amazon-order-id', 'purchase-date', 'last-updated-date', 'order-status', 'fulfillment-channel', 'sales-channel',
              'sku', 'asin', 'item-status', 'quantity', 'currency', 'item-price', 'item-promotion-discount', 'is-business-order', 'order-item-id']
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger('recorder')
app = Flask(__name__)
STATE = {'started': None, 'offers': {'saved': 0, 'last_saved': None, 'last_sent': None, 'last_error': None, 'paused_low_disk': False},
         'segments': {'sweeps': 0, 'last_sweep': None, 'asins': None, 'saved': 0, 'throttled': 0, 'last_saved': None, 'last_error': None},
         'listings': {'last': None, 'rows': None, 'last_error': None}, 'orders': {'last': None, 'rows': None, 'last_error': None},
         'inventory': {'last': None, 'skus': None, 'sellable_skus': None, 'last_error': None},
         'foep': {'sweeps': 0, 'last_sweep': None, 'skus': None, 'saved': 0, 'throttled': 0, 'last_error': None},
         'health': {'last': None, 'account_status': None, 'suppressed_rows': None, 'last_error': None},
         'keepa': {'last': None, 'asins': None, 'tokens_used': None, 'tokens_left': None, 'last_error': None}}
LOCK = threading.Lock()
import shadow as _shadow, live as _live
IN_STOCK = []  # ASINs with FBA stock, from the latest Seller Snap snapshot
SELLABLE = []  # (sku, asin) with fulfillable quantity > 0, from the latest inventory snapshot


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


def _safe(fn, *a):
    """Shadow/live hooks never interrupt recording."""
    try:
        return fn(*a)
    except Exception:
        log.exception('shadow hook')


def save_batch(msgs):
    t = now()
    lines = [json.dumps({'received_at': iso(t), 'message_id': m['MessageId'], 'sent_ms': (m.get('Attributes') or {}).get('SentTimestamp'), 'body': m['Body']}) for m in msgs]
    append_gz(os.path.join(DATA, 'offers', t.strftime('%Y-%m-%d'), t.strftime('%H') + '.jsonl.gz'), lines)
    if SH:
        _safe(SH.on_message_bodies, [m['Body'] for m in msgs])


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
    asins = sorted({r['asin'] for r in rows if r.get('asin') and (r.get('qty') or 0) > 0})
    if SH:
        _safe(SH.on_listings, rows)
        _safe(LV.on_listings, rows)
    global IN_STOCK
    with LOCK:
        IN_STOCK = asins
        STATE['listings'].update(last=iso(t), rows=len(rows), last_error=None)


def sp_token():
    r = requests.post('https://api.amazon.com/auth/o2/token', data={'grant_type': 'refresh_token', 'refresh_token': os.environ['SPAPI_REFRESH_TOKEN'],
                      'client_id': os.environ['SPAPI_CLIENT_ID'], 'client_secret': os.environ['SPAPI_CLIENT_SECRET']}, timeout=30)
    r.raise_for_status()
    return r.json()['access_token']


SH = _shadow.Shadow(DATA, append_gz, iso)
LV = _live.Live(DATA, append_gz, iso, sp_token)
SH.on_decision = LV.maybe_apply
SH.tune_fn = lambda: LV.cfg


def segment_batch(asins, token):
    """One competitive summary call for up to 20 ASINs. Saves each ASIN's raw response. Returns False if Amazon throttled us."""
    b = {'requests': [{'asin': a, 'marketplaceId': MP, 'includedData': ['featuredBuyingOptions'], 'method': 'GET',
                       'uri': '/products/pricing/2022-05-01/items/competitiveSummary'} for a in asins]}
    r = requests.post(f'{SP}/batches/products/pricing/2022-05-01/items/competitiveSummary',
                      headers={'x-amz-access-token': token, 'content-type': 'application/json'}, json=b, timeout=60)
    if r.status_code == 429:
        with LOCK:
            STATE['segments']['throttled'] += 1
        return False
    r.raise_for_status()
    t = now()
    lines = []
    for a, x in zip(asins, r.json().get('responses') or []):
        body = x.get('body') or {}
        lines.append(json.dumps({'taken_at': iso(t), 'asin': body.get('asin') or a, 'status': (x.get('status') or {}).get('statusCode'), 'body': body}))
    append_gz(os.path.join(DATA, 'segments', t.strftime('%Y-%m-%d'), t.strftime('%H') + '.jsonl.gz'), lines)
    with LOCK:
        STATE['segments']['saved'] += len(lines)
        STATE['segments']['last_saved'] = iso(t)
    return True


def segments_sweep(wait=time.sleep):
    with LOCK:
        asins = list(IN_STOCK)
    if not asins:
        return False
    token = sp_token()
    for i in range(0, len(asins), 20):
        batch = asins[i:i + 20]
        tries = 0
        while True:
            if free_frac() < MIN_FREE:
                return False
            if segment_batch(batch, token):
                wait(SEG_GAP)
                break
            tries += 1
            wait(min(300, SEG_GAP * 2 ** tries))
            if tries >= 5:  # skip this batch; the next sweep gets it
                break
    with LOCK:
        STATE['segments'].update(sweeps=STATE['segments']['sweeps'] + 1, last_sweep=iso(), asins=len(asins), last_error=None)
    return True


def segments_loop():
    if not all(os.environ.get(n) for n in ('SPAPI_CLIENT_ID', 'SPAPI_CLIENT_SECRET', 'SPAPI_REFRESH_TOKEN')):
        with LOCK:
            STATE['segments']['last_error'] = 'not configured'
        return
    while True:
        try:
            if not segments_sweep():
                time.sleep(60)
        except Exception as e:
            log.exception('segments')
            with LOCK:
                STATE['segments']['last_error'] = f'{iso()} {type(e).__name__}: {str(e)[:200]}'
            time.sleep(120)


def strip_orders(tsv):
    """Keeps only ORDER_COLS, so no customer city, state, or postal code is ever stored."""
    out = []
    for row in csv.DictReader(io.StringIO(tsv), delimiter='\t'):
        out.append(json.dumps({c: row.get(c) for c in ORDER_COLS}))
    return out


def run_report(report_type, start=None, end=None, options=None, wait=time.sleep):
    """Requests one Amazon report, waits for it, and returns the document bytes (unzipped)."""
    h = {'x-amz-access-token': sp_token(), 'content-type': 'application/json'}
    body = {'reportType': report_type, 'marketplaceIds': [MP]}
    if start:
        body['dataStartTime'] = iso(start)
    if end:
        body['dataEndTime'] = iso(end)
    if options:
        body['reportOptions'] = options
    for k in range(6):
        r = requests.post(f'{SP}/reports/2021-06-30/reports', headers=h, timeout=60, json=body)
        if r.status_code != 429:
            break
        wait(65)
    r.raise_for_status()
    rid = r.json()['reportId']
    for _ in range(120):
        wait(15)
        s = requests.get(f'{SP}/reports/2021-06-30/reports/{rid}', headers=h, timeout=30)
        s.raise_for_status()
        s = s.json()
        if s.get('processingStatus') in ('DONE', 'FATAL', 'CANCELLED'):
            break
    if s.get('processingStatus') != 'DONE':
        raise RuntimeError(f'{report_type} report ' + str(s.get('processingStatus')))
    h = {'x-amz-access-token': sp_token()}  # a long wait can outlive the first token
    d = requests.get(f"{SP}/reports/2021-06-30/documents/{s['reportDocumentId']}", headers=h, timeout=30)
    d.raise_for_status()
    d = d.json()
    b = requests.get(d['url'], timeout=300).content
    if d.get('compressionAlgorithm') == 'GZIP':
        b = gzip.decompress(b)
    return b


def pull_orders():
    end = now()
    b = run_report('GET_FLAT_FILE_ALL_ORDERS_DATA_BY_ORDER_DATE_GENERAL', end - dt.timedelta(days=8), end - dt.timedelta(minutes=5))
    lines = strip_orders(b.decode('latin-1'))
    write_once(os.path.join(DATA, 'orders', end.strftime('%Y-%m-%d') + '.jsonl.gz'), lines)
    with LOCK:
        STATE['orders'].update(last=iso(), rows=len(lines), last_error=None)


def has_qty(x):
    d = x.get('inventoryDetails') or {}
    vals = [x.get('totalQuantity'), d.get('fulfillableQuantity'), d.get('inboundWorkingQuantity'), d.get('inboundShippedQuantity'),
            d.get('inboundReceivingQuantity'), (d.get('reservedQuantity') or {}).get('totalReservedQuantity'),
            (d.get('unfulfillableQuantity') or {}).get('totalUnfulfillableQuantity'), (d.get('researchingQuantity') or {}).get('totalResearchingQuantity')]
    return any((v or 0) > 0 for v in vals)


def snapshot_inventory(wait=time.sleep):
    """Every page of FBA inventory summaries; keeps SKUs with any quantity (a SKU missing from a snapshot had none)."""
    token, nt, rows, seen, t, renewed = sp_token(), None, [], 0, now(), 0
    while True:
        p = {'details': 'true', 'granularityType': 'Marketplace', 'granularityId': MP, 'marketplaceIds': MP}
        if nt:
            p['nextToken'] = nt
        r = requests.get(f'{SP}/fba/inventory/v1/summaries', headers={'x-amz-access-token': token}, params=p, timeout=60)
        if r.status_code == 429:
            wait(5)
            continue
        if r.status_code == 403 and renewed < 3:  # token expired mid-walk
            token, renewed = sp_token(), renewed + 1
            continue
        r.raise_for_status()
        j = r.json()
        inv = (j.get('payload') or {}).get('inventorySummaries') or []
        seen += len(inv)
        rows += [x for x in inv if has_qty(x)]
        nt = (j.get('pagination') or {}).get('nextToken')
        if not nt:
            break
        wait(0.6)  # limit is 2 requests/second
    append_gz(os.path.join(DATA, 'inventory', t.strftime('%Y-%m-%d') + '.jsonl.gz'), [json.dumps({'taken_at': iso(t), 'skus_seen': seen, 'rows': rows})])
    sell = sorted({(x['sellerSku'], x.get('asin')) for x in rows if ((x.get('inventoryDetails') or {}).get('fulfillableQuantity') or 0) > 0})
    if SH:
        _safe(SH.on_inventory, rows)
    global SELLABLE
    with LOCK:
        SELLABLE = sell
        STATE['inventory'].update(last=iso(t), skus=len(rows), sellable_skus=len(sell), last_error=None)


def foep_batch(skus, token):
    """One featured-offer-expected-price call for up to 40 SKUs. Returns False if throttled."""
    b = {'requests': [{'uri': '/products/pricing/2022-05-01/offer/featuredOfferExpectedPrice', 'method': 'GET', 'marketplaceId': MP, 'sku': s} for s in skus]}
    r = requests.post(f'{SP}/batches/products/pricing/2022-05-01/offer/featuredOfferExpectedPrice',
                      headers={'x-amz-access-token': token, 'content-type': 'application/json'}, json=b, timeout=60)
    if r.status_code == 429:
        with LOCK:
            STATE['foep']['throttled'] += 1
        return False
    r.raise_for_status()
    t = now()
    lines = [json.dumps({'taken_at': iso(t), 'sku': s, 'status': (x.get('status') or {}).get('statusCode'), 'body': x.get('body')})
             for s, x in zip(skus, r.json().get('responses') or [])]
    append_gz(os.path.join(DATA, 'foep', t.strftime('%Y-%m-%d'), t.strftime('%H') + '.jsonl.gz'), lines)
    if SH:
        for s_, x in zip(skus, r.json().get('responses') or []):
            _safe(SH.on_foep, s_, x.get('body') or {})
    with LOCK:
        STATE['foep']['saved'] += len(lines)
    return True


def foep_sweep(wait=time.sleep):
    with LOCK:
        skus = [s for s, a in SELLABLE]
    if not skus:
        return False
    token, t0 = sp_token(), time.time()
    for i in range(0, len(skus), 40):
        if time.time() - t0 > 2700:  # refresh the 1-hour token
            token, t0 = sp_token(), time.time()
        tries = 0
        while True:
            if free_frac() < MIN_FREE:
                return False
            if foep_batch(skus[i:i + 40], token):
                wait(FOEP_GAP)
                break
            tries += 1
            wait(min(300, FOEP_GAP * 2 ** tries))
            if tries >= 5:
                break
    with LOCK:
        STATE['foep'].update(sweeps=STATE['foep']['sweeps'] + 1, last_sweep=iso(), skus=len(skus), last_error=None)
    return True


def foep_loop():
    while True:
        t0 = time.time()
        try:
            if not foep_sweep():
                time.sleep(60)
                continue
        except Exception as e:
            log.exception('foep')
            with LOCK:
                STATE['foep']['last_error'] = f'{iso()} {type(e).__name__}: {str(e)[:200]}'
            time.sleep(120)
            continue
        time.sleep(max(0, FOEP_EVERY - (time.time() - t0)))


def snapshot_health():
    t = now()
    perf = json.loads(run_report('GET_V2_SELLER_PERFORMANCE_REPORT').decode('utf-8-sig'))
    supp = run_report('GET_MERCHANTS_LISTINGS_FYP_REPORT').decode('utf-8-sig', errors='replace')
    status = ','.join(f"{x.get('marketplaceId')}:{x.get('status')}" for x in perf.get('accountStatuses') or [])
    nrows = max(0, len([l for l in supp.splitlines() if l.strip()]) - 1)
    write_once(os.path.join(DATA, 'health', t.strftime('%Y-%m-%d') + '.jsonl.gz'),
               [json.dumps({'taken_at': iso(t), 'report': 'seller_performance', 'body': perf}),
                json.dumps({'taken_at': iso(t), 'report': 'suppressed_listings', 'tsv': supp})])
    with LOCK:
        STATE['health'].update(last=iso(t), account_status=status, suppressed_rows=nrows, last_error=None)


def keepa_asins():
    with LOCK:
        return sorted(set(IN_STOCK) | {a for s, a in SELLABLE if a})


def pull_keepa(wait=time.sleep):
    """Live offers with stock for every in-stock ASIN, 20 a call. Saves each full response. Waits rather than dip below KEEPA_RESERVE tokens."""
    asins = keepa_asins()
    if not asins:
        raise RuntimeError('no in-stock ASINs yet')
    t, used, left = now(), 0, None
    path = os.path.join(DATA, 'keepa', t.strftime('%Y-%m-%d'), t.strftime('%H') + '.jsonl.gz')
    for i in range(0, len(asins), KEEPA_BATCH):
        b = asins[i:i + KEEPA_BATCH]
        for k in range(30):
            if left is not None and left < KEEPA_RESERVE + 200:
                wait(60)
                left = None  # recheck from the next response
            r = requests.get('https://api.keepa.com/product', timeout=300, params={'key': os.environ['KEEPA_KEY'], 'domain': 1, 'asin': ','.join(b),
                             'offers': 20, 'stock': 1, 'only_live_offers': 1, 'history': 1, 'days': 1})
            j = r.json()
            left = j.get('tokensLeft', left)
            if r.status_code == 429 or not j.get('products'):
                wait(max(60, (j.get('refillIn') or 60000) / 1000))
                continue
            break
        else:
            continue
        used += j.get('tokensConsumed') or 0
        append_gz(path, [json.dumps({'taken_at': iso(), 'asins': b, 'tokensConsumed': j.get('tokensConsumed'), 'tokensLeft': j.get('tokensLeft'), 'response': j})])
        if SH:
            _safe(SH.on_keepa_products, j.get('products'))
        with LOCK:
            STATE['keepa'].update(tokens_used=used, tokens_left=j.get('tokensLeft'))
    with LOCK:
        STATE['keepa'].update(last=iso(t), asins=len(asins), last_error=None)


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


def delayed(seconds, fn, *args):
    """Keepa waits for the first inventory and Seller Snap snapshots, so it knows which ASINs are in stock."""
    time.sleep(seconds)
    fn(*args)


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
    if SH:
        s['shadow'] = dict(SH.stats, asins_tracked=len(SH.offers), skus_priced=len(SH.listings), foep_known=len(SH.foep), rival_stock_known=len(SH.rival_stock))
        s['live'] = dict(LV.stats, env_live_mode=os.environ.get('LIVE_MODE', 'off'), config={k: (v if k != 'skus' else len(v)) for k, v in LV.cfg.items()})
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


@app.post('/model')
def model_upload():
    need_key()
    try:
        SH.save_model(request.get_data(as_text=True))
    except (ValueError, KeyError) as e:
        return jsonify({'ok': False, 'error': str(e)}), 400
    return jsonify({'ok': True, 'trained_at': SH.stats['model_trained_at']})


@app.get('/shadow')
def shadow_recent():
    need_key()
    n = int(request.args.get('n', 50))
    act, asin, sku = request.args.get('action'), request.args.get('asin'), request.args.get('sku')
    rows = [d for d in list(SH.recent) if (not act or d['action'] == act) and (not asin or d['asin'] == asin) and (not sku or d['sku'] == sku)]
    return jsonify(rows[-n:])


@app.get('/repricer/skus')
def repricer_skus():
    """Every priced SKU with its Seller Snap settings and the latest shadow decision, for the hub's Repricer page."""
    need_key()
    live_set = set(LV.cfg.get('skus') or [])
    out = []
    for asin, s in list(SH.listings.items()):
        d = SH.latest.get(asin) or {}
        out.append({'asin': asin, 'sku': s['sku'], 'title': s.get('title'), 'min': s['min'], 'max': s['max'], 'cost': s['cost'], 'qty': s.get('qty'),
                    'ss_repricing': s.get('ss_repricing'), 'live': s['sku'] in live_set, 'fulfillable': SH.stock.get(s['sku']),
                    'decision': {k: d.get(k) for k in ('t', 'action', 'my_price', 'rec_price', 'low_rival', 'foep', 'gain_24h', 'winner')} if d else None})
    return jsonify(out)


@app.get('/live/log')
def live_log():
    need_key()
    return jsonify(LV.recent_log(int(request.args.get('days', 3)))[:int(request.args.get('n', 200))])


@app.get('/live')
def live_get():
    need_key()
    return jsonify({'env_live_mode': os.environ.get('LIVE_MODE', 'off'), 'config': LV.cfg, 'stats': LV.stats})


@app.post('/live')
def live_set():
    need_key()
    return jsonify({'ok': True, 'config': LV.set_cfg(request.get_json(force=True) or {})})


@app.post('/live/stop')
def live_stop():
    need_key()
    return jsonify({'ok': True, 'config': LV.stop()})


@app.post('/live/preview')
def live_preview():
    """Amazon validates a price change for one SKU without applying it (VALIDATION_PREVIEW). Never changes a price."""
    need_key()
    b = request.get_json(force=True) or {}
    code, resp, body = LV.send(str(b['sku']), float(b['price']), preview=True)
    return jsonify({'status_code': code, 'response': resp, 'sent': body})


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
    threading.Thread(target=SH.loop, daemon=True).start()
    STATE['pid'] = os.getpid()
    threading.Thread(target=offers_loop, daemon=True).start()
    threading.Thread(target=segments_loop, daemon=True).start()
    threading.Thread(target=every, args=('listings', SNAP_EVERY, snapshot_listings, ['SS_BOX_URL', 'SS_BOX_KEY']), daemon=True).start()
    sp = ['SPAPI_CLIENT_ID', 'SPAPI_CLIENT_SECRET', 'SPAPI_REFRESH_TOKEN']
    threading.Thread(target=every, args=('orders', ORDER_EVERY, pull_orders, sp), daemon=True).start()
    threading.Thread(target=every, args=('inventory', INV_EVERY, snapshot_inventory, sp), daemon=True).start()
    if all(os.environ.get(n) for n in sp):
        threading.Thread(target=foep_loop, daemon=True).start()
    else:
        STATE['foep']['last_error'] = 'not configured'
    threading.Thread(target=every, args=('health', HEALTH_EVERY, snapshot_health, sp), daemon=True).start()
    threading.Thread(target=delayed, args=(900, every, 'keepa', KEEPA_EVERY, pull_keepa, ['KEEPA_KEY']), daemon=True).start()
