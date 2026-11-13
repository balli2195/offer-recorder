"""Offer recorder and repricer (Nick, Nov 2026): records what Seller Snap and the market do, and runs your repricer (shadow decisions always; price changes only for live SKUs once switched on).
Four loops write gzip JSONL to the Render disk:
  offers/   every ANY_OFFER_CHANGED message from SQS, raw (saved to disk before it is deleted from the queue)
  listings/ Seller Snap listing snapshot, hourly, through the Seller Snap box (read action only)
  segments/ Amazon's Featured Offer breakdown and reference prices (competitive summary: CompetitivePrice = the outside price above
            which Amazon suppresses the Buy Box, WasPrice) for every in-stock ASIN, one call per 31 s, about 15 min a sweep
  orders/   Amazon all-orders report, last 8 days, pulled every 30 min (one file per day, replaced each pull), customer location columns removed
  inventory/ Amazon FBA inventory summaries (fulfillable, reserved, inbound, unfulfillable), hourly, every SKU with any quantity
  foep/     Amazon's featured offer expected price (the Buy Box target price) for every sellable SKU, a sweep every 30 min
  health/   daily seller performance report (account status, metrics) and suppressed-listings report
  keepa/    Keepa live offers with stock for in-stock ASINs, every 6 h, full responses (needs KEEPA_KEY)
  traffic/  Amazon's Sales & Traffic report, last 30 days by child ASIN, daily, full JSON
  catalog/  Amazon's all-listings report (every SKU with Active / Inactive / Incomplete), every 3 h, full TSV
  fees/     Amazon's fee estimate (Product Fees API) for every sellable SKU at a price in each fee band; daily per SKU, full responses
Endpoints: /healthz (open); /status, /files, /files/<path> (header X-Recorder-Key).
Env: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION, QUEUE_URL, RECORDER_KEY, SS_BOX_URL, SS_BOX_KEY,
     SPAPI_CLIENT_ID, SPAPI_CLIENT_SECRET, SPAPI_REFRESH_TOKEN, optional KEEPA_KEY, DATA_DIR (default /var/data), MIN_FREE_FRAC.
"""
import csv, datetime as dt, gzip, hmac, io, json, logging, os, resource, shutil, subprocess, sys, threading, time
from zoneinfo import ZoneInfo
import boto3, requests
from flask import Flask, Response, abort, jsonify, request, send_from_directory

DATA = os.environ.get('DATA_DIR', '/var/data')
QUEUE = os.environ.get('QUEUE_URL', '')
REGION = os.environ.get('AWS_REGION', 'us-east-1')
KEY = os.environ.get('RECORDER_KEY', '')
MIN_FREE = float(os.environ.get('MIN_FREE_FRAC', '0.05'))
SEG_GAP = float(os.environ.get('SEG_GAP_S', '31'))  # Amazon's limit for this call is 0.033 requests/second
SNAP_EVERY = int(os.environ.get('SNAP_EVERY_S', '3600'))
ORDER_EVERY = int(os.environ.get('ORDER_EVERY_S', '1800'))   # every 30 min: the repricer's 'a sale since the last cut?' check and its sales pace (Nick, Nov 13)
INV_EVERY = int(os.environ.get('INV_EVERY_S', '3600'))
FOEP_EVERY = int(os.environ.get('FOEP_EVERY_S', '1800'))
FOEP_GAP = float(os.environ.get('FOEP_GAP_S', '31'))  # Amazon's limit for the FOEP batch call is 0.033 requests/second, 40 SKUs a call
HEALTH_EVERY = int(os.environ.get('HEALTH_EVERY_S', '86400'))
KEEPA_EVERY = int(os.environ.get('KEEPA_EVERY_S', '21600'))
KEEPA_RESERVE = int(os.environ.get('KEEPA_RESERVE', '2500'))  # leave this many tokens for the Lead Score server
KEEPA_BATCH = 20
YEAR_EVERY = 7 * 86400 - 4 * 3600   # Keepa year pull: weekly (Nick, Nov 12); the 4 h slack keeps it from drifting a pull later each week
YEAR_ASINS = {}                      # asin -> last time it was pulled with a year of history
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
         'keepa': {'last': None, 'asins': None, 'tokens_used': None, 'tokens_left': None, 'last_error': None},
         'fees': {'last': None, 'skus': None, 'refreshed': None, 'sellable': None, 'last_error': None},
         'catalog': {'last': None, 'skus': None, 'by_status': None, 'last_error': None},
         'traffic': {'last': None, 'asins': None, 'last_error': None},
         'limits': {'last': None, 'costs': None, 'overrides': None, 'last_error': None},
         'history': {'last': None, 'last_error': None},
         'rivals': {'last': None, 'seconds': None, 'events': None, 'last_error': None},
         'retrain': {'last': None, 'state': None, 'verdict': None, 'reasons': None, 'new_auc': None, 'current_auc': None, 'seconds': None,
                     'peak_mb': None, 'next': None, 'last_error': None}}
# What each stream is, for the Hub's system map (GET /components). A new loop needs an entry here and in start().
STREAMS = {
    'offers': ('Offer changes', 'continuous', 'Every Amazon offer-change message for your listings (SQS), saved raw before it leaves the queue.'),
    'segments': ('Buy Box segments and reference prices', 'a sweep about every 15 min', "Amazon's featured-offer breakdown by customer segment, and its reference prices (the competitive price that decides Buy Box suppression, and the was price), for every in-stock ASIN."),
    'listings': ('Seller Snap snapshot', 'hourly', 'Min, max, price, cost, ROI, sales speed and Buy Box share for every SKU, read through the Seller Snap box.'),
    'orders': ('Orders', 'daily', 'Last 8 days of the all-orders report, customer location removed.'),
    'inventory': ('FBA inventory', 'hourly', 'Fulfillable, reserved, inbound and unfulfillable units for every SKU with stock.'),
    'foep': ('Buy Box target price', 'every 30 min', "Amazon's featured offer expected price (the Buy Box target) for every sellable SKU."),
    'health': ('Account health', 'daily', 'Seller performance report and suppressed listings.'),
    'keepa': ('Keepa rival stock and year of prices', 'every 6 h; a year of history weekly', 'Live offers with stock for in-stock ASINs, full Keepa responses. Once a week (and for a new in-stock ASIN, at its next pull) a year of price history, for the max price rule.'),
    'fees': ('Amazon fees', 'daily per SKU, hourly for new SKUs', "Amazon's own fee estimate for every sellable SKU at a price under $10, $10-$50 and over $50, used by the repricer."),
    'catalog': ('Amazon listing status', 'every 3 h', "Amazon's all-listings report: every SKU on the account with Active / Inactive / Incomplete, so the Repricer page can show each listing's state."),
    'traffic': ('Sales & Traffic', 'daily', "Amazon's Sales & Traffic report for the last 30 days by ASIN (units ordered, Buy Box %, page views): the repricer's sales speed and Buy Box share."),
    'limits': ('Price limits from the Hub', 'hourly (pushed by the Hub)', 'Unit costs (InventoryLab, order ledger, lead cost), min/max overrides (liquidation, manual) and the limit settings, sent by the Hub.'),
    'rivals': ('Hourly rival refresh', 'hourly', 'Recounts how often and how fast each rival answers undercuts (overall and per listing) and the price-war table from the last 7 days, so new or newly aggressive rivals are learned within the hour.'),
    'retrain': ('Nightly model retraining', 'daily, 3:17 AM Central', 'Refits the Buy Box, rival-reaction and price-war models on the last 7 days of recorded data, in a separate process on this server, and swaps them in only if the new Buy Box model scores at least as well as the current one on the same held-out 2 days.'),
    'abtest': ('Repricer vs Seller Snap test', 'daily switch, 3 AM Central', 'Alternating-day test: groups A and B swap between the repricer and Seller Snap each day, both on Seller Snap\'s min/max; decisions are tagged with their arm.'),
    'shadow': ('Repricer (shadow)', 'about 2 min after each offer change', 'Works out the price it would set for each listing and logs it; changes nothing.'),
    'live': ('Repricer (live)', 'with each decision', 'Changes prices only for the live SKUs, with Render LIVE_MODE=on and Seller Snap repricing off; off by default.'),
}
LOCK = threading.Lock()
import shadow as _shadow, live as _live, fees as _fees, abtest as _abtest, history as _history
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
    if HI:   # first, so the repricer's 'held the Buy Box lately?' check sees this batch
        _safe(HI.add, [m['Body'] for m in msgs])
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
AB = _abtest.ABTest(DATA, append_gz, iso, live=LV)
try:
    HI = _history.History(DATA)
except Exception:   # the chart index is a convenience; recording never depends on it
    log.exception('history index')
    HI = None
SH.ab = LV.ab = AB


def cut_signals(sku, asin, since):
    """For live.py's inching rule: did Nick hold the Buy Box (offer messages) or sell a unit (30-minute order pulls) since `since`?"""
    held = bool(HI and asin and HI.held(asin, since, _shadow.ME))
    return held, SH.units_since(sku, since)


LV.signals = cut_signals
SH.tune_fn = lambda: LV.cfg
FE = _fees.Fees(DATA, append_gz, iso, sp_token, on_fees=SH.on_fees)
FEES_EVERY = int(os.environ.get('FEES_EVERY_S', '3600'))


def pull_fees():
    """Amazon fee estimates for sellable SKUs that are new or over a day old (fees.py)."""
    with LOCK:
        sell = list(SELLABLE)
    by_asin = dict(SH.listings)
    todo = [(s, (by_asin.get(a) or {}).get('listed')) for s, a in sell]
    n = FE.pull(todo)
    with LOCK:
        STATE['fees'].update(FE.stats, refreshed=n, sellable=len(todo))


def segment_batch(asins, token):
    """One competitive summary call for up to 20 ASINs. Saves each ASIN's raw response. Returns False if Amazon throttled us."""
    b = {'requests': [{'asin': a, 'marketplaceId': MP, 'includedData': ['featuredBuyingOptions', 'referencePrices'], 'method': 'GET',
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
        if SH and body:
            _safe(SH.on_competitive_summary, body.get('asin') or a, body)
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


def order_units(lines):
    """{sku: [(epoch, units)]} from stripped order lines: Amazon.com orders, cancelled ones left out."""
    out = {}
    for ln in lines:
        try:
            r = json.loads(ln)
            if (r.get('order-status') or '').lower() == 'cancelled' or (r.get('item-status') or '').lower() == 'cancelled':
                continue
            if (r.get('sales-channel') or 'Amazon.com') != 'Amazon.com' or not r.get('sku'):
                continue
            q = int(r.get('quantity') or 0)
            if q <= 0:
                continue
            t = dt.datetime.fromisoformat(r['purchase-date'].replace('Z', '+00:00')).timestamp()
            out.setdefault(r['sku'], []).append((t, q))
        except (ValueError, KeyError, TypeError):
            continue
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
    write_once(os.path.join(DATA, 'orders', end.strftime('%Y-%m-%d') + '.jsonl.gz'), lines)   # one file per day, replaced by each pull
    _safe(SH.on_orders, order_units(lines))
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
    if SH:
        _safe(SH.on_fyp, tsv_rows(supp))
    write_once(os.path.join(DATA, 'health', t.strftime('%Y-%m-%d') + '.jsonl.gz'),
               [json.dumps({'taken_at': iso(t), 'report': 'seller_performance', 'body': perf}),
                json.dumps({'taken_at': iso(t), 'report': 'suppressed_listings', 'tsv': supp})])
    with LOCK:
        STATE['health'].update(last=iso(t), account_status=status, suppressed_rows=nrows, last_error=None)


CATALOG_EVERY = int(os.environ.get('CATALOG_EVERY_S', '10800'))


def tsv_rows(text):
    return list(csv.DictReader(io.StringIO(text.lstrip('\ufeff')), delimiter='\t'))


def snapshot_catalog():
    """Amazon's all-listings report, saved in full; gives each SKU's listing status for the Repricer page."""
    t = now()
    tsv = run_report('GET_MERCHANT_LISTINGS_ALL_DATA').decode('utf-8-sig', errors='replace')
    write_once(os.path.join(DATA, 'catalog', t.strftime('%Y-%m-%d'), t.strftime('%H') + '.jsonl.gz'), [json.dumps({'taken_at': iso(t), 'tsv': tsv})])
    rows = tsv_rows(tsv)
    if SH:
        _safe(SH.on_catalog, rows)
    by = {}
    for r in rows:
        by[r.get('status') or '?'] = by.get(r.get('status') or '?', 0) + 1
    with LOCK:
        STATE['catalog'].update(last=iso(t), skus=len(rows), by_status=by, last_error=None)


TRAFFIC_EVERY = int(os.environ.get('TRAFFIC_EVERY_S', '86400'))


def pull_traffic():
    """Amazon Sales & Traffic for the last 30 days by child ASIN, saved in full; units and Buy Box % feed the repricer."""
    t = now()
    end = (t - dt.timedelta(days=1)).replace(hour=23, minute=59, second=59, microsecond=0)
    start = (end - dt.timedelta(days=29)).replace(hour=0, minute=0, second=0)
    b = run_report('GET_SALES_AND_TRAFFIC_REPORT', start, end, options={'dateGranularity': 'DAY', 'asinGranularity': 'CHILD'})
    j = json.loads(b.decode('utf-8-sig'))
    write_once(os.path.join(DATA, 'traffic', t.strftime('%Y-%m-%d') + '.jsonl.gz'), [json.dumps({'taken_at': iso(t), 'start': iso(start), 'end': iso(end), 'body': j})])
    by = {}
    for a in j.get('salesAndTrafficByAsin') or []:
        tr, sa = a.get('trafficByAsin') or {}, a.get('salesByAsin') or {}
        k = a.get('childAsin') or a.get('asin')
        if k:
            by[k] = {'units': sa.get('unitsOrdered') or 0, 'bb': tr.get('buyBoxPercentage'), 'pv': tr.get('pageViews') or 0}
    if SH:
        _safe(SH.on_traffic, by)
    with LOCK:
        STATE['traffic'].update(last=iso(t), asins=len(by), last_error=None)


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
    yfile = os.path.join(DATA, 'keepa_year_at.json')
    with LOCK:
        if not STATE['keepa'].get('year_at'):   # remembered across restarts, so a deploy doesn't trigger another year pull
            try:
                y = json.load(open(yfile))
                STATE['keepa']['year_at'] = y['t']; YEAR_ASINS.update(y.get('asins') or {})
            except (OSError, ValueError, KeyError):
                pass
        year = time.time() - (STATE['keepa'].get('year_at') or 0) > YEAR_EVERY   # weekly: a year of history (same tokens) for the max price rule
        tried = dict(YEAR_ASINS)
    # Between weekly pulls, an ASIN with no year of history yet (new listing or restock) gets its year in the next 6-hourly pull,
    # once a week at most, so its max comes from the year high rather than 2 x min.
    if year:
        need = set(asins)
    else:
        have = set(SH.fba_high) if SH else set()
        need = {a for a in asins if a not in have and time.time() - tried.get(a, 0) > YEAR_EVERY}
    plan = [(sorted(need)[i:i + KEEPA_BATCH], 365) for i in range(0, len(need), KEEPA_BATCH)] + \
           [(rest[i:i + KEEPA_BATCH], 1) for rest in [[a for a in asins if a not in need]] for i in range(0, len(rest), KEEPA_BATCH)]
    for b, days in plan:
        for k in range(30):
            if left is not None and left < KEEPA_RESERVE + 200:
                wait(60)
                left = None  # recheck from the next response
            r = requests.get('https://api.keepa.com/product', timeout=300, params={'key': os.environ['KEEPA_KEY'], 'domain': 1, 'asin': ','.join(b),
                             'offers': 20, 'stock': 1, 'only_live_offers': 1, 'history': 1, 'days': days})
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
        if days == 365:
            for a in b:
                tried[a] = time.time()
        with LOCK:
            STATE['keepa'].update(tokens_used=used, tokens_left=j.get('tokensLeft'))
    with LOCK:
        YEAR_ASINS.update(tried)
        STATE['keepa'].update(last=iso(t), asins=len(asins), last_error=None, year_new_last=0 if year else len(need))
        if year:
            STATE['keepa']['year_at'] = time.time()
            STATE['keepa']['year_last'] = iso(t)
        try:
            with open(yfile, 'w') as f:
                json.dump({'t': STATE['keepa'].get('year_at') or 0, 'asins': tried}, f)
        except OSError:
            pass


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


RETRAIN_AT = (3, 17)          # Central time
# One math thread in the training child: OpenBLAS otherwise reserves a buffer per CPU it sees and, under the memory cap, failed with
# "Memory allocation still failed" (the 3:17 AM run on Nov 13). The data is small; one thread is plenty.
CHILD_ENV = {**os.environ, 'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
RETRAIN_TIMEOUT = int(os.environ.get('RETRAIN_TIMEOUT_MIN', '180')) * 60   # the first run on Starter (0.5 CPU) needed over 75 min
RETRAIN_NEED_MB = int(os.environ.get('RETRAIN_NEED_MB', '300'))   # a 7-day run peaked at 340 MB locally (Nov 12); the child is capped at the free memory, so a tight run fails on its own
_retrain_lock = threading.Lock()


def mem_headroom_mb():
    """Memory left in this container (cgroup limit minus current use), or None if the limit can't be read."""
    for lim, cur in (('/sys/fs/cgroup/memory.max', '/sys/fs/cgroup/memory.current'),
                     ('/sys/fs/cgroup/memory/memory.limit_in_bytes', '/sys/fs/cgroup/memory/memory.usage_in_bytes')):
        try:
            mx = open(lim).read().strip()
            if mx == 'max' or int(mx) > 1 << 50:
                return None
            return (int(mx) - int(open(cur).read().strip())) // 2**20
        except (OSError, ValueError):
            continue
    return None


def next_retrain(t=None):
    ct = dt.datetime.fromtimestamp(t or time.time(), ZoneInfo('America/Chicago'))
    nxt = ct.replace(hour=RETRAIN_AT[0], minute=RETRAIN_AT[1], second=0, microsecond=0)
    if nxt <= ct:
        nxt = (nxt + dt.timedelta(days=1)).replace(hour=RETRAIN_AT[0], minute=RETRAIN_AT[1])
    return nxt.timestamp()


def retrain():
    """Runs tools/train.py on this server's own data in a child process (low priority, so recording and the repricer keep their CPU),
    then installs the new model only if train.py's safety check says replace. Every run's report is saved to retrain/<month>.jsonl.gz."""
    if not _retrain_lock.acquire(blocking=False):
        return {'ok': False, 'error': 'already running'}
    t0 = time.time()
    try:
        with LOCK:
            STATE['retrain'].update(state='running', last_error=None)
        mdir = os.path.join(DATA, 'model')
        os.makedirs(mdir, exist_ok=True)
        cand, rep, cur = os.path.join(mdir, 'candidate.json'), os.path.join(mdir, 'retrain_report.json'), SH.model_path()
        for f in (cand, rep):
            if os.path.exists(f):
                os.remove(f)
        tmp = os.path.join(DATA, 'tmp')   # scratch left by a run that was stopped (timeout or restart)
        for n in (os.listdir(tmp) if os.path.isdir(tmp) else []):
            if n.startswith('train_'):
                shutil.rmtree(os.path.join(tmp, n), ignore_errors=True)
        room = mem_headroom_mb()
        if room is not None and room < RETRAIN_NEED_MB:
            raise RuntimeError(f'not enough memory to train: {room} MB free in the container, a run needs about {RETRAIN_NEED_MB} MB')
        cap = (room - 64) * 2**20 if room else None

        def limit():   # the child gets a memory error instead of the container being killed
            if cap:
                resource.setrlimit(resource.RLIMIT_DATA, (cap, cap))
        cmd = ['nice', '-n', '15', sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tools', 'train.py'),
               '--local', DATA, '--days', '7', '--out', cand, '--report', rep] + (['--current', cur] if os.path.exists(cur) else [])
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=RETRAIN_TIMEOUT, preexec_fn=limit, env=CHILD_ENV)
        peak = round(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024)
        tail = (r.stdout or '')[-3000:] + (r.stderr or '')[-2000:]
        if r.returncode != 0 or not os.path.exists(rep):
            raise RuntimeError(f'train.py exit {r.returncode}: {tail[-600:]}')
        report = json.load(open(rep))
        installed = False
        if report.get('verdict') == 'replace':
            SH.save_model(open(cand).read())
            installed = True
        rec = {'taken_at': iso(), 'seconds': round(time.time() - t0), 'peak_mb': peak, 'installed': installed, **report, 'log_tail': tail[-1500:]}
        append_gz(os.path.join(DATA, 'retrain', now().strftime('%Y-%m') + '.jsonl.gz'), [json.dumps(rec)])
        with LOCK:
            STATE['retrain'].update(last=iso(), state='done', last_error=None, verdict=report.get('verdict'), reasons=report.get('reasons'),
                                    new_auc=((report.get('new') or {}).get('bb_foep') or {}).get('auc'),
                                    current_auc=((report.get('current') or {}).get('bb_foep') or {}).get('auc'),
                                    seconds=rec['seconds'], peak_mb=peak, installed=installed)
        return {'ok': True, **rec}
    except Exception as e:
        log.exception('retrain')
        with LOCK:
            STATE['retrain'].update(state='failed', last_error=f'{iso()} {type(e).__name__}: {str(e)[:300]}', seconds=round(time.time() - t0))
        try:
            append_gz(os.path.join(DATA, 'retrain', now().strftime('%Y-%m') + '.jsonl.gz'), [json.dumps({'taken_at': iso(), 'error': str(e)[:2000]})])
        except OSError:
            pass
        return {'ok': False, 'error': str(e)[:300]}
    finally:
        _retrain_lock.release()


RIVALS_EVERY = 3600


def refresh_rivals():
    """Hourly: rival answer rates (overall and per listing), response times and the price-war table from the last 7 days of offer
    messages, so a new or newly aggressive rival is learned within the hour instead of at the nightly training. Same counting as training;
    keeps its message index on the data disk and reads only new or changed files. Skipped while the nightly training runs."""
    if _retrain_lock.locked():
        return {'ok': False, 'error': 'training running'}
    t0 = time.time()
    out = os.path.join(DATA, 'model', 'rivals.json')
    try:
        os.makedirs(os.path.dirname(out), exist_ok=True)
        cmd = ['nice', '-n', '15', sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tools', 'train.py'),
               '--local', DATA, '--days', '7', '--rivals-only', out, '--out', os.path.join(DATA, 'model', 'unused.json')]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30 * 60, env=CHILD_ENV)
        if r.returncode != 0:
            raise RuntimeError(f'exit {r.returncode}: {((r.stdout or "") + (r.stderr or ""))[-500:]}')
        SH.on_rivals()
        with LOCK:
            STATE['rivals'].update(last=iso(), seconds=round(time.time() - t0), last_error=None,
                                   events=(json.load(open(out)).get('events')))
        return {'ok': True}
    except Exception as e:
        with LOCK:
            STATE['rivals'].update(last_error=f'{iso()} {type(e).__name__}: {str(e)[:300]}', seconds=round(time.time() - t0))
        return {'ok': False, 'error': str(e)[:300]}


def retrain_loop():
    while True:
        nxt = next_retrain()
        with LOCK:
            STATE['retrain']['next'] = iso(dt.datetime.fromtimestamp(nxt, dt.timezone.utc))
        time.sleep(max(30, nxt - time.time()))
        retrain()


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
        s['shadow'] = dict(SH.stats, asins_tracked=len(SH.offers), skus_priced=len(SH.listings), foep_known=len(SH.foep), rival_stock_known=len(SH.rival_stock),
                           fees_known=len(SH.fees), capital_rate=SH.capital_rate,
                           suppressed_now=sum(1 for d in list(SH.latest.values()) if d.get('suppressed')))
        s['live'] = dict(LV.stats, env_live_mode=os.environ.get('LIVE_MODE', 'off'), config={k: (v if k != 'skus' else len(v)) for k, v in LV.cfg.items()})
    s['disk'] = {'used_gb': round(u.used / 2**30, 2), 'total_gb': round(u.total / 2**30, 2), 'free_frac': round(u.free / u.total, 3)}
    s['memory'] = {'free_mb': mem_headroom_mb(), 'retrain_needs_mb': RETRAIN_NEED_MB}
    s['abtest'] = AB.public()
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


@app.get('/model')
def model_get():
    need_key()
    try:
        return send_from_directory(os.path.dirname(SH.model_path()), 'model.json')
    except Exception:
        abort(404)


@app.get('/repricer/history')
def repricer_history():
    """Every new offer on one listing over the last `hours` (default 48, max 336), as change points, plus Nick's live price changes,
    the SKU's min and max, and the A/B test arm by time, for the Hub's price chart."""
    need_key()
    asin = (request.args.get('asin') or '').strip().upper()
    if not asin or not HI:
        abort(400 if not asin else 503)
    hours = max(1, min(336, int(request.args.get('hours', 48))))
    out = HI.query(asin, hours, me=_shadow.ME)
    out['live'] = [{k: r.get(k) for k in ('t', 'sku', 'from', 'to', 'action', 'response_status')}
                   for r in LV.recent_log(hours // 24 + 2) if r.get('asin') == asin and ts_iso(r.get('t')) >= out['from']]
    lst = (SH.listings.get(asin) if SH else None) or {}
    sku = lst.get('sku')
    test = sku in AB.cfg['groups']
    out.update(sku=sku, title=lst.get('title'), min=lst.get('ss_min') if test else lst.get('min'), max=lst.get('ss_max') if test else lst.get('max'),
               limits_from='Seller Snap (A/B test)' if test else 'repricer')
    bands = []
    if test and AB.cfg.get('enabled') and AB.cfg.get('start'):
        t = out['from']
        while t < out['to']:
            nxt = min(AB.next_switch(t), out['to'])
            arm = AB.arm(sku, t + 1)
            if arm:
                bands.append([round(t), round(nxt), arm])
            t = nxt
    out['arms'] = bands
    return Response(gzip.compress(json.dumps(out, separators=(',', ':')).encode()), mimetype='application/json', headers={'Content-Encoding': 'gzip'})


def ts_iso(s):
    try:
        return dt.datetime.fromisoformat(str(s).replace('Z', '+00:00')).timestamp()
    except ValueError:
        return 0


@app.get('/abtest')
def abtest_get():
    need_key()
    return jsonify(AB.public())


@app.post('/abtest')
def abtest_set():
    """Test config: enabled, dry_run, start (YYYY-MM-DD), switch_hour, washout_h, groups {sku: A|B}, asins {sku: asin}, box_url, box_key."""
    need_key()
    try:
        return jsonify(AB.set_cfg(request.get_json(force=True) or {}))
    except (ValueError, TypeError) as e:
        return jsonify({'error': str(e)}), 400


@app.post('/abtest/switch')
def abtest_switch():
    need_key()
    return jsonify(AB.switch())


@app.post('/abtest/end')
def abtest_end():
    need_key()
    return jsonify(AB.end())


@app.get('/abtest/groups')
def abtest_groups():
    need_key()
    return jsonify({'groups': AB.cfg['groups'], 'asins': AB.cfg['asins']})


@app.post('/retrain')
def retrain_now():
    """Runs the nightly retraining now, in the background. GET /status -> retrain shows the result."""
    need_key()
    if _retrain_lock.locked():
        return jsonify({'ok': False, 'error': 'already running'}), 409
    threading.Thread(target=retrain, daemon=True).start()
    return jsonify({'ok': True, 'started': iso()})


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
                    'decision': {k: d.get(k) for k in ('t', 'action', 'my_price', 'rec_price', 'low_rival', 'low_any', 'foep', 'gain_24h', 'winner', 'suppressed', 'backordered')} if d else None})
    return jsonify(out)


@app.post('/limits')
def limits_set():
    """From the Hub (hourly and on every change): unit costs, min/max overrides and limit settings."""
    need_key()
    r = SH.on_limits(request.get_json(force=True) or {})
    with LOCK:
        STATE['limits'].update(last=iso(), last_error=None, **r)
    return jsonify({'ok': True, **r, 'priced': len(SH.listings)})


@app.get('/repricer/limits_check')
def limits_check():
    """The repricer's own min/max next to Seller Snap's, SKU by SKU, for the switch-over check."""
    need_key()
    return jsonify(SH.limits_check())


@app.get('/repricer/table')
def repricer_table():
    """Every SKU (Seller Snap + Amazon's listing report) with listing state, stock, Seller Snap settings, live flag and latest decision."""
    need_key()
    return jsonify({'catalog_at': iso(dt.datetime.fromtimestamp(SH.catalog_t, dt.timezone.utc)) if SH.catalog_t else None,
                    'rows': SH.table(LV.cfg.get('skus') or [])})


@app.get('/repricer/suppressed')
def repricer_suppressed():
    """In-stock listings with the Buy Box suppressed now, with Amazon's competitive price vs min, break-even and cost."""
    need_key()
    return jsonify(SH.suppressed_list())


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


@app.post('/live/unfreeze')
def live_unfreeze():
    """Lets a SKU that the chase brake paused take price cuts again."""
    need_key()
    b = request.get_json(force=True) or {}
    return jsonify({'ok': True, 'frozen': LV.unfreeze(str(b['sku']))})


@app.get('/components')
def components():
    """Each recording stream and repricer part with what it does, how often, when it last ran and its last error (Hub system map)."""
    need_key()
    with LOCK:
        st = json.loads(json.dumps(STATE))
    st['shadow'] = {'last': SH.stats.get('last_decision'), 'last_error': SH.stats.get('last_error')}
    st['live'] = {'last': LV.stats.get('last_applied'), 'last_error': LV.stats.get('last_error'),
                  'state': 'on' if (LV.env_on() and LV.cfg.get('enabled') and (LV.cfg.get('skus') or (AB.running() and not AB.cfg['dry_run']))) else 'off'}
    ls = AB.last_switch or {}
    st['abtest'] = {'last': ls.get('t'), 'last_error': '; '.join(ls.get('errors') or []) or None,
                    'state': ('dry run' if AB.cfg['dry_run'] else 'live') if AB.running() else ('scheduled' if AB.cfg.get('enabled') else 'off')}
    out = []
    for k, (name, when, what) in STREAMS.items():
        x = st.get(k) or {}
        out.append({'id': k, 'name': name, 'when': when, 'what': what, 'state': x.get('state'),
                    'last': x.get('last') or x.get('last_saved') or x.get('last_sweep'), 'last_error': x.get('last_error')})
    return jsonify({'service': 'offer-recorder', 'started': STATE['started'], 'what': __doc__.split('\n')[0], 'components': out})


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
    threading.Thread(target=retrain_loop, daemon=True).start()
    threading.Thread(target=AB.loop, daemon=True).start()
    if HI:
        threading.Thread(target=delayed, args=(120, _safe, HI.backfill), daemon=True).start()
        threading.Thread(target=delayed, args=(3600, every, 'history', 6 * 3600, HI.prune, []), daemon=True).start()
    threading.Thread(target=delayed, args=(1200, every, 'rivals', RIVALS_EVERY, refresh_rivals, []), daemon=True).start()
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
    threading.Thread(target=delayed, args=(120, every, 'catalog', CATALOG_EVERY, snapshot_catalog, sp), daemon=True).start()
    threading.Thread(target=delayed, args=(300, every, 'traffic', TRAFFIC_EVERY, pull_traffic, sp), daemon=True).start()
    threading.Thread(target=delayed, args=(900, every, 'keepa', KEEPA_EVERY, pull_keepa, ['KEEPA_KEY']), daemon=True).start()
    _safe(FE.load)
    STATE['fees']['skus'] = len(FE.by_sku)
    threading.Thread(target=delayed, args=(240, every, 'fees', FEES_EVERY, pull_fees, sp), daemon=True).start()  # after the first inventory and Seller Snap snapshots
