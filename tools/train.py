"""Train the shadow repricer's models from the recorder's last N days and (optionally) upload model.json.
Usage: RECORDER_KEY=... python3 tools/train.py [--days 7] [--upload] [--out model.json]
       python3 tools/train.py --local /var/data --current /var/data/model/model.json --report retrain.json   (on the recorder, nightly)
Safety check: the new Buy Box model is scored next to the current one on the same held-out 2 days. It replaces the current model only if
its AUC is no more than 0.005 lower and its Brier score no more than 0.005 higher (the current model has usually seen part of those days,
so it starts with a small edge), with at least 1,000 test rows and an average prediction within 5 points of what happened.
Models:
  bb_foep / bb_nofoep  logistic: does Nick hold the Buy Box in the 30 min after a moment (fractional label), from his price vs the
                       lowest rival, vs Amazon's target (bb_foep only), number of rivals, stock share. Validated on the last 2 days.
  rivals               per rival: share of undercuts (by anyone) it matched or beat within 60 min, Beta-smoothed toward the pooled rate;
                       median lag; lowest price seen.
  war_table            median % drop of the listing low 24 h after an undercut, by the reactive rate of the rivals undercut.
"""
import argparse, bisect, collections, datetime as dt, gzip, io, json, math, os, sqlite3, sys, tempfile
import numpy as np, pandas as pd, requests
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import shadow as S
import ctxfeat as CX
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss

ME = S.ME; CAP = 1800
U = os.environ.get('RECORDER_URL', 'https://offer-recorder.onrender.com'); H = {'X-Recorder-Key': os.environ.get('RECORDER_KEY', '')}
ap = argparse.ArgumentParser(); ap.add_argument('--days', type=int, default=7); ap.add_argument('--upload', action='store_true'); ap.add_argument('--out', default='model.json'); ap.add_argument('--cache', default=None)
ap.add_argument('--holder', action='store_true', help='also fit the holder models (bb_foep_h / bb_nofoep_h) for moments a rival holds the Buy Box')
ap.add_argument('--changes', action='store_true', help='also build rows at every change of Nick\'s price (both repricers) and save ch_rows.pkl')
ap.add_argument('--v2', action='store_true', help='Nov 13 setup: held label (share until Nick\'s next price change, at most 6 h), boosted trees '
                'on context inputs (ctxfeat.py) and the post-change model; each kept only if it beats the simpler model on the held-out days')
ap.add_argument('--save-rows', action='store_true', help='test: save the Buy Box rows (bb_rows.pkl) next to --out')
ap.add_argument('--local', default=None, help='read the recorder data folder directly instead of over HTTP')
ap.add_argument('--current', default=None, help='current model.json to compare against (default: the recorder\'s, over HTTP)')
ap.add_argument('--report', default=None, help='write the comparison and the keep/replace verdict here')
ap.add_argument('--rivals-only', default=None, help='hourly: refresh only the rival tables (rates, per listing, war table) into this JSON file;'
                ' with --local, keeps its message index between runs and reads only new or changed files')
A = ap.parse_args()
if A.local:   # on the recorder: scratch space on its data disk (the container's own disk is small), outside the folders it records into
    os.makedirs(os.path.join(A.local, 'tmp'), exist_ok=True)
TMP = tempfile.mkdtemp(prefix='train_', dir=os.path.join(A.local, 'tmp') if A.local else None)
ts_ = lambda s: dt.datetime.fromisoformat(s.replace('Z', '+00:00')).timestamp()
cut = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=A.days)).strftime('%Y-%m-%d')
if A.local:
    files = [{'path': os.path.relpath(os.path.join(r, n), A.local)} for r, _, ns in os.walk(A.local) for n in ns if n.endswith('.jsonl.gz')]
else:
    files = requests.get(U + '/files', headers=H, timeout=60).json()
def _stream(fp):
    """Lines of a gzip file one at a time; the hour still being written can end mid-member, so keep what was readable."""
    try:
        with gzip.open(fp, 'rt') as f:
            for line in f:
                yield line.rstrip('\n')
    except (OSError, EOFError) as e:
        print('partly read', fp, e, flush=True)
def get(path):
    """Lines of one recorder file, streamed (never the whole file in memory)."""
    if A.local:
        yield from _stream(os.path.join(A.local, path)); return
    cdir = A.cache or TMP
    cp = os.path.join(cdir, path.replace('/', '_'))
    today = dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%d')
    if not (A.cache and os.path.exists(cp) and today not in path):  # files from earlier days are complete
        err = None
        for k in range(4):
            try:
                with requests.get(U + '/files/' + path, headers=H, timeout=300, stream=True) as r:
                    r.raise_for_status()
                    with open(cp, 'wb') as f:
                        for ch in r.iter_content(1 << 20):
                            f.write(ch)
                err = None; break
            except Exception as e:
                err = e
        if err:
            print('skipped unreadable file', path, err, flush=True); return
    yield from _stream(cp)
    if not A.cache:
        os.remove(cp)
def want(kind):
    return [f['path'] for f in files if f['path'].startswith(kind + '/') and f['path'].split('/')[1][:10] >= cut]
# ---------------- offers: streamed into a temporary on-disk index, then worked through one ASIN at a time, so memory stays small
# (holding every message in memory took 1.5 GB for 7 days; the recorder server has 512 MB)
KEEP = bool(A.rivals_only and A.local)   # hourly rival refresh on the recorder: one index kept on the data disk, topped up each hour
DB = sqlite3.connect(os.path.join(A.local, 'tmp', 'rivals.db') if KEEP else os.path.join(TMP, 'msgs.db'))
DB.execute('PRAGMA journal_mode=OFF'); DB.execute('PRAGMA synchronous=OFF')
DB.execute('CREATE TABLE IF NOT EXISTS m (id TEXT PRIMARY KEY, asin TEXT, t REAL, off TEXT, supp INTEGER) WITHOUT ROWID')
DB.execute('CREATE TABLE IF NOT EXISTS files (path TEXT PRIMARY KEY, mtime REAL)')
V2 = bool(A.v2 and not KEEP)
if V2:   # context inputs of every message (ctxfeat.MSG_COLS + the lowest rival), for the trees
    DB.execute('CREATE TABLE IF NOT EXISTS cx (id TEXT PRIMARY KEY, asin TEXT, t REAL, v BLOB) WITHOUT ROWID')
seen_files = dict(DB.execute('SELECT path, mtime FROM files').fetchall()) if KEEP else {}
n_msgs = 0
for pth in want('offers'):
    if KEEP:
        mt = os.path.getmtime(os.path.join(A.local, pth))
        if seen_files.get(pth) == mt:
            continue
        DB.execute('INSERT OR REPLACE INTO files VALUES (?, ?)', (pth, mt))
    batch = []; cxb = []
    for l in get(pth):
        d = json.loads(l)
        p = json.loads(d['body'])['Payload']['AnyOfferChangedNotification']
        t = ts_(p['OfferChangeTrigger']['TimeOfOfferChange'])
        # FBA new offers available now, plus Nick's own whatever its availability (same rule as shadow.fba_offers)
        # 4th field: 1 = competes (FBA available, or Nick's own); 0 = kept only because it is flagged Buy Box winner (FBM or a backordered
        # FBA offer), so the number of flagged winners is known (Nov 13: 43% of messages flag 2+ winners)
        off = [(o['SellerId'], o['ListingPrice']['Amount'] + o['Shipping']['Amount'], bool(o.get('IsBuyBoxWinner')),
                1 if (o.get('IsFulfilledByAmazon') and (o.get('SellerId') == ME or not S.backordered(o))) else 0) for o in p['Offers']
               if o.get('SubCondition') == 'new' and ((o.get('IsFulfilledByAmazon') and (o.get('SellerId') == ME or not S.backordered(o))) or o.get('IsBuyBoxWinner'))]
        me_bo = any(o.get('SellerId') == ME and S.backordered(o) for o in p['Offers'])
        batch.append((d['message_id'], p['OfferChangeTrigger']['ASIN'], t, json.dumps(off, separators=(',', ':')), int(bool(S.suppression(p)[0] or me_bo))))
        if V2:
            cx_ = CX.message_ctx(p)
            if cx_ is not None:
                cxb.append((d['message_id'], p['OfferChangeTrigger']['ASIN'], t, np.array(cx_[0] + [cx_[1]], dtype=np.float32).tobytes()))
        if len(batch) >= 5000:
            DB.executemany('INSERT OR IGNORE INTO m VALUES (?,?,?,?,?)', batch); batch = []
            if V2: DB.executemany('INSERT OR IGNORE INTO cx VALUES (?,?,?,?)', cxb); cxb = []
    DB.executemany('INSERT OR IGNORE INTO m VALUES (?,?,?,?,?)', batch)
    if V2: DB.executemany('INSERT OR IGNORE INTO cx VALUES (?,?,?,?)', cxb)
if KEEP:   # drop what fell out of the window
    old = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=A.days)).timestamp()
    DB.execute('DELETE FROM m WHERE t < ?', (old,))
    DB.execute("DELETE FROM files WHERE substr(path, 8, 10) < ?", (cut,))
DB.commit()
DB.execute('CREATE INDEX IF NOT EXISTS ma ON m (asin, t)')
if V2: DB.execute('CREATE INDEX IF NOT EXISTS cxa ON cx (asin, t)')
n_msgs, T1 = DB.execute('SELECT COUNT(*), MAX(t) FROM m').fetchone()
ASINS = [r[0] for r in DB.execute('SELECT DISTINCT asin FROM m')]
print('messages', n_msgs, 'asins', len(ASINS), flush=True)
def load_asin(a):
    """(times, {seller: price} per message, {Buy Box winners} per message, suppressed/backordered flag per message) for one ASIN."""
    ts, P, W, SP = [], [], [], []
    for t, off, sp in DB.execute('SELECT t, off, supp FROM m WHERE asin = ? ORDER BY t', (a,)):
        o = [tuple(z) + (1,) * (4 - len(z)) for z in json.loads(off)]   # rows written before Nov 13 have 3 fields (all competing)
        ts.append(t); P.append({x: q for x, q, w, c in o if c}); W.append({x for x, q, w, c in o if w}); SP.append(bool(sp))
    return np.array(ts), P, W, SP
# ---------------- small inputs first: Amazon's target price readings, FBA inventory and Keepa rival stock
fo = collections.defaultdict(list)
for pth in ([] if A.rivals_only else want('foep')):
    for l in get(pth):
        x = json.loads(l); b = x.get('body') or {}; r = (b.get('featuredOfferExpectedPriceResults') or [{}])[0]
        a = (b.get('offerIdentifier') or {}).get('asin')
        if r.get('resultStatus') == 'VALID_FOEP' and a:
            fo[a].append((ts_(x['taken_at']), x['sku'], a, r['featuredOfferExpectedPrice']['listingPrice']['amount']))
FO_ALL = {a_: sorted(v) for a_, v in fo.items()}   # every reading, for the price-change rows
# one training moment per SKU per 25 min at most: readings got more frequent on Nov 13 (10-min sweeps + changed listings),
# and the Buy Box label covers 30 min, so closer readings would mostly repeat the same moment
for a_ in fo:
    fo[a_].sort(); kept, last = [], {}
    for x in fo[a_]:
        if x[0] - last.get(x[1], -1e18) >= 1500:
            kept.append(x); last[x[1]] = x[0]
    fo[a_] = kept
SEG = {}   # asin -> ([times], [Nick's page-view-weighted featured share])
_sg = collections.defaultdict(list)
for pth in ([] if A.rivals_only else want('segments')):
    for l in get(pth):
        try:
            x = json.loads(l)
        except ValueError:
            continue
        b = x.get('body') or {}; a_ = x.get('asin') or b.get('asin')
        if not a_ or x.get('status') not in (200, None): continue
        tot = mine_ = 0.0
        for o in b.get('featuredBuyingOptions') or []:
            if (o.get('buyingOptionType') or 'New') != 'New': continue
            for so in o.get('segmentedFeaturedOffers') or []:
                for sgm in so.get('featuredOfferSegments') or []:
                    g = (sgm.get('segmentDetails') or {}).get('glanceViewWeightPercentage') or 0.0
                    tot += g; mine_ += g if so.get('sellerId') == ME else 0.0
        _sg[a_].append((ts_(x['taken_at']), mine_ / tot if tot > 0 else 0.0))
for a_, v in _sg.items():
    v.sort(); SEG[a_] = ([q[0] for q in v], [q[1] for q in v])
del _sg
print('segment snapshots', sum(len(v[0]) for v in SEG.values()), 'asins', len(SEG), flush=True)
inv = collections.defaultdict(list)
for pth in ([] if A.rivals_only else want('inventory')):
    for l in get(pth):
        x = json.loads(l); t = ts_(x['taken_at'])
        for r in x['rows']: inv[r['sellerSku']].append((t, (r.get('inventoryDetails') or {}).get('fulfillableQuantity') or 0))
kp = collections.defaultdict(list)
sh = S.Shadow.__new__(S.Shadow); sh.lock = __import__('threading').Lock(); sh.rival_stock = {}
sh.fba_high = {}; sh.demand = {}; sh.save_state = lambda *a, **k: None; sh.rebuild = lambda: None
for pth in ([] if A.rivals_only else want('keepa')):
    for l in get(pth):
        x = json.loads(l); t = ts_(x['taken_at']); sh.rival_stock = {}
        sh.on_keepa_products(x['response'].get('products'))
        for a, (_, v) in sh.rival_stock.items(): kp[a].append((t, v))
for k in inv: inv[k].sort()
for k in kp: kp[k].sort()
def lookup(series, t, max_age):
    if not series: return None
    i = bisect.bisect_right([s[0] for s in series], t) - 1
    return series[i][1] if i >= 0 and t - series[i][0] <= max_age else None
# The label: Nick's share of the Buy Box over the 30 min after a moment (Nov 13 fix).
#  1. Amazon's segment snapshots (competitive summary, about every 20 min per ASIN): Nick's page-view-weighted share of the featured
#     offer across segments (Prime / non-Prime / default x sample ZIP). The mean of the snapshots in the window, when there is one.
#  2. Otherwise the offer messages, each credited by how many offers it flags as winner: Amazon often flags 2-3 at once, and only one
#     of them holds the featured offer. Credits measured Nov 12-13 against the snapshots (151,418 matched):
CREDIT = {1: 0.96, 2: 0.43, 3: 0.17}
def credit(w):
    return CREDIT.get(len(w), 0.05) if ME in w else 0.0
def msg_share(X, t, dur=1800):
    ts, P, W = X; i = max(0, bisect.bisect_right(ts, t) - 1); tot = win = 0
    while i < len(ts) and ts[i] < t + dur:
        nxt = ts[i + 1] if i + 1 < len(ts) else t + dur; seg = max(0, min(nxt, t + dur, ts[i] + CAP) - max(ts[i], t))
        if ME in P[i]: tot += seg; win += seg * credit(W[i])
        i += 1
    return win / tot if tot > 0 else None
LABEL_SRC = collections.Counter()
def bb_next(X, t, dur=1800, a=None):
    sg = SEG.get(a) if a else None
    if sg:
        j0, j1 = bisect.bisect_left(sg[0], t), bisect.bisect_right(sg[0], t + dur)
        if j1 > j0:
            LABEL_SRC['snapshots'] += 1
            return float(np.mean(sg[1][j0:j1]))
    LABEL_SRC['messages'] += 1
    return msg_share(X, t, dur)
# ---------------- per ASIN: rival reactions to undercuts (by anyone), price-war events, and Buy Box training rows
react = collections.defaultdict(lambda: [0, 0]); lags = collections.defaultdict(list); events = []; R = []; CH = []
STRANDED = (ts_('2026-11-13T16:40:00Z'), ts_('2026-11-13T18:10:00Z'))   # Nick's listings stranded by an account issue: left out
react_l = collections.defaultdict(lambda: [0, 0]); lags_l = collections.defaultdict(list)   # per (rival, listing)
HELD_MAX = 6 * 3600
def held_label(a, t, oc):
    """Nick's page-view share from t until his next price change (oc: his change times on the listing), at most 6 h; None when the
    window has no segment snapshot, runs past the data, or touches the stranded window. Tested Nov 13 (BUYBOX_ROTATION_AND_LABEL_WINDOW):
    as accurate as 30 min on the share while a price stands, without diluting the price effect as a fixed 6-h window does."""
    k = bisect.bisect_right(oc, t + 1)
    end = min(t + HELD_MAX, oc[k] if k < len(oc) else float('inf'))
    if end > T1 or (t <= STRANDED[1] and end >= STRANDED[0]):
        return None
    sg = SEG.get(a)
    if not sg:
        return None
    j0, j1 = bisect.bisect_left(sg[0], t), bisect.bisect_right(sg[0], end)
    return float(np.mean(sg[1][j0:j1])) if j1 > j0 else None
NANV = [float('nan')] * (len(CX.MSG_COLS) + len(CX.ROLL_COLS))
for a in ASINS:
    ts, P, W, SP = load_asin(a)
    if V2:
        cxr = DB.execute('SELECT t, v FROM cx WHERE asin = ? ORDER BY t', (a,)).fetchall()
        ctt = np.array([q[0] for q in cxr]); cv = np.array([np.frombuffer(q[1], dtype=np.float32) for q in cxr]) if cxr else np.zeros((0, 21))
        cM, cL = (cv[:, :-1], cv[:, -1]) if len(cv) else (cv, cv)
        oc = [ts[i] for i in range(1, len(ts)) if ME in P[i] and ME in P[i - 1] and abs(P[i][ME] - P[i - 1][ME]) >= 0.01]
        def ctx_of(t, n_riv, hk, strict):
            i = int(np.searchsorted(ctt, t, side='left' if strict else 'right')) - 1
            v = CX.vector_at(ctt, cM, cL, i, t, n_riv, hk) if len(ctt) else None
            return v if v is not None else NANV + CX.hold_cols(hk)
    for i in range(1, len(ts)):
        P0, P1 = P[i - 1], P[i]
        if len(P1) < 2: continue
        for s_, p in P1.items():
            if s_ not in P0 or p >= P0[s_] - 0.005: continue
            oth = [q for x, q in P1.items() if x != s_]
            if p > min(oth) - 0.01: continue
            t = ts[i]; L0 = min(P0.values()); at_low = [x for x, q in P0.items() if x != s_ and q <= L0 * 1.005]
            if t + 3600 > T1 or not at_low: continue
            resp = {}
            j = i + 1
            while j < len(ts) and ts[j] <= t + 3600:
                for x in at_low:
                    if x not in resp and x in P[j] and abs(P[j][x] - P0[x]) >= 0.005:
                        resp[x] = (P[j][x] <= p + 0.005, ts[j] - t)
                j += 1
            for x in at_low:
                ok = resp.get(x, (False, None))
                react[x][0] += int(ok[0]); react[x][1] += 1
                if ok[0]: lags[x].append(ok[1] / 60.0)
                react_l[(x, a)][0] += int(ok[0]); react_l[(x, a)][1] += 1
                if ok[0]: lags_l[(x, a)].append(ok[1] / 60.0)
            low24 = None
            if t + 86400 <= T1:
                k = bisect.bisect_right(ts, t + 86400) - 1
                low24 = 100.0 * (min(P[k].values()) / L0 - 1.0) if P[k] else None
            events.append((at_low, low24))
    if A.changes or V2:   # every change of Nick's own price (his repricer or Seller Snap): what happened in the 30 min after Amazon showed it
        fa = FO_ALL.get(a) or []; fts = [q[0] for q in fa]
        for i in range(1, len(ts)):
            if ME not in P[i] or ME not in P[i - 1] or SP[i] or len(P[i]) < 2: continue
            p0, p1 = P[i - 1][ME], P[i][ME]
            if abs(p1 - p0) < 0.01: continue
            t = ts[i]
            if t + 1800 > T1 or STRANDED[0] - 1800 <= t <= STRANDED[1]: continue
            y = bb_next((ts, P, W), t, a=a)
            if y is None: continue
            k = bisect.bisect_right(fts, t) - 1
            f, sku = (fa[k][3], fa[k][1]) if k >= 0 and t - fa[k][0] <= 45 * 60 else (None, fa[k][1] if k >= 0 else None)
            riv_p = {x: q for x, q in P[i].items() if x != ME}; low = min(riv_p.values())
            w0 = W[i - 1]; wf0 = [x for x in w0 if x in P[i - 1] and x != ME]
            if ME in w0: hk, hp = 'me', None
            elif not wf0: hk, hp = 'other', None
            else:
                hp = min(P[i - 1][x] for x in wf0); hk = 'above' if hp > low + 0.005 else 'low'
            mine = lookup(inv.get(sku), t, 2 * 3600) if sku else None; riv = lookup(kp.get(a), t, 8 * 3600)
            share = mine / (mine + riv) if (mine is not None and riv is not None and mine + riv > 0) else None
            row = dict(t=t, a=a, y=y, me=p1, frm=p0, low=low, n=len(P[i]) - 1, foep=f, share=share, hk=hk, hp=hp)
            if V2:
                row['y'] = held_label(a, t, oc)
                if row['y'] is None: continue
                row['cx'] = ctx_of(t, len(P[i]) - 1, hk, True)   # the message before the change: its flags don't show the result
            CH.append(row)
    for t, sku, _, f in fo.get(a, ()):
        if t + 1800 > T1: continue
        i = bisect.bisect_right(ts, t) - 1
        if i < 0 or t - ts[i] > 6 * 3600: continue
        Pi = P[i]
        if ME not in Pi or len(Pi) < 2: continue
        if SP[i]: continue   # no Buy Box to win: the repricer matches the lowest offer instead (shadow.decide_suppressed)
        y = bb_next((ts, P, W), t, a=a)
        if y is None: continue
        me = Pi[ME]; low = min(q for x, q in Pi.items() if x != ME)
        mine = lookup(inv.get(sku), t, 2 * 3600); riv = lookup(kp.get(a), t, 8 * 3600)
        share = mine / (mine + riv) if (mine is not None and riv is not None and mine + riv > 0) else None
        w = W[i]; wf_ = [x for x in w if x in Pi and x != ME]
        if ME in w: hk, hp = 'me', None
        elif not wf_: hk, hp = 'other', None   # the winner isn't an available FBA offer: FBM or backordered
        else:
            hp = min(Pi[x] for x in wf_); hk = 'above' if hp > low + 0.005 else 'low'
        row = dict(t=t, a=a, y=y, me=me, low=low, n=len(Pi) - 1, foep=f, share=share, hk=hk, hp=hp)
        if V2:
            row['y'] = held_label(a, t, oc)
            if row['y'] is None: continue
            row['cx'] = ctx_of(t, len(Pi) - 1, hk, False)
        R.append(row)
DB.close()
n = np.array([v[1] for v in react.values()]); k_ = np.array([v[0] for v in react.values()])
m5 = n >= 5; pr = (k_[m5] / n[m5]) if m5.any() else np.array([0.3])
mu = float(np.clip(pr.mean(), 0.05, 0.95)); var = float(max(pr.var(), 1e-4))
strength = float(np.clip(mu * (1 - mu) / var - 1, 2, 50))  # method-of-moments Beta prior
a0, b0 = mu * strength, (1 - mu) * strength
allags = [x for v in lags.values() for x in v]
rival_default = {'rate': round(mu, 4), 'lag_min': round(float(np.median(allags)) if allags else 10.0, 2), 'prior_a': round(a0, 3), 'prior_b': round(b0, 3)}
rivals = {}
for x, (kk, nn) in react.items():
    rivals[x] = {'rate': round((kk + a0) / (nn + a0 + b0), 4), 'n': nn, 'k': kk, 'lag_min': round(float(np.median(lags[x])), 2) if len(lags[x]) >= 3 else None}
# Per listing too: rivals fight hard on some listings and ignore others (Nov 12 check: per-listing rates, pulled toward the seller's,
# predicted next answers ~10% better than one rate per seller). shadow.rival_info blends them with LISTING_WEIGHT.
rival_listing = {f'{x}|{a}': {'k': kk, 'n': nn, 'lag_min': round(float(np.median(lags_l[(x, a)])), 2) if lags_l[(x, a)] else None}
                 for (x, a), (kk, nn) in react_l.items()}
print('rivals', len(rivals), 'pooled rate', rival_default, 'rival-listing pairs', len(rival_listing), flush=True)
# ---------------- price-war table
rows = []
for at_low, low24 in events:
    if low24 is None: continue
    r = max(rivals[x]['rate'] for x in at_low); rows.append((r, -low24))
W_ = pd.DataFrame(rows, columns=['r', 'drop'])
bins = [0, .1, .3, .5, .7, 1.0001]; W_['bin'] = pd.cut(W_.r, bins, right=False)
war = []
for b, g in W_.groupby('bin', observed=True):
    if len(g) >= 30: war.append([round(float(g.r.median()), 4), round(float(max(g['drop'].median(), 0.0)), 4)])
if len(war) < 2: war = [[0.05, 0.4], [0.85, 3.9]]
print('war table', war, flush=True)
if A.rivals_only:
    out = {'updated_at': dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds'), 'window_days': A.days, 'messages': n_msgs,
           'events': int(sum(v[1] for v in react.values())), 'rivals': rivals, 'rival_default': rival_default, 'rival_listing': rival_listing,
           'listing_weight': 10, 'war_table': war}
    tmpf = A.rivals_only + '.tmp'
    json.dump(out, open(tmpf, 'w')); os.replace(tmpf, A.rivals_only)
    print('wrote rivals', A.rivals_only, out['events'], 'events', flush=True)
    __import__('shutil').rmtree(TMP, ignore_errors=True)
    sys.exit(0)
D = pd.DataFrame(R); print('Buy Box rows', len(D), 'asins', D.a.nunique(), 'with stock share', int(D.share.notna().sum()), 'label from', dict(LABEL_SRC), flush=True)
if A.save_rows:
    D.to_pickle(os.path.join(os.path.dirname(os.path.abspath(A.out)), 'bb_rows.pkl'))
if A.changes:
    pd.DataFrame(CH).to_pickle(os.path.join(os.path.dirname(os.path.abspath(A.out)), 'ch_rows.pkl')); print('price-change rows', len(CH), flush=True)
HOLD = bool(A.holder)
def fx(r, wf, holder=False):
    x = S.feats(r.me, r.low, r.n, r.foep if wf else None, None if pd.isna(r.share) else r.share,
                {'kind': r.hk, 'price': None if pd.isna(r.hp) else r.hp} if holder else None)
    if holder:
        x.pop('h_me', None)   # the holder models only see moments a rival (or an FBM/backordered offer) holds the Buy Box
    return x
def X_of(df, with_foep, holder=False):
    out = [fx(r, with_foep, holder) for r in df.itertuples()]
    keys = sorted(out[0].keys())
    return np.array([[x[k] for k in keys] for x in out]), keys
def fit(df, with_foep, holder=False):
    X, keys = X_of(df, with_foep, holder); y = df.y.values
    XX = np.vstack([X, X]); yy = np.r_[np.ones(len(y)), np.zeros(len(y))]; ww = np.r_[y, 1 - y] + 1e-6
    m = LogisticRegression(C=1.0, max_iter=3000, fit_intercept=False).fit(XX, yy, sample_weight=ww)
    return dict(zip(keys, [float(c) for c in m.coef_[0]]))
def metrics(coefs, name, df, wf, holder):
    pred = np.array([S.p_win({name: {'coef': coefs}}, fx(r, wf, holder)) for r in df.itertuples()])
    yb = (df.y >= 0.5).astype(int)
    return {'test_rows': int(len(df)), 'auc': round(float(roc_auc_score(yb, pred)), 4), 'brier': round(float(brier_score_loss(yb, pred)), 4),
            'mean_pred': round(float(pred.mean()), 4), 'mean_actual': round(float(df.y.mean()), 4)}
# Holder models (Nov 13, Nick): the plain models over-predicted Nick's Buy Box share when a rival held it (18.7% predicted vs 7.6%
# actual on held-out days, with a target), so undercuts were credited with ~2.5x their real gain. When someone else holds the Buy Box,
# shadow.decide uses bb_foep_h / bb_nofoep_h, fitted only on those moments; when Nick holds it, the plain models (raises scored as before).
NOT_ME = lambda df: df[df.hk != 'me']
cutoff = T1 - 2 * 86400; tr, te = D[D.t < cutoff], D[D.t >= cutoff]
val = {}
for name, wf in (('bb_foep', True), ('bb_nofoep', False)):
    val[name] = metrics(fit(tr, wf), name, te, wf, False)
    if HOLD:
        nm = name + '_h'
        val[nm] = metrics(fit(NOT_ME(tr), wf, True), nm, NOT_ME(te), wf, True)
        val[nm]['plain_on_same_rows'] = {k: v for k, v in metrics(fit(tr, wf), name, NOT_ME(te), wf, False).items() if k in ('auc', 'brier', 'mean_pred')}
rule = (te.me <= te.foep + 0.01).astype(float); val['rule_at_or_below_target'] = {'auc': round(float(roc_auc_score((te.y >= 0.5).astype(int), rule)), 4)}
print('validation (last 2 days held out)', json.dumps(val), flush=True)
model = {'trained_at': dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds'), 'window_days': A.days, 'messages': n_msgs,
         'bb_foep': {'coef': fit(D, True)}, 'bb_nofoep': {'coef': fit(D, False)}, 'validation': val,
         'rivals': rivals, 'rival_default': rival_default, 'rival_listing': rival_listing, 'listing_weight': 10, 'war_table': war, 'bb_rows': int(len(D))}
holder_ok = {}
if HOLD:   # kept only if, on the held-out moments a rival held the Buy Box, it ranks as well as the plain model and predicts the level
    for name in ('bb_foep', 'bb_nofoep'):
        v = val[name + '_h']; pv = v['plain_on_same_rows']; why = []
        if v['test_rows'] < 500: why.append(f"only {v['test_rows']} test rows")
        if v['auc'] < pv['auc'] - 0.005: why.append(f"AUC {v['auc']:.4f} vs plain {pv['auc']:.4f}")
        if v['brier'] > pv['brier'] + 0.005: why.append(f"Brier {v['brier']:.4f} vs plain {pv['brier']:.4f}")
        if abs(v['mean_pred'] - v['mean_actual']) > 0.05: why.append(f"average {v['mean_pred']:.3f} vs actual {v['mean_actual']:.3f}")
        holder_ok[name + '_h'] = why or 'ok'
        if not why:
            model[name + '_h'] = {'coef': fit(NOT_ME(D), name == 'bb_foep', True)}
    print('holder models', json.dumps(holder_ok), flush=True)
# ---------------- Nov 13 setup (--v2): boosted trees on context inputs, and the post-change model
# Trees start from the logistic's log-odds and see no price inputs (ctxfeat.FEATURES), so the price response stays the logistic's.
# The post-change model scores prices other than the current one (shadow.decide); fitted on every change of Nick's price.
# Each part is kept only if, on the held-out last 2 days, it has a lower mean squared error on the held share than the simpler
# model it would replace, and its average is within 5 points of what happened. Tested Nov 13: BUYBOX_HEAD_TO_HEAD_2026-11-13.md.
V2_REPORT = {}
try:
    import lightgbm as lgb
except Exception as e:   # no LightGBM: the plain and holder models are still trained; trees and the post-change model are skipped
    lgb = None; V2_REPORT['error'] = f'lightgbm: {type(e).__name__}: {e}'
if V2 and lgb is not None and len(D) > 2000:
    TREE_PARAMS = dict(objective='cross_entropy', learning_rate=0.03, num_leaves=15, min_data_in_leaf=300, feature_fraction=0.8,
                       bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, verbose=-1, seed=7, num_threads=1)
    zlin = lambda c, x: sum(c.get(k, 0.0) * v for k, v in x.items())
    sigm = lambda z: 1 / (1 + np.exp(-np.clip(z, -30, 30)))
    def base_z(rows, wf, cp, ch):   # the live split: plain model when Nick is flagged, holder model otherwise
        return np.array([zlin(cp, fx(r, wf, False)) if (r.hk == 'me' or ch is None) else zlin(ch, fx(r, wf, True)) for r in rows.itertuples()])
    def chfx(r, wf):
        x = S.feats(r.me, r.low, r.n, r.foep if wf else None, None if pd.isna(r.share) else r.share, {'kind': r.hk, 'price': None if pd.isna(r.hp) else r.hp})
        x.update(S.change_terms(r.me, r.frm, x.get('h_me', 0.0)))
        return x
    def fit_ch(rows, wf):
        xs = [chfx(r, wf) for r in rows.itertuples()]; keys = sorted(xs[0]); X = np.array([[x[k] for k in keys] for x in xs]); y = rows.y.values
        m = LogisticRegression(C=1.0, max_iter=3000, fit_intercept=False).fit(np.vstack([X, X]), np.r_[np.ones(len(y)), np.zeros(len(y))], sample_weight=np.r_[y, 1 - y] + 1e-6)
        return dict(zip(keys, [float(c) for c in m.coef_[0]]))
    Xof = lambda rows: np.array(rows.cx.tolist(), dtype=np.float32)
    def tree(rows_fit, z_fit, rows_va=None, z_va=None, rounds=None):
        d = lgb.Dataset(Xof(rows_fit), rows_fit.y.values, init_score=z_fit, feature_name=CX.FEATURES, free_raw_data=True)
        if rounds is None:
            v = lgb.Dataset(Xof(rows_va), rows_va.y.values, init_score=z_va, reference=d)
            return lgb.train(TREE_PARAMS, d, 2000, valid_sets=[v], callbacks=[lgb.early_stopping(100, verbose=False)])
        return lgb.train(TREE_PARAMS, d, max(int(rounds), 1))
    mse = lambda p, y: float(np.mean((p - y) ** 2))
    def judge(name, p_new, p_old, y):
        why = []
        if len(y) < 500: why.append(f'only {len(y)} test rows')
        if mse(p_new, y) >= mse(p_old, y): why.append(f'MSE {mse(p_new, y):.4f} not below {mse(p_old, y):.4f}')
        if abs(float(p_new.mean()) - float(y.mean())) > 0.05: why.append(f'average {p_new.mean():.3f} vs actual {y.mean():.3f}')
        V2_REPORT[name] = {'test_rows': int(len(y)), 'mse_new': round(mse(p_new, y), 4), 'mse_old': round(mse(p_old, y), 4),
                           'mean_new': round(float(p_new.mean()), 4), 'mean_old': round(float(p_old.mean()), 4), 'mean_actual': round(float(y.mean()), 4),
                           'kept': not why, 'why_not': why}
        return not why
    vcut = cutoff - 86400
    CHD = pd.DataFrame(CH)
    for name, wf in (('foep', True), ('nofoep', False)):
        dsel = (lambda df: df[df.foep.notna()]) if wf else (lambda df: df)
        # ordinary moments: tree on top of the plain/holder logistic
        Dw = dsel(D); trw, tew = Dw[Dw.t < cutoff], Dw[Dw.t >= cutoff]; fw, vw = trw[trw.t < vcut], trw[trw.t >= vcut]
        cp_f = fit(fw, wf); ch_f = fit(NOT_ME(fw), wf, True) if ('bb_' + name + '_h') in model else None
        b = tree(fw, base_z(fw, wf, cp_f, ch_f), vw, base_z(vw, wf, cp_f, ch_f))
        z_te = base_z(tew, wf, cp_f, ch_f)
        if judge('tree_' + name, sigm(z_te + b.predict(Xof(tew), raw_score=True)), sigm(z_te), tew.y.values):
            cp_all = model['bb_' + name]['coef']; ch_all = (model.get('bb_' + name + '_h') or {}).get('coef')
            bf = tree(Dw, base_z(Dw, wf, cp_all, ch_all), rounds=b.best_iteration)
            model['tree_' + name] = {'booster': bf.model_to_string(), 'features': CX.FEATURES, 'rounds': int(b.best_iteration)}
        # price changes: post-change logistic (vs the ordinary model at the new price), then its tree
        if len(CHD) < 2000: continue
        Cw = dsel(CHD); ctr, cte = Cw[Cw.t < cutoff], Cw[Cw.t >= cutoff]; cfw, cvw = ctr[ctr.t < vcut], ctr[ctr.t >= vcut]
        cp_t = fit(dsel(tr), wf); ch_t = fit(NOT_ME(dsel(tr)), wf, True) if ('bb_' + name + '_h') in model else None
        cc_t = fit_ch(ctr, wf)
        z_cc = np.array([zlin(cc_t, chfx(r, wf)) for r in cte.itertuples()])
        if judge('bb_change_' + name, sigm(z_cc), sigm(base_z(cte, wf, cp_t, ch_t)), cte.y.values):
            model['bb_change_' + name] = {'coef': fit_ch(Cw, wf)}
            cc_f = fit_ch(cfw, wf)
            zc = lambda rows, c: np.array([zlin(c, chfx(r, wf)) for r in rows.itertuples()])
            bc = tree(cfw, zc(cfw, cc_f), cvw, zc(cvw, cc_f))
            z2 = zc(cte, cc_f)
            if judge('tree_change_' + name, sigm(z2 + bc.predict(Xof(cte), raw_score=True)), sigm(z2), cte.y.values):
                bcf = tree(Cw, zc(Cw, model['bb_change_' + name]['coef']), rounds=bc.best_iteration)
                model['tree_change_' + name] = {'booster': bcf.model_to_string(), 'features': CX.FEATURES, 'rounds': int(bc.best_iteration)}
    model['label'] = 'held: page-view share until the next own price change, at most 6 h'
    model['v2'] = V2_REPORT
    print('v2 models', json.dumps(V2_REPORT), flush=True)
# ---------------- safety check: new vs current model on the same held-out days
def score(coefs, name, wf):
    return {k: v for k, v in metrics(coefs, name, te, wf, False).items() if k in ('auc', 'brier', 'mean_pred')}
cur = None
try:
    if A.current:
        cur = json.load(open(A.current))
    elif not A.local:
        cur = requests.get(U + '/model', headers=H, timeout=60).json()
except (OSError, ValueError, requests.RequestException) as e:
    print('no current model to compare:', e, flush=True)
check = {'test_rows': int(len(te)), 'mean_actual': round(float(te.y.mean()), 4), 'new': {}, 'current': {}, 'reasons': []}
for name, wf in (('bb_foep', True), ('bb_nofoep', False)):
    check['new'][name] = {k: val[name][k] for k in ('auc', 'brier', 'mean_pred')}
    if cur and (cur.get(name) or {}).get('coef'):
        check['current'][name] = score(cur[name]['coef'], name, wf)
n_, c_ = check['new']['bb_foep'], check['current'].get('bb_foep')
if len(te) < 1000:
    check['reasons'].append(f'only {len(te)} test rows (need 1,000)')
if abs(n_['mean_pred'] - check['mean_actual']) > 0.05:
    check['reasons'].append(f"average prediction {n_['mean_pred']:.3f} vs actual {check['mean_actual']:.3f} (more than 5 points off)")
if c_:
    if n_['auc'] < c_['auc'] - 0.005:
        check['reasons'].append(f"AUC {n_['auc']:.4f} vs current {c_['auc']:.4f}")
    if n_['brier'] > c_['brier'] + 0.005:
        check['reasons'].append(f"Brier {n_['brier']:.4f} vs current {c_['brier']:.4f}")
check['verdict'] = 'keep current' if check['reasons'] else 'replace'
check['current_trained_at'] = (cur or {}).get('trained_at')
check['holder_models'] = holder_ok
check['v2'] = {k: {kk: v[kk] for kk in ('kept', 'mse_new', 'mse_old', 'test_rows')} for k, v in V2_REPORT.items()}
model['check'] = check
print('safety check', json.dumps(check), flush=True)
raw = json.dumps(model); open(A.out, 'w').write(raw); print('wrote', A.out, len(raw), 'bytes')
if A.report:
    json.dump({**check, 'trained_at': model['trained_at'], 'messages': n_msgs, 'bb_rows': int(len(D))}, open(A.report, 'w'))
if A.upload and check['verdict'] == 'replace':
    r = requests.post(U + '/model', headers={**H, 'content-type': 'application/json'}, data=raw, timeout=120); print('upload', r.status_code, r.text[:200])
elif A.upload:
    print('not uploaded: the current model stays (' + '; '.join(check['reasons']) + ')')
__import__('shutil').rmtree(TMP, ignore_errors=True)
