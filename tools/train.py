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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss

ME = S.ME; CAP = 1800
U = os.environ.get('RECORDER_URL', 'https://offer-recorder.onrender.com'); H = {'X-Recorder-Key': os.environ.get('RECORDER_KEY', '')}
ap = argparse.ArgumentParser(); ap.add_argument('--days', type=int, default=7); ap.add_argument('--upload', action='store_true'); ap.add_argument('--out', default='model.json'); ap.add_argument('--cache', default=None)
ap.add_argument('--holder', action='store_true', help='also fit the holder models (bb_foep_h / bb_nofoep_h) for moments a rival holds the Buy Box')
ap.add_argument('--changes', action='store_true', help='also build rows at every change of Nick\'s price (both repricers) and save ch_rows.pkl')
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
seen_files = dict(DB.execute('SELECT path, mtime FROM files').fetchall()) if KEEP else {}
n_msgs = 0
for pth in want('offers'):
    if KEEP:
        mt = os.path.getmtime(os.path.join(A.local, pth))
        if seen_files.get(pth) == mt:
            continue
        DB.execute('INSERT OR REPLACE INTO files VALUES (?, ?)', (pth, mt))
    batch = []
    for l in get(pth):
        d = json.loads(l)
        p = json.loads(d['body'])['Payload']['AnyOfferChangedNotification']
        t = ts_(p['OfferChangeTrigger']['TimeOfOfferChange'])
        # FBA new offers available now, plus Nick's own whatever its availability (same rule as shadow.fba_offers)
        off = [(o['SellerId'], o['ListingPrice']['Amount'] + o['Shipping']['Amount'], bool(o.get('IsBuyBoxWinner'))) for o in p['Offers']
               if o.get('SubCondition') == 'new' and o.get('IsFulfilledByAmazon') and (o.get('SellerId') == ME or not S.backordered(o))]
        me_bo = any(o.get('SellerId') == ME and S.backordered(o) for o in p['Offers'])
        batch.append((d['message_id'], p['OfferChangeTrigger']['ASIN'], t, json.dumps(off, separators=(',', ':')), int(bool(S.suppression(p)[0] or me_bo))))
        if len(batch) >= 5000:
            DB.executemany('INSERT OR IGNORE INTO m VALUES (?,?,?,?,?)', batch); batch = []
    DB.executemany('INSERT OR IGNORE INTO m VALUES (?,?,?,?,?)', batch)
if KEEP:   # drop what fell out of the window
    old = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=A.days)).timestamp()
    DB.execute('DELETE FROM m WHERE t < ?', (old,))
    DB.execute("DELETE FROM files WHERE substr(path, 8, 10) < ?", (cut,))
DB.commit()
DB.execute('CREATE INDEX IF NOT EXISTS ma ON m (asin, t)')
n_msgs, T1 = DB.execute('SELECT COUNT(*), MAX(t) FROM m').fetchone()
ASINS = [r[0] for r in DB.execute('SELECT DISTINCT asin FROM m')]
print('messages', n_msgs, 'asins', len(ASINS), flush=True)
def load_asin(a):
    """(times, {seller: price} per message, {Buy Box winners} per message, suppressed/backordered flag per message) for one ASIN."""
    ts, P, W, SP = [], [], [], []
    for t, off, sp in DB.execute('SELECT t, off, supp FROM m WHERE asin = ? ORDER BY t', (a,)):
        o = json.loads(off); ts.append(t); P.append({x: q for x, q, w in o}); W.append({x for x, q, w in o if w}); SP.append(bool(sp))
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
def bb_next(X, t, dur=1800):
    ts, P, W = X; i = max(0, bisect.bisect_right(ts, t) - 1); tot = win = 0
    while i < len(ts) and ts[i] < t + dur:
        nxt = ts[i + 1] if i + 1 < len(ts) else t + dur; seg = max(0, min(nxt, t + dur, ts[i] + CAP) - max(ts[i], t))
        if ME in P[i]: tot += seg; win += seg * (ME in W[i])
        i += 1
    return win / tot if tot > 0 else None
# ---------------- per ASIN: rival reactions to undercuts (by anyone), price-war events, and Buy Box training rows
react = collections.defaultdict(lambda: [0, 0]); lags = collections.defaultdict(list); events = []; R = []; CH = []
STRANDED = (ts_('2026-11-13T16:40:00Z'), ts_('2026-11-13T18:10:00Z'))   # Nick's listings stranded by an account issue: left out
react_l = collections.defaultdict(lambda: [0, 0]); lags_l = collections.defaultdict(list)   # per (rival, listing)
for a in ASINS:
    ts, P, W, SP = load_asin(a)
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
    if A.changes:   # every change of Nick's own price (his repricer or Seller Snap): what happened in the 30 min after Amazon showed it
        fa = FO_ALL.get(a) or []; fts = [q[0] for q in fa]
        for i in range(1, len(ts)):
            if ME not in P[i] or ME not in P[i - 1] or SP[i] or len(P[i]) < 2: continue
            p0, p1 = P[i - 1][ME], P[i][ME]
            if abs(p1 - p0) < 0.01: continue
            t = ts[i]
            if t + 1800 > T1 or STRANDED[0] - 1800 <= t <= STRANDED[1]: continue
            y = bb_next((ts, P, W), t)
            if y is None: continue
            k = bisect.bisect_right(fts, t) - 1
            f, sku = (fa[k][3], fa[k][1]) if k >= 0 and t - fa[k][0] <= 45 * 60 else (None, fa[k][1] if k >= 0 else None)
            riv_p = {x: q for x, q in P[i].items() if x != ME}; low = min(riv_p.values())
            w0 = W[i - 1]
            if ME in w0: hk, hp = 'me', None
            elif not w0: hk, hp = 'other', None
            else:
                hp = min(P[i - 1][x] for x in w0 if x in P[i - 1]); hk = 'above' if hp > low + 0.005 else 'low'
            mine = lookup(inv.get(sku), t, 2 * 3600) if sku else None; riv = lookup(kp.get(a), t, 8 * 3600)
            share = mine / (mine + riv) if (mine is not None and riv is not None and mine + riv > 0) else None
            CH.append(dict(t=t, a=a, y=y, me=p1, frm=p0, low=low, n=len(P[i]) - 1, foep=f, share=share, hk=hk, hp=hp))
    for t, sku, _, f in fo.get(a, ()):
        if t + 1800 > T1: continue
        i = bisect.bisect_right(ts, t) - 1
        if i < 0 or t - ts[i] > 6 * 3600: continue
        Pi = P[i]
        if ME not in Pi or len(Pi) < 2: continue
        if SP[i]: continue   # no Buy Box to win: the repricer matches the lowest offer instead (shadow.decide_suppressed)
        y = bb_next((ts, P, W), t)
        if y is None: continue
        me = Pi[ME]; low = min(q for x, q in Pi.items() if x != ME)
        mine = lookup(inv.get(sku), t, 2 * 3600); riv = lookup(kp.get(a), t, 8 * 3600)
        share = mine / (mine + riv) if (mine is not None and riv is not None and mine + riv > 0) else None
        w = W[i]
        if ME in w: hk, hp = 'me', None
        elif not w: hk, hp = 'other', None   # the winner isn't an available FBA offer: FBM or backordered
        else:
            hp = min(Pi[x] for x in w); hk = 'above' if hp > low + 0.005 else 'low'
        R.append(dict(t=t, a=a, y=y, me=me, low=low, n=len(Pi) - 1, foep=f, share=share, hk=hk, hp=hp))
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
D = pd.DataFrame(R); print('Buy Box rows', len(D), 'asins', D.a.nunique(), 'with stock share', int(D.share.notna().sum()), flush=True)
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
