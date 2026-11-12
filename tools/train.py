"""Train the shadow repricer's models from the recorder's last N days and (optionally) upload model.json.
Usage: RECORDER_KEY=... python3 tools/train.py [--days 7] [--upload] [--out model.json]
Models:
  bb_foep / bb_nofoep  logistic: does Nick hold the Buy Box in the 30 min after a moment (fractional label), from his price vs the
                       lowest rival, vs Amazon's target (bb_foep only), number of rivals, stock share. Validated on the last 2 days.
  rivals               per rival: share of undercuts (by anyone) it matched or beat within 60 min, Beta-smoothed toward the pooled rate;
                       median lag; lowest price seen.
  war_table            median % drop of the listing low 24 h after an undercut, by the reactive rate of the rivals undercut.
"""
import argparse, bisect, collections, datetime as dt, gzip, io, json, math, os, sys
import numpy as np, pandas as pd, requests
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import shadow as S
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss

ME = S.ME; CAP = 1800
U = os.environ.get('RECORDER_URL', 'https://offer-recorder.onrender.com'); H = {'X-Recorder-Key': os.environ['RECORDER_KEY']}
ap = argparse.ArgumentParser(); ap.add_argument('--days', type=int, default=7); ap.add_argument('--upload', action='store_true'); ap.add_argument('--out', default='model.json'); ap.add_argument('--cache', default=None)
A = ap.parse_args()
ts_ = lambda s: dt.datetime.fromisoformat(s.replace('Z', '+00:00')).timestamp()
cut = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=A.days)).strftime('%Y-%m-%d')
files = requests.get(U + '/files', headers=H, timeout=60).json()
def get(path):
    if A.cache:
        cp = os.path.join(A.cache, path.replace('/', '_'))
        today = dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%d')
        if os.path.exists(cp) and today not in path:  # files from earlier days are complete
            return gzip.open(cp, 'rt').read().splitlines()
    for k in range(4):
        try:
            b = requests.get(U + '/files/' + path, headers=H, timeout=300).content
            if A.cache:
                os.makedirs(A.cache, exist_ok=True); open(os.path.join(A.cache, path.replace('/', '_')), 'wb').write(b)
            return gzip.open(io.BytesIO(b), 'rt').read().splitlines()
        except Exception as e:
            err = e
    print('skipped unreadable file', path, err, flush=True)
    return []
def want(kind):
    return [f['path'] for f in files if f['path'].startswith(kind + '/') and f['path'].split('/')[1][:10] >= cut]
# ---------------- offers
seen = set(); by = collections.defaultdict(list)
for pth in want('offers'):
    for l in get(pth):
        d = json.loads(l)
        if d['message_id'] in seen: continue
        seen.add(d['message_id'])
        p = json.loads(d['body'])['Payload']['AnyOfferChangedNotification']
        t = ts_(p['OfferChangeTrigger']['TimeOfOfferChange'])
        # FBA new offers available now, plus Nick's own whatever its availability (same rule as shadow.fba_offers)
        off = [(o['SellerId'], o['ListingPrice']['Amount'] + o['Shipping']['Amount'], bool(o.get('IsBuyBoxWinner'))) for o in p['Offers']
               if o.get('SubCondition') == 'new' and o.get('IsFulfilledByAmazon') and (o.get('SellerId') == ME or not S.backordered(o))]
        me_bo = any(o.get('SellerId') == ME and S.backordered(o) for o in p['Offers'])
        by[p['OfferChangeTrigger']['ASIN']].append((t, off, S.suppression(p)[0] or me_bo))
print('messages', len(seen), 'asins', len(by), flush=True)
IDX = {}
SUPP = {}   # per message: Buy Box suppressed (no Buy Box shown) or Nick's offer backordered; those moments are left out of the Buy Box model
for a, ms in by.items():
    ms.sort(key=lambda x: x[0])
    IDX[a] = (np.array([m[0] for m in ms]), [{s: p for s, p, w in off} for t, off, _ in ms], [{s for s, p, w in off if w} for t, off, _ in ms])
    SUPP[a] = [m[2] for m in ms]
T1 = max(v[0][-1] for v in IDX.values())
# ---------------- rival reactions to undercuts (by anyone)
react = collections.defaultdict(lambda: [0, 0]); lags = collections.defaultdict(list); floor = collections.defaultdict(lambda: 1e9); events = []
for a, (ts, P, W) in IDX.items():
    for i in range(len(ts)):
        for s, p in P[i].items():
            floor[(s, a)] = min(floor[(s, a)], p)
    for i in range(1, len(ts)):
        P0, P1 = P[i - 1], P[i]
        if len(P1) < 2: continue
        for s, p in P1.items():
            if s not in P0 or p >= P0[s] - 0.005: continue
            oth = [q for x, q in P1.items() if x != s]
            if p > min(oth) - 0.01: continue
            t = ts[i]; L0 = min(P0.values()); at_low = [x for x, q in P0.items() if x != s and q <= L0 * 1.005]
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
            low24 = None
            if t + 86400 <= T1:
                k = bisect.bisect_right(ts, t + 86400) - 1
                low24 = 100.0 * (min(P[k].values()) / L0 - 1.0) if P[k] else None
            events.append((at_low, low24))
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
print('rivals', len(rivals), 'pooled rate', rival_default, flush=True)
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
# ---------------- Buy Box training rows: FOEP readings matched to the offer state
fo = []
for pth in want('foep'):
    for l in get(pth):
        x = json.loads(l); b = x.get('body') or {}; r = (b.get('featuredOfferExpectedPriceResults') or [{}])[0]
        a = (b.get('offerIdentifier') or {}).get('asin')
        if r.get('resultStatus') == 'VALID_FOEP' and a:
            fo.append((ts_(x['taken_at']), x['sku'], a, r['featuredOfferExpectedPrice']['listingPrice']['amount']))
inv = collections.defaultdict(list)
for pth in want('inventory'):
    for l in get(pth):
        x = json.loads(l); t = ts_(x['taken_at'])
        for r in x['rows']: inv[r['sellerSku']].append((t, (r.get('inventoryDetails') or {}).get('fulfillableQuantity') or 0))
kp = collections.defaultdict(list)
sh = S.Shadow.__new__(S.Shadow); sh.lock = __import__('threading').Lock(); sh.rival_stock = {}
for pth in want('keepa'):
    for l in get(pth):
        x = json.loads(l); t = ts_(x['taken_at']); sh.rival_stock = {}
        sh.on_keepa_products(x['response'].get('products'))
        for a, (_, v) in sh.rival_stock.items(): kp[a].append((t, v))
def lookup(series, t, max_age):
    if not series: return None
    i = bisect.bisect_right([s[0] for s in series], t) - 1
    return series[i][1] if i >= 0 and t - series[i][0] <= max_age else None
def bb_next(a, t, dur=1800):
    ts, P, W = IDX[a]; i = max(0, bisect.bisect_right(ts, t) - 1); tot = win = 0
    while i < len(ts) and ts[i] < t + dur:
        nxt = ts[i + 1] if i + 1 < len(ts) else t + dur; seg = max(0, min(nxt, t + dur, ts[i] + CAP) - max(ts[i], t))
        if ME in P[i]: tot += seg; win += seg * (ME in W[i])
        i += 1
    return win / tot if tot > 0 else None
for k in inv: inv[k].sort()
for k in kp: kp[k].sort()
R = []
for t, sku, a, f in fo:
    if a not in IDX or t + 1800 > T1: continue
    ts, P, W = IDX[a]; i = bisect.bisect_right(ts, t) - 1
    if i < 0 or t - ts[i] > 6 * 3600: continue
    Pi = P[i]
    if ME not in Pi or len(Pi) < 2: continue
    if SUPP[a][i]: continue   # no Buy Box to win: the repricer matches the lowest offer instead (shadow.decide_suppressed)
    y = bb_next(a, t)
    if y is None: continue
    me = Pi[ME]; low = min(q for s, q in Pi.items() if s != ME)
    mine = lookup(inv.get(sku), t, 2 * 3600); riv = lookup(kp.get(a), t, 8 * 3600)
    share = mine / (mine + riv) if (mine is not None and riv is not None and mine + riv > 0) else None
    R.append(dict(t=t, a=a, y=y, me=me, low=low, n=len(Pi) - 1, foep=f, share=share))
D = pd.DataFrame(R); print('Buy Box rows', len(D), 'asins', D.a.nunique(), 'with stock share', int(D.share.notna().sum()), flush=True)
def X_of(df, with_foep):
    keys = None; out = []
    for r in df.itertuples():
        x = S.feats(r.me, r.low, r.n, r.foep if with_foep else None, None if pd.isna(r.share) else r.share)
        out.append(x)
    keys = sorted(out[0].keys())
    return np.array([[x[k] for k in keys] for x in out]), keys
def fit(df, with_foep):
    X, keys = X_of(df, with_foep); y = df.y.values
    XX = np.vstack([X, X]); yy = np.r_[np.ones(len(y)), np.zeros(len(y))]; ww = np.r_[y, 1 - y] + 1e-6
    j = keys.index('b'); m = LogisticRegression(C=1.0, max_iter=2000, fit_intercept=False).fit(XX, yy, sample_weight=ww)
    return dict(zip(keys, [float(c) for c in m.coef_[0]]))
cutoff = T1 - 2 * 86400; tr, te = D[D.t < cutoff], D[D.t >= cutoff]
val = {}
for name, wf in (('bb_foep', True), ('bb_nofoep', False)):
    c = fit(tr, wf); pred = np.array([S.p_win({name: {'coef': c}}, x) for x in [S.feats(r.me, r.low, r.n, r.foep if wf else None, None if pd.isna(r.share) else r.share) for r in te.itertuples()]])
    yb = (te.y >= 0.5).astype(int)
    val[name] = {'test_rows': int(len(te)), 'auc': round(float(roc_auc_score(yb, pred)), 4), 'brier': round(float(brier_score_loss(yb, pred)), 4),
                 'mean_pred': round(float(pred.mean()), 4), 'mean_actual': round(float(te.y.mean()), 4)}
rule = (te.me <= te.foep + 0.01).astype(float); val['rule_at_or_below_target'] = {'auc': round(float(roc_auc_score((te.y >= 0.5).astype(int), rule)), 4)}
print('validation (last 2 days held out)', json.dumps(val), flush=True)
model = {'trained_at': dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds'), 'window_days': A.days, 'messages': len(seen),
         'bb_foep': {'coef': fit(D, True)}, 'bb_nofoep': {'coef': fit(D, False)}, 'validation': val,
         'rivals': rivals, 'rival_default': rival_default, 'war_table': war, 'bb_rows': int(len(D))}
raw = json.dumps(model); open(A.out, 'w').write(raw); print('wrote', A.out, len(raw), 'bytes')
if A.upload:
    r = requests.post(U + '/model', headers={**H, 'content-type': 'application/json'}, data=raw, timeout=120); print('upload', r.status_code, r.text[:200])
