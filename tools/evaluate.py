"""Scorecard for shadow mode. Usage: RECORDER_KEY=... python3 tools/evaluate.py [--days 3]
  1. Calibration: the model's Buy Box chance at Nick's actual price vs what happened in the next 30 min.
  2. Actions: how often it would hold, raise, lower or undercut, and the expected gain it claims.
  3. Agreement: when it recommended a different price, did Seller Snap get there within 2 h anyway?
  4. Live: changes applied and blocked (if live mode was on)."""
import argparse, bisect, collections, datetime as dt, gzip, io, json, os
import numpy as np, pandas as pd, requests
ME = 'A2UYRB3AYUMGAJ'; CAP = 1800
U = os.environ.get('RECORDER_URL', 'https://offer-recorder.onrender.com'); H = {'X-Recorder-Key': os.environ['RECORDER_KEY']}
ap = argparse.ArgumentParser(); ap.add_argument('--days', type=int, default=3); A = ap.parse_args()
cut = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=A.days)).strftime('%Y-%m-%d')
F = requests.get(U + '/files', headers=H, timeout=60).json()
get = lambda p: gzip.open(io.BytesIO(requests.get(U + '/files/' + p, headers=H, timeout=300).content), 'rt').read().splitlines()
pick = lambda k: [f['path'] for f in F if f['path'].startswith(k + '/') and f['path'].split('/')[1][:10] >= cut]
ts_ = lambda s: dt.datetime.fromisoformat(s.replace('Z', '+00:00')).timestamp()
D = pd.DataFrame([json.loads(l) for p in pick('shadow') for l in get(p)])
if D.empty:
    print('no shadow decisions yet'); raise SystemExit
D['ts'] = D.t.map(ts_)
seen = set(); by = collections.defaultdict(list)
for p in pick('offers'):
    for l in get(p):
        d = json.loads(l)
        if d['message_id'] in seen: continue
        seen.add(d['message_id']); pl = json.loads(d['body'])['Payload']['AnyOfferChangedNotification']
        t = ts_(pl['OfferChangeTrigger']['TimeOfOfferChange'])
        by[pl['OfferChangeTrigger']['ASIN']].append((t, {o['SellerId']: o['ListingPrice']['Amount'] + o['Shipping']['Amount'] for o in pl['Offers'] if o.get('SubCondition') == 'new' and o.get('IsFulfilledByAmazon')},
                                                     {o['SellerId'] for o in pl['Offers'] if o.get('IsBuyBoxWinner')}))
for a in by: by[a].sort(key=lambda x: x[0])
T1 = max(v[-1][0] for v in by.values())
def bb_next(a, t, dur=1800):
    ms = by.get(a) or []; ts = [m[0] for m in ms]; i = max(0, bisect.bisect_right(ts, t) - 1); tot = win = 0
    while i < len(ms) and ms[i][0] < t + dur:
        nxt = ms[i + 1][0] if i + 1 < len(ms) else t + dur; seg = max(0, min(nxt, t + dur, ms[i][0] + CAP) - max(ms[i][0], t))
        if ME in ms[i][1]: tot += seg; win += seg * (ME in ms[i][2])
        i += 1
    return win / tot if tot > 0 else np.nan
def reached(a, t, price, dur=7200):
    for m in by.get(a) or []:
        if t < m[0] <= t + dur and ME in m[1] and abs(m[1][ME] - price) <= 0.015: return True
    return False
X = D[D.ts + 1800 <= T1].copy(); X['pred'] = X.cur.map(lambda c: c['p_win']); X['actual'] = [bb_next(a, t) for a, t in zip(X.asin, X.ts)]
X = X.dropna(subset=['actual'])
X['band'] = pd.cut(X.pred, [0, .1, .25, .5, .75, .9, 1.0001], right=False)
cal = X.groupby('band', observed=True).agg(decisions=('pred', 'size'), predicted=('pred', 'mean'), actual=('actual', 'mean')).round(3)
print(f'decisions {len(D)} over {D.asin.nunique()} listings, {D.t.min()} to {D.t.max()}')
print('\n1. CALIBRATION (Buy Box chance at Nick\'s actual price vs the next 30 min)\n', cal.to_string())
print('   mean predicted', round(X.pred.mean(), 3), 'actual', round(X.actual.mean(), 3), '| with target price', round(X[X.foep.notna()].pred.mean(), 3), 'vs', round(X[X.foep.notna()].actual.mean(), 3))
print('\n2. ACTIONS\n', (D.action.value_counts(normalize=True) * 100).round(1).to_string())
ch = D[D.action != 'hold']; print('   expected gain per suggested change: median $', round(ch.gain_24h.median(), 2), '| move size % median', round((100 * (ch.rec_price / ch.my_price - 1)).median(), 2))
Y = ch[ch.ts + 7200 <= T1].copy(); Y['ss_got_there'] = [reached(a, t, p) for a, t, p in zip(Y.asin, Y.ts, Y.rec_price)]
print('\n3. AGREEMENT: Seller Snap reached the recommended price within 2 h:', round(100 * Y.ss_got_there.mean(), 1), '% of', len(Y), 'suggestions;')
print(Y.groupby('action').ss_got_there.agg(['size', 'mean']).round(3).to_string())
L = pd.DataFrame([json.loads(l) for p in pick('live') for l in get(p)])
print('\n4. LIVE:', 'no live changes' if L.empty else L.groupby(['status_code']).size().to_string())
