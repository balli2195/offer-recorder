"""How Nick's sales depend on his Buy Box share (Nick, Nov 13 2026): units = rate x (s + beta x (1 - s)).

s is Nick's share of the listing's featured page views; beta is the sales rate with no featured share as a fraction of the rate at a
100% share (buyers who pick his offer from the other-sellers list). The repricer used beta = 0 (sales proportional to share) until
Nov 13; a week of orders and segment snapshots gave beta = 0.20 (bootstrap 95% 0.15-0.30), with test deviance per 3-h bin 0.946
vs 2.726 for beta = 0 (SALES_LINK doc).

Fit: Poisson, E[units of ASIN i in 3-h bin t] = A_i x T_t x (s_it + beta (1 - s_it)), with a level per ASIN (A_i) and per bin (T_t,
shared by all ASINs: time of day, weekday, holidays), fitted by alternating closed-form updates; beta by grid search on the
log-likelihood, 95% range from the profile likelihood (a drop of 1.92). ASINs with no sales in the window carry no information
and are left out.
"""
import numpy as np

DEFAULT_BETA = 0.20
GRID = np.r_[0.0, 0.02, 0.05, np.arange(0.1, 0.801, 0.025)]


def m(s, beta):
    """Sales weight of a Buy Box share s (0-1)."""
    return s + beta * (1.0 - s)


def _ll(ia, it, s, y, beta, na, nb, iters=60):
    w = m(s, beta)
    A = np.ones(na); T = np.ones(nb)
    for _ in range(iters):
        A = np.bincount(ia, y, minlength=na) / np.maximum(np.bincount(ia, T[it] * w, minlength=na), 1e-12)
        T = np.bincount(it, y, minlength=nb) / np.maximum(np.bincount(it, A[ia] * w, minlength=nb), 1e-12)
    mu = np.maximum(A[ia] * T[it] * w, 1e-12)
    return float(np.sum(y * np.log(mu) - mu)), A, T


def build(seg, sales, t0, t1, skip=(), bin_s=10800):
    """seg: asin -> (times, shares); sales: asin -> [(time, units)]; skip: [(start, end)] windows to leave out.
    Returns arrays (asin index, bin index, mean share in the bin, units in the bin) and the bin count."""
    nb = int((t1 - t0) // bin_s)
    rows = []
    names = {}
    for a, (tt, ss) in seg.items():
        acc = {}
        for t, s in zip(tt, ss):
            if t0 <= t < t1:
                acc.setdefault(int((t - t0) // bin_s), []).append(s)
        if not acc:
            continue
        units = {}
        for t, q in sales.get(a, []):
            if t0 <= t < t1:
                b = int((t - t0) // bin_s); units[b] = units.get(b, 0.0) + q
        if sum(units.values()) <= 0:
            continue
        i = names.setdefault(a, len(names))
        for b, v in acc.items():
            bt = t0 + b * bin_s
            if any(s0 < bt + bin_s and bt < s1 for s0, s1 in skip):
                continue
            rows.append((i, b, float(np.mean(v)), units.get(b, 0.0)))
    if not rows:
        return None
    r = np.array(rows)
    return r[:, 0].astype(int), r[:, 1].astype(int), r[:, 2], r[:, 3], len(names), nb


def fit(seg, sales, t0, t1, skip=(), test_from=None, bin_s=10800):
    """beta with its 95% range, and a held-out check (fitted before test_from, scored after it) against beta = 0."""
    d = build(seg, sales, t0, t1, skip, bin_s)
    if d is None:
        return None
    ia, it, s, y, na, nb = d
    ll = [_ll(ia, it, s, y, b, na, nb)[0] for b in GRID]
    k = int(np.argmax(ll)); beta = float(GRID[k])
    ok = [float(g) for g, l in zip(GRID, ll) if l >= ll[k] - 1.92]
    out = {'beta': round(beta, 3), 'lo': round(min(ok), 3), 'hi': round(max(ok), 3), 'asins': int(na), 'bins': int(len(y)),
           'units': float(y.sum()), 'mean_share': round(float(s.mean()), 4), 'll_gain_vs_0': round(ll[k] - ll[0], 1)}
    if test_from:
        cut = int((test_from - t0) // bin_s)
        tr, te = it < cut, it >= cut
        if tr.sum() and te.sum() and y[te].sum() > 0:
            def dev(b):
                _, A, _ = _ll(ia[tr], it[tr], s[tr], y[tr], b, na, nb)
                w = m(s[te], b); T = np.ones(nb)
                for _ in range(30):
                    T = np.bincount(it[te], y[te], minlength=nb) / np.maximum(np.bincount(it[te], A[ia[te]] * w, minlength=nb), 1e-12)
                mu = np.maximum(A[ia[te]] * T[it[te]] * w, 1e-9); yy = y[te]
                return float(2 * np.sum(np.where(yy > 0, yy * np.log(np.maximum(yy, 1e-12) / mu), 0) - (yy - mu)) / len(yy))
            out['test_dev'] = round(dev(beta), 4); out['test_dev_beta0'] = round(dev(0.0), 4)
    return out
