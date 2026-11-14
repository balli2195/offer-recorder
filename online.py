"""Per-listing Buy Box correction learned online (Nick, Nov 14 2026), shadow mode: logged next to the model, not used for prices.

The nightly model is the prior for every listing. Each time one of Nick's prices has stood and Amazon's segment snapshots show the
share he actually got, the gap between that share and the model's prediction (on the log-odds scale) is added to the listing's
running evidence. The correction is
    offset = (sum of gaps) / (n + K)
with n the number of results, both faded with a half-life, so a new listing starts at the model (offset 0) and a busy listing moves
toward its own record within about a day, and keeps following it as it changes.

A result = one "window": from the decision that first saw Nick at a price until his price changes or 6 h pass (the held label);
its outcome is the mean of his featured share over the snapshots in the window; the gap is logit(outcome) - logit(model's
prediction at the window start), both clipped to 2-98%.

Replay on Nov 10-13 (45,487 predictions, no peeking): K = 20 results, half-life 12 h cut the error (MSE) 5.0% (holding the price
-6.5%, after a change -2.4%) vs the model alone.
"""
import json, math, os, threading, time

K = 20.0
HALF_LIFE_S = 12 * 3600
MAX_WINDOW_S = 6 * 3600
CLIP = (0.02, 0.98)


def logit(p):
    p = min(max(p, CLIP[0]), CLIP[1])
    return math.log(p / (1 - p))


def sigmoid(z):
    return 1 / (1 + math.exp(-z))


class Online:
    def __init__(self, path=None, k=K, half_life=HALF_LIFE_S, max_window=MAX_WINDOW_S):
        self.path, self.k, self.half, self.maxw = path, k, half_life, max_window
        self.ev = {}     # asin -> [n, sum of gaps, time of last fade]
        self.win = {}    # asin -> [start, price, model prediction, [shares]]
        self.closed = 0
        self.lock = threading.Lock()
        self._saved = 0
        self.load()

    def _faded(self, asin, now):
        e = self.ev.get(asin)
        if not e:
            return 0.0, 0.0
        w = 0.5 ** (max(now - e[2], 0) / self.half)
        return e[0] * w, e[1] * w

    def offset(self, asin, now=None):
        """(offset on the log-odds scale, effective number of results)."""
        now = time.time() if now is None else now
        with self.lock:
            n, s = self._faded(asin, now)
        return s / (n + self.k), n

    def corrected(self, asin, p, now=None):
        off, n = self.offset(asin, now)
        return sigmoid(logit(p) + off), off, n

    def _close(self, asin, now):
        w = self.win.pop(asin, None)
        if not w or not w[3]:
            return False
        y = sum(w[3]) / len(w[3])
        gap = logit(y) - logit(w[2])
        n, s = self._faded(asin, now)
        self.ev[asin] = [n + 1.0, s + gap, now]
        self.closed += 1
        return True

    def on_decision(self, asin, price, p_model, now=None):
        """A decision saw Nick at `price` with the model's prediction p_model (his share while that price stands)."""
        now = time.time() if now is None else now
        if price is None or p_model is None:
            return
        with self.lock:
            w = self.win.get(asin)
            if w and (abs(w[1] - price) >= 0.005 or now - w[0] >= self.maxw):
                self._close(asin, now)
                w = None
            if not w:
                self.win[asin] = [now, price, float(p_model), []]
        self.save()

    def on_snapshot(self, asin, share, now=None):
        """One segment snapshot: Nick's share of the listing's featured page views (0-1)."""
        now = time.time() if now is None else now
        with self.lock:
            w = self.win.get(asin)
            if not w:
                return
            if now - w[0] >= self.maxw:
                self._close(asin, now)
                return
            w[3].append(float(share))

    def stats(self):
        with self.lock:
            now = time.time()
            ns = [self._faded(a, now)[0] for a in self.ev]
        return {'asins': len(ns), 'with_5plus_results': sum(1 for n in ns if n >= 5), 'windows_closed': self.closed,
                'open_windows': len(self.win), 'k': self.k, 'half_life_h': self.half / 3600}

    def save(self, force=False):
        if not self.path or (not force and time.time() - self._saved < 300):
            return
        try:
            with self.lock:
                obj = {'ev': self.ev, 'win': self.win, 'closed': self.closed}
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            tmp = self.path + '.tmp'
            with open(tmp, 'w') as f:
                json.dump(obj, f)
            os.replace(tmp, self.path)
            self._saved = time.time()
        except OSError:
            pass

    def load(self):
        if not self.path:
            return
        try:
            with open(self.path) as f:
                obj = json.load(f)
            self.ev = obj.get('ev') or {}
            self.win = obj.get('win') or {}
            self.closed = obj.get('closed') or 0
        except (FileNotFoundError, ValueError, OSError):
            pass
