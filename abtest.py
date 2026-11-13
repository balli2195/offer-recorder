"""Alternating-day test: Nick's repricer vs Seller Snap on the same SKUs (Nick, Nov 12 2026).

Design
  - Test SKUs are split into groups A and B (matched on sales speed and price). Each test day one group is priced by Nick's repricer
    and the other by Seller Snap; they swap every day at SWITCH_HOUR (3 AM Central). So every day, and every weekday, hits both sides.
  - Both sides use the same limits: Seller Snap's min and max (Nick's choice). For test SKUs the repricer reads Seller Snap's min/max
    and does not raise the max to a rival's Buy Box price.
  - Each switch turns Seller Snap repricing off for the group the repricer takes and on for the other, through the Seller Snap box
    (Apps Script action set_repricing, limited there to the approved test SKUs).
  - Every shadow decision on a test SKU is tagged with its arm ('repricer' or 'sellersnap') and whether it falls in the WASHOUT_H hours
    after a switch (carry-over from the other repricer's last price; left out of the scorecard).

Safety
  - dry_run (the default): the schedule, tags and planned switches run, but Seller Snap is not switched and no price is changed.
  - Live price changes still need everything live.py requires: Render env LIVE_MODE=on and the live config enabled (the Hub's Stop button
    turns it off, which stops the test's price changes too). Test SKUs don't need to be in the live SKU list.
  - If the Seller Snap switch fails for a SKU, live.py's own gate still blocks it while Seller Snap shows it as repricing.
Every switch (planned or done) is logged to abtest/<month>.jsonl.gz; the config lives in abtest/config.json.
"""
import datetime as dt, gzip, json, os, threading, time
from zoneinfo import ZoneInfo

import requests

CT = ZoneInfo('America/Chicago')
DEFAULTS = {'enabled': False, 'dry_run': True, 'start': None, 'switch_hour': 3, 'washout_h': 2.0, 'groups': {}, 'asins': {},
            'box_url': '', 'box_key': '', 'limits': 'sellersnap'}
PUBLIC = ('enabled', 'dry_run', 'start', 'switch_hour', 'washout_h', 'limits')


class ABTest:
    def __init__(self, data_dir, append_gz, iso, live=None, wait=time.sleep, http=requests):
        self.data, self.append_gz, self.iso, self.live, self.wait, self.http = data_dir, append_gz, iso, live, wait, http
        self.lock = threading.Lock()
        self.cfg = dict(DEFAULTS)
        self.last_switch = None     # {'t', 'day', 'repricer_group', 'dry_run', 'ok', 'errors', ...}
        self.load()

    # ---- config
    def path(self):
        return os.path.join(self.data, 'abtest', 'config.json')

    def load(self):
        try:
            with open(self.path()) as f:
                x = json.load(f)
            self.cfg = {**DEFAULTS, **x.get('cfg', {})}
            self.last_switch = x.get('last_switch')
        except (OSError, ValueError):
            pass

    def save(self):
        os.makedirs(os.path.dirname(self.path()), exist_ok=True)
        tmp = self.path() + '.tmp'
        with open(tmp, 'w') as f:
            json.dump({'cfg': self.cfg, 'last_switch': self.last_switch}, f)
        os.replace(tmp, self.path())

    def set_cfg(self, upd):
        with self.lock:
            c = dict(self.cfg)
            for k in ('enabled', 'dry_run'):
                if k in upd:
                    c[k] = bool(upd[k])
            for k in ('switch_hour',):
                if k in upd:
                    c[k] = int(upd[k]) % 24
            if 'washout_h' in upd:
                c['washout_h'] = max(0.0, float(upd['washout_h']))
            if 'start' in upd:
                dt.date.fromisoformat(upd['start'])   # YYYY-MM-DD: the Central date whose switch hour begins day 0
                c['start'] = upd['start']
            if 'groups' in upd:
                g = {str(k): str(v).upper() for k, v in (upd['groups'] or {}).items()}
                if any(v not in ('A', 'B') for v in g.values()):
                    raise ValueError('groups must be A or B')
                c['groups'] = g
            if 'asins' in upd:
                c['asins'] = {str(k): str(v) for k, v in (upd['asins'] or {}).items()}
            for k in ('box_url', 'box_key'):
                if k in upd:
                    c[k] = str(upd[k] or '')
            self.cfg = c
            self.save()
        return self.public()

    def public(self):
        c = self.cfg
        now = time.time()
        g = c['groups']
        day = self.day_index(now)
        return {**{k: c[k] for k in PUBLIC}, 'skus': len(g), 'group_sizes': {x: sum(1 for v in g.values() if v == x) for x in 'AB'},
                'box_ready': bool(c['box_url'] and c['box_key']), 'day': day, 'repricer_group': self.repricer_group(now),
                'washout_now': self.washout(now), 'next_switch': self.iso(dt.datetime.fromtimestamp(self.next_switch(now), dt.timezone.utc)),
                'last_switch': {k: v for k, v in (self.last_switch or {}).items() if k != 'results'}}

    # ---- schedule
    def _test_date(self, t):
        """The test day a moment belongs to: the Central date, with the day starting at the switch hour."""
        return (dt.datetime.fromtimestamp(t, CT) - dt.timedelta(hours=self.cfg['switch_hour'])).date()

    def day_index(self, t):
        if not self.cfg.get('start'):
            return None
        return (self._test_date(t) - dt.date.fromisoformat(self.cfg['start'])).days

    def running(self, t=None):
        d = self.day_index(t or time.time())
        return bool(self.cfg.get('enabled') and self.cfg['groups'] and d is not None and d >= 0)

    def repricer_group(self, t):
        d = self.day_index(t)
        if d is None or d < 0:
            return None
        return 'A' if d % 2 == 0 else 'B'

    def in_test(self, sku):
        return self.running() and sku in self.cfg['groups']

    def arm(self, sku, t=None):
        t = t or time.time()
        if not self.running(t) or sku not in self.cfg['groups']:
            return None
        return 'repricer' if self.cfg['groups'][sku] == self.repricer_group(t) else 'sellersnap'

    def washout(self, t):
        if not self.running(t):
            return False
        ct = dt.datetime.fromtimestamp(t, CT)
        start = ct.replace(hour=self.cfg['switch_hour'], minute=0, second=0, microsecond=0)
        if start > ct:
            start -= dt.timedelta(days=1)
        return (ct - start).total_seconds() < self.cfg['washout_h'] * 3600

    def next_switch(self, t):
        ct = dt.datetime.fromtimestamp(t, CT)
        n = ct.replace(hour=self.cfg['switch_hour'], minute=0, second=0, microsecond=0)
        if n <= ct:
            n = (n + dt.timedelta(days=1)).replace(hour=self.cfg['switch_hour'])
        return n.timestamp()

    def live_allows(self, sku, t=None):
        """For live.py: None = not a test SKU (normal rules); True/False = test SKU, priced live by the repricer right now or not."""
        if not self.running(t) or sku not in self.cfg['groups']:
            return None
        return (not self.cfg['dry_run']) and self.arm(sku, t) == 'repricer'

    # ---- Seller Snap switch
    def box(self, action, **kw):
        """Seller Snap box call: Apps Script answers a POST with a 302 to a one-time URL, which is followed with a single GET."""
        r = self.http.post(self.cfg['box_url'], json={'key': self.cfg['box_key'], 'action': action, **kw}, allow_redirects=False, timeout=120)
        if r.status_code in (301, 302, 303) and r.headers.get('Location'):
            r = self.http.get(r.headers['Location'], timeout=120)
        r.raise_for_status()
        return r.json()

    def switch(self, t=None):
        """Applies today's arms: Seller Snap off for the repricer's group, on for the other. Logged either way."""
        t = t or time.time()
        if not self.running(t):
            return {'ok': False, 'error': 'test not running'}
        rg = self.repricer_group(t)
        g = self.cfg['groups']
        off = sorted(s for s, x in g.items() if x == rg)
        on = sorted(s for s, x in g.items() if x != rg)
        rec = {'t': self.iso(), 'day': self.day_index(t), 'repricer_group': rg, 'dry_run': self.cfg['dry_run'], 'off': len(off), 'on': len(on)}
        if self.cfg['dry_run']:
            rec.update(ok=True, note='dry run: Seller Snap not switched')
        elif not (self.cfg['box_url'] and self.cfg['box_key']):
            rec.update(ok=False, errors=['Seller Snap box not configured'])
        else:
            errors, results = [], []
            for skus, flag in ((off, False), (on, True)):   # off first: Seller Snap stops before the repricer starts on those SKUs
                for i in range(0, len(skus), 100):
                    chunk = skus[i:i + 100]
                    try:
                        res = self.box('set_repricing', skus=chunk, on=flag)
                        results.append({'on': flag, 'n': len(chunk), 'res': res})
                        bad = set(res.get('failed') or []) if isinstance(res, dict) else set()
                        if isinstance(res, dict) and res.get('error'):
                            errors.append(str(res['error'])[:300]); bad = set(chunk)
                        if self.live:
                            with self.live.lock:
                                for s in chunk:
                                    if s not in bad:
                                        self.live.ss_repricing[s] = flag
                    except Exception as e:
                        errors.append(f'{type(e).__name__}: {str(e)[:200]}')
                    self.wait(2)
            rec.update(ok=not errors, errors=errors, results=results)
        with self.lock:
            self.last_switch = rec
            self.save()
        self.append_gz(os.path.join(self.data, 'abtest', dt.datetime.now(dt.timezone.utc).strftime('%Y-%m') + '.jsonl.gz'), [json.dumps(rec)])
        return rec

    def end(self):
        """Ends the test: Seller Snap back on for every test SKU (unless dry run), test disabled. The groups are kept for the record."""
        rec = {'t': self.iso(), 'end': True, 'dry_run': self.cfg['dry_run']}
        if not self.cfg['dry_run'] and self.cfg['box_url'] and self.cfg['box_key']:
            errors, skus = [], sorted(self.cfg['groups'])
            for i in range(0, len(skus), 100):
                try:
                    res = self.box('set_repricing', skus=skus[i:i + 100], on=True)
                    if isinstance(res, dict) and res.get('error'):
                        errors.append(str(res['error'])[:300])
                except Exception as e:
                    errors.append(f'{type(e).__name__}: {str(e)[:200]}')
            rec.update(ok=not errors, errors=errors)
        with self.lock:
            self.cfg['enabled'] = False
            self.last_switch = rec
            self.save()
        self.append_gz(os.path.join(self.data, 'abtest', dt.datetime.now(dt.timezone.utc).strftime('%Y-%m') + '.jsonl.gz'), [json.dumps(rec)])
        return rec

    def loop(self):
        while True:
            nxt = self.next_switch(time.time())
            self.wait(max(30, nxt - time.time() + 5))
            try:
                if self.running():
                    self.switch()
            except Exception as e:
                with self.lock:
                    self.last_switch = {'t': self.iso(), 'ok': False, 'errors': [f'{type(e).__name__}: {str(e)[:200]}']}
