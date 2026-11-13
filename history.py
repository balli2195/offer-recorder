"""Per-listing offer history for the Hub's price chart (Nick, Nov 13 2026).

An index over the offer messages the recorder already saves (the raw files in offers/ stay the record): for each message, the
listing, the time Amazon gives for the change, every new-condition offer (seller, price + shipping, FBA or not, backordered or not)
and the Buy Box winner. Kept in /var/data/history.db for 14 days, so a chart of the last day or week opens in a moment instead of
scanning the hourly files. Backfilled from the raw files on first start (last 3 days, in the background, yielding often).
"""
import datetime as dt, glob, gzip, json, os, sqlite3, threading, time

KEEP_DAYS = 14
AMAZON = 'ATVPDKIKX0DER'
AMAZON_ME = 'A2UYRB3AYUMGAJ'   # Nick's seller id


def ts(s):
    return dt.datetime.fromisoformat(s.replace('Z', '+00:00')).timestamp()


def parse(body):
    """(message-free row) asin, t, offers [[seller, landed, fba, backordered]], winner, all flagged winners, or None."""
    try:
        p = json.loads(body)['Payload']['AnyOfferChangedNotification']
        trig = p['OfferChangeTrigger']
        offs, wins = [], []
        for o in p.get('Offers') or []:
            if (o.get('SubCondition') or '').lower() != 'new':
                continue
            price = (o.get('ListingPrice') or {}).get('Amount') or 0.0
            ship = (o.get('Shipping') or {}).get('Amount') or 0.0
            bo = ((o.get('ShippingTime') or {}).get('AvailabilityType') or '') == 'FUTURE_WITH_DATE'
            offs.append([o.get('SellerId'), round(price + ship, 2), 1 if o.get('IsFulfilledByAmazon') else 0, 1 if bo else 0])
            if o.get('IsBuyBoxWinner'):
                wins.append(o.get('SellerId'))
        # Amazon often flags 2-3 offers as winner at once; only one holds the featured offer (Nov 13). 'winner' is Nick when he is
        # flagged at all (so the cut check and the chart don't miss a shared Buy Box), else the first flagged; all are in 'winners'.
        win = AMAZON_ME if AMAZON_ME in wins else (wins[0] if wins else None)
        return trig['ASIN'], ts(trig['TimeOfOfferChange']), offs, win, wins
    except Exception:
        return None


class History:
    def __init__(self, data_dir):
        self.data = data_dir
        self.path = os.path.join(data_dir, 'history.db')
        self.lock = threading.Lock()
        self.db = sqlite3.connect(self.path, check_same_thread=False, timeout=30)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('CREATE TABLE IF NOT EXISTS h (asin TEXT, t REAL, offers TEXT, winner TEXT)')
        self.db.execute('CREATE INDEX IF NOT EXISTS h_at ON h (asin, t)')
        if 'winners' not in [r[1] for r in self.db.execute('PRAGMA table_info(h)')]:
            self.db.execute('ALTER TABLE h ADD COLUMN winners TEXT')   # rows before Nov 13 keep NULL (one winner known)
        self.db.execute('CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT)')
        self.db.commit()
        self.stats = {'rows': None, 'backfill': None, 'last_error': None}

    def add(self, bodies):
        rows = []
        for b in bodies:
            x = parse(b)
            if x:
                rows.append((x[0], x[1], json.dumps(x[2], separators=(',', ':')), x[3], json.dumps(x[4]) if len(x[4]) > 1 else None))
        if rows:
            with self.lock:
                self.db.executemany('INSERT INTO h (asin, t, offers, winner, winners) VALUES (?, ?, ?, ?, ?)', rows)
                self.db.commit()

    def prune(self):
        with self.lock:
            self.db.execute('DELETE FROM h WHERE t < ?', (time.time() - KEEP_DAYS * 86400,))
            self.db.commit()
            self.stats['rows'] = self.db.execute('SELECT COUNT(*) FROM h').fetchone()[0]

    def backfill(self, days=3):
        """Once: load the last few days from the raw offer files (duplicates of what add() wrote since start are skipped)."""
        with self.lock:
            done = self.db.execute("SELECT v FROM meta WHERE k = 'backfill'").fetchone()
            first = self.db.execute('SELECT MIN(t) FROM h').fetchone()[0]
        if done:
            return
        cut = time.time() - days * 86400
        n = 0
        files = sorted(glob.glob(os.path.join(self.data, 'offers', '*', '*.jsonl.gz')))
        for fp in files:
            day = os.path.basename(os.path.dirname(fp))
            try:
                if dt.datetime.strptime(day, '%Y-%m-%d').replace(tzinfo=dt.timezone.utc).timestamp() + 86400 < cut:
                    continue
            except ValueError:
                continue
            rows, seen = [], set()
            try:
                with gzip.open(fp, 'rt') as f:
                    for i, line in enumerate(f):
                        if i % 200 == 0:
                            time.sleep(0.002)   # it runs inside the recorder: yield so recording and pricing keep up
                        try:
                            m = json.loads(line)
                        except ValueError:
                            continue
                        if m.get('message_id') in seen:
                            continue
                        seen.add(m.get('message_id'))
                        x = parse(m.get('body') or '')
                        if x and x[1] >= cut and (first is None or x[1] < first):
                            rows.append((x[0], x[1], json.dumps(x[2], separators=(',', ':')), x[3], json.dumps(x[4]) if len(x[4]) > 1 else None))
            except (OSError, EOFError):
                pass
            if rows:
                with self.lock:
                    self.db.executemany('INSERT INTO h (asin, t, offers, winner, winners) VALUES (?, ?, ?, ?, ?)', rows)
                    self.db.commit()
                n += len(rows)
            time.sleep(0.05)   # leave the CPU to recording
        with self.lock:
            self.db.execute("INSERT OR REPLACE INTO meta VALUES ('backfill', ?)", (json.dumps({'rows': n, 'at': time.time()}),))
            self.db.commit()
        self.stats['backfill'] = n

    def held(self, asin, since, me):
        """Did seller `me` hold the Buy Box at any moment since `since`? (The last message before the window gives the state at its start.)"""
        with self.lock:
            prev = self.db.execute('SELECT winner FROM h WHERE asin = ? AND t < ? ORDER BY t DESC LIMIT 1', (asin, since)).fetchone()
            hit = self.db.execute('SELECT 1 FROM h WHERE asin = ? AND t >= ? AND winner = ? LIMIT 1', (asin, since, me)).fetchone()
        return bool(hit) or bool(prev and prev[0] == me)

    def query(self, asin, hours=48, me=None):
        """Change points only: a row is kept when any offer's price, its presence, or the Buy Box winner changed."""
        since = time.time() - hours * 3600
        with self.lock:
            rows = self.db.execute('SELECT t, offers, winner, winners FROM h WHERE asin = ? AND t >= ? ORDER BY t', (asin, since)).fetchall()
        # the state just before the window, so lines start at the left edge
        with self.lock:
            prev = self.db.execute('SELECT t, offers, winner, winners FROM h WHERE asin = ? AND t < ? ORDER BY t DESC LIMIT 1', (asin, since)).fetchone()
        if prev:
            rows = [(since, prev[1], prev[2], prev[3])] + rows
        sellers, pts, last = {}, [], None
        for t, offs, win, wins in rows:
            o = json.loads(offs)
            nw = len(json.loads(wins)) if wins else (1 if win else 0)
            key = (tuple(sorted((s, p, b) for s, p, f, b in o)), win, nw)
            if key == last:
                continue
            last = key
            for s, p, f, b in o:
                d = sellers.setdefault(s, {'fba': bool(f), 'amazon': s == AMAZON, 'me': s == me, 'low': p, 'high': p, 'changes': 0})
                d['low'], d['high'] = min(d['low'], p), max(d['high'], p)
            pts.append([round(t), {s: [p, b] for s, p, f, b in o}, win, nw])   # nw = offers flagged winner (2+ = shared flag)
        # count each seller's price changes
        prevp = {}
        for _, o, _, _ in pts:
            for s, (p, b) in o.items():
                if s in prevp and abs(prevp[s] - p) >= 0.005:
                    sellers[s]['changes'] += 1
                prevp[s] = p
        return {'asin': asin, 'hours': hours, 'from': round(since), 'to': round(time.time()), 'me': me, 'sellers': sellers, 'points': pts}
