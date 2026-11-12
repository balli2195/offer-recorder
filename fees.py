"""Amazon's own fee estimates per SKU (Product Fees API, getMyFeesEstimates), so the repricer never has to back fees out of Seller Snap's ROI.

Amazon's FBA fee steps at the price bands under $10 / $10-$50 / over $50, and the referral rate steps at $10 (8% at or below, 15% above, for
beauty). So for each sellable SKU we ask for the fee at one price in each band and keep, per band: the referral rate, the FBA fee and any
closing fee. fee_at(price) then gives the exact fee at any candidate price.
  - sampled prices: $9.99, the SKU's listed price clipped to $10.01-$49.99, and the listed price if over $50 (else $59.99)
  - 20 estimates per call, one call every 2.5 s (Amazon allows 0.5 calls/s); about 2 min for 370 SKUs
  - refreshed when a SKU's fees are over a day old, and for new sellable SKUs within the hour
  - every response saved in full to fees/<date>.jsonl.gz; the latest per SKU is reloaded from disk at start
"""
import datetime as dt, gzip, json, os, time

import requests

SP = 'https://sellingpartnerapi-na.amazon.com'
MP = 'ATVPDKIKX0DER'
MAX_AGE_S = 24 * 3600
GAP_S = 2.5


def band_prices(listed):
    lp = float(listed or 25.0)
    mid = min(max(lp, 10.01), 49.99)
    hi = lp if lp > 50 else 59.99
    return {'lo': 9.99, 'mid': round(mid, 2), 'hi': round(hi, 2)}


def parse(est):
    """One FeesEstimateResult -> (sku, band, {price, ref, fba, close, total}) or None."""
    ident = est.get('FeesEstimateIdentifier') or {}
    sku, _, band = (ident.get('SellerInputIdentifier') or '').rpartition('|')
    if est.get('Status') != 'Success' or not sku:
        return None
    price = ((ident.get('PriceToEstimateFees') or {}).get('ListingPrice') or {}).get('Amount')
    fe = est.get('FeesEstimate') or {}
    d = {}
    for f in fe.get('FeeDetailList') or []:
        d[f.get('FeeType')] = ((f.get('FinalFee') or f.get('FeeAmount') or {}).get('Amount') or 0.0)
    return sku, band, {'price': price, 'ref': d.get('ReferralFee', 0.0), 'fba': d.get('FBAFees', 0.0),
                       'close': d.get('VariableClosingFee', 0.0) + d.get('PerItemFee', 0.0),
                       'total': (fe.get('TotalFeesEstimate') or {}).get('Amount')}


def fee_at(bands, p):
    """Exact Amazon fee at price p from the three sampled bands, or None if the band needed is missing.
    Referral: the rate of the band at that price (8%/15% step at $10, minimum $0.30). FBA: <$10 / $10-$50 / >$50."""
    if not bands:
        return None
    rb = 'lo' if p <= 10 else ('mid' if p <= 50 else 'hi')
    fb = 'lo' if p < 10 else ('mid' if p <= 50 else 'hi')
    r, f = bands.get(rb), bands.get(fb)
    if not r or not f or not r.get('price'):
        return None
    rate = round(r['ref'] / r['price'] * 200) / 200.0   # Amazon rounds the sampled fee to the cent; referral rates are whole or half percents
    ref = max(rate * p, 0.30) if r['ref'] > 0 else 0.0
    return ref + f['fba'] + f['close'], rate


class Fees:
    def __init__(self, data_dir, append_gz, iso, sp_token, on_fees=None, wait=time.sleep):
        self.data, self.append_gz, self.iso, self.sp_token, self.on_fees, self.wait = data_dir, append_gz, iso, sp_token, on_fees, wait
        self.by_sku = {}     # sku -> {'t': epoch, 'bands': {'lo': {...}, 'mid': {...}, 'hi': {...}}}
        self.stats = {'last': None, 'skus': 0, 'calls': 0, 'throttled': 0, 'failed': 0, 'last_error': None}

    def load(self, days=3):
        """Latest fees per SKU from the last few days on disk, so a restart doesn't re-pull everything."""
        for i in range(days - 1, -1, -1):
            d = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=i)).strftime('%Y-%m-%d')
            p = os.path.join(self.data, 'fees', d + '.jsonl.gz')
            if not os.path.exists(p):
                continue
            try:
                with gzip.open(p, 'rt') as f:
                    for line in f:
                        try:
                            r = json.loads(line)
                        except ValueError:
                            continue
                        self._keep(r)
            except (OSError, EOFError):
                continue
        self.stats['skus'] = len(self.by_sku)
        if self.on_fees:
            self.on_fees(dict(self.by_sku))

    def _keep(self, r):
        x = parse(r.get('result') or {})
        if not x:
            return
        sku, band, v = x
        t = r.get('t_epoch') or time.time()
        cur = self.by_sku.setdefault(sku, {'t': t, 'bands': {}})
        cur['bands'][band] = v
        cur['t'] = max(cur['t'], t)

    def due(self, sellable, now=None):
        now = now or time.time()
        return [(s, lp) for s, lp in sellable if now - (self.by_sku.get(s) or {}).get('t', 0) > MAX_AGE_S or len((self.by_sku.get(s) or {}).get('bands', {})) < 3]

    def call(self, reqs, token):
        body = [{'FeesEstimateRequest': {'MarketplaceId': MP, 'IsAmazonFulfilled': True, 'Identifier': f'{sku}|{band}',
                                         'PriceToEstimateFees': {'ListingPrice': {'CurrencyCode': 'USD', 'Amount': price},
                                                                 'Shipping': {'CurrencyCode': 'USD', 'Amount': 0}}},
                 'IdType': 'SellerSKU', 'IdValue': sku} for sku, band, price in reqs]
        for tries in range(6):
            r = requests.post(f'{SP}/products/fees/v0/feesEstimate', headers={'x-amz-access-token': token, 'content-type': 'application/json'},
                              json=body, timeout=60)
            if r.status_code != 429:
                break
            self.stats['throttled'] += 1
            self.wait(min(60, GAP_S * 2 ** (tries + 1)))
        r.raise_for_status()
        return r.json()

    def pull(self, sellable):
        """sellable: [(sku, listed_price)]. Pulls the SKUs that are due. Returns how many SKUs were refreshed."""
        todo = self.due(sellable)
        if not todo:
            return 0
        reqs = [(s, b, p) for s, lp in todo for b, p in band_prices(lp).items()]
        token, t0, done = self.sp_token(), time.time(), set()
        for i in range(0, len(reqs), 20):
            if time.time() - t0 > 2700:
                token, t0 = self.sp_token(), time.time()
            chunk = reqs[i:i + 20]
            try:
                res = self.call(chunk, token)
            except requests.RequestException as e:
                self.stats['failed'] += 1
                self.stats['last_error'] = f'{self.iso()} {type(e).__name__}: {str(e)[:200]}'
                self.wait(GAP_S)
                continue
            self.stats['calls'] += 1
            t = time.time()
            now = dt.datetime.now(dt.timezone.utc)
            lines = []
            for est in res if isinstance(res, list) else []:
                rec = {'taken_at': self.iso(), 't_epoch': t, 'result': est}
                lines.append(json.dumps(rec))
                self._keep(rec)
                x = parse(est)
                if x:
                    done.add(x[0])
            if lines:
                self.append_gz(os.path.join(self.data, 'fees', now.strftime('%Y-%m-%d') + '.jsonl.gz'), lines)
            self.wait(GAP_S)
        self.stats.update(last=self.iso(), skus=len(self.by_sku))
        if self.on_fees:
            self.on_fees(dict(self.by_sku))
        return len(done)
