#!/usr/bin/env python3
"""Fetch Bybit USDT-perp funding history (public endpoint, no key) for many symbols and save one CSV.
Run on the VPS (Bybit API is geo-blocked from the development sandbox):
    python3 fetch_bybit_funding.py --months 24 --out data/bybit_funding.csv [--symbols BTCUSDT,ETHUSDT,...]
Without --symbols: every linear USDT perpetual currently listed (from /v5/market/instruments-info). Output columns:
symbol, ts_ms, rate (funding rate actually settled at ts_ms). ~6 requests per symbol per year of 8h funding; paced at 5 req/s."""
import argparse, csv, sys, time, requests
BASE = 'https://api.bybit.com'
S = requests.Session()


def get(path, params, retries=4):
    for att in range(retries):
        try:
            r = S.get(BASE + path, params=params, timeout=20)
            j = r.json()
            if j.get('retCode') == 0:
                return j['result']
            time.sleep(1 + att)
        except Exception:
            time.sleep(1 + att)
    return {}


def symbols_all():
    out, cursor = [], ''
    while True:
        p = {'category': 'linear', 'limit': 1000}
        if cursor:
            p['cursor'] = cursor
        r = get('/v5/market/instruments-info', p)
        for it in r.get('list', []):
            if it.get('quoteCoin') == 'USDT' and it.get('contractType') == 'LinearPerpetual' and it.get('status') == 'Trading':
                out.append(it['symbol'])
        cursor = r.get('nextPageCursor', '')
        if not cursor:
            return sorted(set(out))


def funding(symbol, start_ms, end_ms):
    rows, end = [], end_ms
    while end > start_ms:
        r = get('/v5/market/funding/history', {'category': 'linear', 'symbol': symbol, 'startTime': start_ms, 'endTime': end, 'limit': 200})
        lst = r.get('list', [])
        if not lst:
            break
        for it in lst:
            rows.append((symbol, int(it['fundingRateTimestamp']), float(it['fundingRate'])))
        oldest = min(int(it['fundingRateTimestamp']) for it in lst)
        if oldest <= start_ms or len(lst) < 200:
            break
        end = oldest - 1
        time.sleep(0.2)
    return rows


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--months', type=int, default=24); ap.add_argument('--out', default='data/bybit_funding.csv'); ap.add_argument('--symbols', default='')
    a = ap.parse_args()
    syms = [s.strip().upper() for s in a.symbols.split(',') if s.strip()] or symbols_all()
    now = int(time.time() * 1000); start = now - a.months * 30 * 24 * 3600 * 1000
    print(f'{len(syms)} symbols, {a.months} months', flush=True)
    with open(a.out, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['symbol', 'ts_ms', 'rate']); n = 0
        for k, s in enumerate(syms):
            rows = funding(s, start, now); n += len(rows)
            for r in rows:
                w.writerow(r)
            if k % 25 == 0:
                print(f'  {k}/{len(syms)} {s}: {len(rows)} rows (total {n})', flush=True)
            time.sleep(0.2)
    print(f'DONE -> {a.out} ({n} rows)')
