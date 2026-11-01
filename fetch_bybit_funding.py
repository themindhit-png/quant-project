#!/usr/bin/env python3
"""Fetch Bybit USDT-perp funding history (public endpoint, no key) for many symbols and save one CSV.
Run on the VPS (Bybit API is geo-blocked from the development sandbox):
    python3 fetch_bybit_funding.py --months 24 --out data/bybit_funding.csv [--symbols BTCUSDT,ETHUSDT,...]
    python3 fetch_bybit_funding.py --since 2020-01-01 --symbols-file research/vps_symbols.txt --add-listed --out research/vps_dump/funding_bybit_full.csv.gz
      (full history, reg. 5 step R-1: research symbols incl. delisted ones + everything listed now; ~1-2 h; ops/funding_history.sh wraps it)
Without --symbols: every linear USDT perpetual currently listed (from /v5/market/instruments-info). Output columns:
symbol, ts_ms, rate (funding rate actually settled at ts_ms). ~6 requests per symbol per year of 8h funding; paced at 5 req/s."""
import argparse, csv, gzip, io, sys, time, requests
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
    ap.add_argument('--since', default=''); ap.add_argument('--symbols-file', default=''); ap.add_argument('--add-listed', action='store_true')
    a = ap.parse_args()
    syms = [s.strip().upper() for s in a.symbols.split(',') if s.strip()]
    if a.symbols_file:
        syms += [x.strip().upper() for x in open(a.symbols_file) if x.strip()]
    if a.add_listed or not syms:
        syms += symbols_all()
    syms = sorted(set(syms))
    now = int(time.time() * 1000)
    start = int(time.mktime(time.strptime(a.since, '%Y-%m-%d')) * 1000) if a.since else now - a.months * 30 * 24 * 3600 * 1000
    print(f'{len(syms)} symbols, from {time.strftime("%Y-%m-%d", time.gmtime(start / 1000))}', flush=True)
    opener = (lambda p: io.TextIOWrapper(gzip.open(p, 'wb'), newline='')) if a.out.endswith('.gz') else (lambda p: open(p, 'w', newline=''))
    with opener(a.out) as f:
        w = csv.writer(f); w.writerow(['symbol', 'ts_ms', 'rate']); n = 0
        for k, s in enumerate(syms):
            rows = funding(s, start, now); n += len(rows)
            for r in rows:
                w.writerow(r)
            if k % 25 == 0:
                print(f'  {k}/{len(syms)} {s}: {len(rows)} rows (total {n})', flush=True)
            time.sleep(0.2)
    print(f'DONE -> {a.out} ({n} rows)')
