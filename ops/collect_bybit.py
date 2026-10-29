#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Hourly Bybit data collector (timer propsleeves-collect, :35 every hour, Nice 15; Phase 5.1). Public endpoints only, no keys.
For the top COLLECT_TOP USDT perps by 24h turnover (+ BTC, ETH) appends closed hourly klines (OHLCV + turnover), premium-index klines,
funding settlements and open interest (1h) to /opt/propsleeves/data/bybit/<kind>/<YYYY-MM>.csv.gz (gzip members appended, no duplicates:
the last stored timestamp per kind/symbol is kept in state.json; a new symbol / a gap is back-filled up to BACKFILL_D days).
3 failed runs in a row -> Telegram. Data for later research (Bybit-native features, order flow comes from public.bybit.com archives)."""
import os, sys, json, gzip, time
import requests
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import opslib as L

BASE = os.environ.get('COLLECT_BASE', 'https://api.bybit.com')
TOP = int(os.environ.get('COLLECT_TOP', '300'))
BACKFILL_D = int(os.environ.get('COLLECT_BACKFILL_D', '7'))
DATA = os.environ.get('COLLECT_DIR', os.path.join(L.APP, 'data', 'bybit'))
H = 3600 * 1000
KINDS = {  # kind: (path, params builder, row parser, time key in the result rows, page limit)
    'kline': ('/v5/market/kline', lambda s, t: {'category': 'linear', 'symbol': s, 'interval': '60', 'start': t, 'limit': 1000},
              lambda x: [int(x[0]), x[1], x[2], x[3], x[4], x[5], x[6]], 'ts,open,high,low,close,volume,turnover'),
    'premium': ('/v5/market/premium-index-price-kline', lambda s, t: {'category': 'linear', 'symbol': s, 'interval': '60', 'start': t, 'limit': 1000},
                lambda x: [int(x[0]), x[1], x[2], x[3], x[4]], 'ts,open,high,low,close'),
    'funding': ('/v5/market/funding/history', lambda s, t: {'category': 'linear', 'symbol': s, 'startTime': t, 'endTime': t + 200 * 8 * H, 'limit': 200},
                lambda x: [int(x['fundingRateTimestamp']), x['fundingRate']], 'ts,rate'),
    'oi': ('/v5/market/open-interest', lambda s, t: {'category': 'linear', 'symbol': s, 'intervalTime': '1h', 'startTime': t, 'endTime': t + 200 * H, 'limit': 200},
           lambda x: [int(x['timestamp']), x['openInterest']], 'ts,open_interest'),
}


def get(path, params, tries=4):
    for k in range(tries):
        try:
            r = requests.get(BASE + path, params=params, timeout=20).json()
            if r.get('retCode') == 0:
                return r['result']
            if r.get('retCode') in (10006, 10018):          # rate limit
                time.sleep(2 + 2 * k); continue
            raise RuntimeError(f"{path} {params.get('symbol')}: {r.get('retCode')} {r.get('retMsg')}")
        except requests.RequestException as e:
            if k == tries - 1:
                raise
            time.sleep(1 + k)
    raise RuntimeError(f'{path}: rate limited')


def symbols():
    t = get('/v5/market/tickers', {'category': 'linear'})['list']
    t = [x for x in t if x['symbol'].endswith('USDT') and '-' not in x['symbol']]
    t.sort(key=lambda x: -float(x.get('turnover24h') or 0))
    out = [x['symbol'] for x in t[:TOP]]
    return list(dict.fromkeys(['BTCUSDT', 'ETHUSDT'] + out))


def write(kind, header, rows):
    """rows: [symbol, ts, ...] sorted; one gzip member per (kind, month) per run."""
    by_month = {}
    for r in rows:
        by_month.setdefault(time.strftime('%Y-%m', time.gmtime(r[1] / 1000)), []).append(r)
    for month, rr in by_month.items():
        d = os.path.join(DATA, kind); os.makedirs(d, exist_ok=True); f = os.path.join(d, f'{month}.csv.gz')
        new = not os.path.exists(f)
        with gzip.open(f, 'at') as g:
            if new:
                g.write('symbol,' + header + '\n')
            g.write(''.join(','.join(str(v) for v in r) + '\n' for r in rr))


def collect(state, now_ms):
    closed = now_ms // H * H                                   # only fully closed hours
    n_rows = 0
    for kind, (path, params, parse, header) in KINDS.items():
        rows = []
        for s in symbols_cache:
            last = state.get(kind, {}).get(s)
            start = max(last + 1 if last else 0, closed - BACKFILL_D * 24 * H)
            got = []
            while start < closed:
                res = get(path, params(s, start))
                lst = [parse(x) for x in res.get('list', [])]
                lst = [x for x in lst if start <= x[0] < closed]
                if not lst:
                    break
                got += lst; start = max(x[0] for x in lst) + 1
                if len(res.get('list', [])) < 200:
                    break
                time.sleep(0.05)
            got = sorted({x[0]: x for x in got}.values())
            if got:
                rows += [[s] + x for x in got]; state.setdefault(kind, {})[s] = got[-1][0]
            time.sleep(0.03)
        write(kind, header, rows); n_rows += len(rows)
    return n_rows


def main():
    global symbols_cache
    os.makedirs(DATA, exist_ok=True); sf = os.path.join(DATA, 'state.json')
    state = json.load(open(sf)) if os.path.exists(sf) else {}
    try:
        symbols_cache = symbols()
        t0 = time.time(); n = collect(state, int(time.time() * 1000))
        state['fails'] = 0; state['last_ok'] = L.now_utc().isoformat()
        print(f'collected {n} rows for {len(symbols_cache)} symbols in {time.time() - t0:.0f}s')
    except Exception as e:
        state['fails'] = int(state.get('fails', 0)) + 1
        print(f'collect failed ({state["fails"]} in a row): {e}')
        if state['fails'] == 3:
            L.tg_send(f'⚠️ Сборщик данных Bybit падает 3 часа подряд: {e}')
    tmp = sf + '.tmp'; json.dump(state, open(tmp, 'w')); os.replace(tmp, sf)


symbols_cache = []
if __name__ == '__main__':
    main()
