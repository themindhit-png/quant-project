#!/usr/bin/env python3
"""1-minute klines (Binance Vision, USDT-M futures, monthly archives) for the grid experiment -> data/m1/<SYM>.parquet (OHLC float32)."""
import os, io, sys, zipfile, requests, pandas as pd
from concurrent.futures import ThreadPoolExecutor
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SYMS = sys.argv[1].split(',') if len(sys.argv) > 1 else ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'DOGEUSDT']
MONTHS = [str(p) for p in pd.period_range('2023-01', '2026-09', freq='M')]
def one(sm):
    s, m = sm
    u = f'https://data.binance.vision/data/futures/um/monthly/klines/{s}/1m/{s}-1m-{m}.zip'
    for _ in range(4):
        try:
            r = requests.get(u, timeout=120)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            z = zipfile.ZipFile(io.BytesIO(r.content)); raw = z.read(z.namelist()[0]).decode()
            df = pd.read_csv(io.StringIO(raw), header=None if not raw[:1].isalpha() else 0, usecols=[0, 1, 2, 3, 4])
            df.columns = ['t', 'o', 'h', 'l', 'c']
            return df
        except Exception as e:
            err = e
    print('fail', u, err); return None
for s in SYMS:
    with ThreadPoolExecutor(8) as ex:
        parts = [p for p in ex.map(one, [(s, m) for m in MONTHS]) if p is not None]
    df = pd.concat(parts); df.index = pd.to_datetime(df.pop('t'), unit='ms', utc=True); df = df[~df.index.duplicated()].sort_index().astype('float32')
    df.to_parquet(os.path.join(ROOT, 'data', 'm1', f'{s}.parquet')); print(s, df.shape, df.index[0], df.index[-1], flush=True)
