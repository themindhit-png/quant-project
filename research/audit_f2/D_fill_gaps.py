"""D_ audit: fill the Binance-Vision monthly-archive holes (2022-02-26..28, 2022-04-01..02) from DAILY archives.
Writes patched close/qvol/funding panels to $OUT (default scratch) — originals untouched. Only gaps listed in logs/D_segments.csv
with restart on 2022-03-01 or 2022-04-03 are filled."""
import os, io, zipfile, sys, numpy as np, pandas as pd, requests, concurrent.futures as cf
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); PAN = os.path.join(ROOT, 'data', 'panels')
OUT = os.environ['OUT']; RAW = os.path.join(OUT, 'raw'); os.makedirs(RAW, exist_ok=True)
seg = pd.read_csv(os.path.join(ROOT, 'logs', 'D_segments.csv'))
seg = seg[seg.restart.str[:10].isin(['2022-03-01', '2022-04-03'])]
jobs = []
for _, r in seg.iterrows():
    for dday in pd.date_range(pd.Timestamp(r.last_before_gap).floor('D') + pd.Timedelta(days=1), pd.Timestamp(r.restart).floor('D') - pd.Timedelta(days=1)):
        jobs.append((r.sym, dday.strftime('%Y-%m-%d')))
print(len(jobs), 'daily files')
s = requests.Session()
def get(a):
    sym, dd = a; dest = os.path.join(RAW, f'{sym}-{dd}.zip')
    if os.path.exists(dest): return dest
    u = f'https://data.binance.vision/data/futures/um/daily/klines/{sym}/1h/{sym}-1h-{dd}.zip'
    for _ in range(4):
        try:
            x = s.get(u, timeout=60)
            if x.status_code == 200 and x.content[:2] == b'PK': open(dest, 'wb').write(x.content); return dest
            if x.status_code == 404: return None
        except Exception: pass
    return None
with cf.ThreadPoolExecutor(16) as ex: paths = list(ex.map(get, jobs))
KC = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_volume', 'count', 'tbv', 'tbq', 'ignore']
close = pd.read_parquet(os.path.join(PAN, 'close.parquet')); qvol = pd.read_parquet(os.path.join(PAN, 'qvol.parquet'))
nfill = 0
for (sym, dd), p in zip(jobs, paths):
    if p is None: print('missing', sym, dd); continue
    raw = zipfile.ZipFile(p).read(zipfile.ZipFile(p).namelist()[0]); skip = 1 if b'open_time' in raw[:100] else 0
    df = pd.read_csv(io.BytesIO(raw), header=None, names=KC, skiprows=skip); ot = df.open_time.astype('int64'); ot = np.where(ot > 1e14, ot // 1000, ot)
    df.index = pd.to_datetime(ot, unit='ms', utc=True)
    ix = df.index.intersection(close.index)
    m = close.loc[ix, sym].isna()
    close.loc[ix[m.values], sym] = df.loc[ix[m.values], 'close'].astype(float).values; qvol.loc[ix[m.values], sym] = df.loc[ix[m.values], 'quote_volume'].astype(np.float32).values
    nfill += int(m.sum())
print('filled bars:', nfill)
close.to_parquet(os.path.join(OUT, 'close.parquet')); qvol.to_parquet(os.path.join(OUT, 'qvol.parquet'))
os.symlink(os.path.join(PAN, 'funding.parquet'), os.path.join(OUT, 'funding.parquet')) if not os.path.exists(os.path.join(OUT, 'funding.parquet')) else None
print('DONE')
