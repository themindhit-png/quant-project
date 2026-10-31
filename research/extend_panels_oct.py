#!/usr/bin/env python3
"""Дописать панели Binance (data/panels) днями с 2026-10-01 по 2026-10-26 из ДНЕВНЫХ архивов data.binance.vision
(klines, premiumIndexKlines). Фандинг за октябрь на Vision отсутствует -> берётся Bybit из research/vps_dump (прокси, corr ~0.88).
Оригиналы сохраняются в data/panels_orig. Запуск из корня: python3 research/extend_panels_oct.py
Рег. 4: после дописывания — python3 build_panels.py "2026-10-26 23:00" freeze (замороженные цены -> NaN, и в октябре)."""
import os, io, zipfile, shutil, gzip, concurrent.futures as cf
import numpy as np, pandas as pd, requests
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAN = os.environ.get('PANELS_OUT') or os.path.join(ROOT, 'data', 'panels'); ORIG = os.environ.get('PANELS_BASE') or os.path.join(ROOT, 'data', 'panels_orig'); TMP = os.path.join(ROOT, 'data', 'raw_oct')
DAYS = pd.date_range('2026-10-01', '2026-10-26', freq='D')
NEW = pd.date_range('2026-10-01 00:00', '2026-10-26 23:00', freq='h', tz='UTC')
if not os.path.exists(ORIG):
    shutil.copytree(PAN, ORIG)
close0 = pd.read_parquet(os.path.join(ORIG, 'close.parquet'))
alive = [c for c in close0.columns if close0[c].iloc[-24 * 30:].notna().any()]
print(len(alive), 'живых символов', flush=True)
sess = requests.Session(); sess.headers['User-Agent'] = 'research'
def get(kind, sym, day):
    d = day.strftime('%Y-%m-%d'); dest = os.path.join(TMP, kind, sym, f'{d}.zip')
    if os.path.exists(dest): return dest
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    url = f'https://data.binance.vision/data/futures/um/daily/{kind}/{sym}/1h/{sym}-1h-{d}.zip'
    for a in range(4):
        try:
            r = sess.get(url, timeout=60)
            if r.status_code == 200 and r.content[:2] == b'PK':
                open(dest, 'wb').write(r.content); return dest
            if r.status_code == 404: return None
        except Exception: pass
    return None
jobs = [(k, s, d) for k in ('klines', 'premiumIndexKlines') for s in alive for d in DAYS]
with cf.ThreadPoolExecutor(24) as ex:
    list(ex.map(lambda a: get(*a), jobs))
print('скачано', flush=True)
KC = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_volume', 'count', 'tbv', 'tbq', 'ignore']
def read(path):
    raw = zipfile.ZipFile(path).read(zipfile.ZipFile(path).namelist()[0]); skip = 1 if b'open_time' in raw[:100] else 0
    df = pd.read_csv(io.BytesIO(raw), header=None, names=KC, skiprows=skip); ot = df.open_time.astype('int64'); ot = np.where(ot > 1e14, ot // 1000, ot)
    df.index = pd.to_datetime(ot, unit='ms', utc=True); return df
names = dict(close='close', open='open', high='high', low='low', qvol='quote_volume', vol='volume', ntrades='count', tbq='tbq')
new = {k: pd.DataFrame(np.nan, index=NEW, columns=alive) for k in names}; newp = pd.DataFrame(np.nan, index=NEW, columns=alive)
for s in alive:
    parts = [read(p) for d in DAYS if (p := os.path.join(TMP, 'klines', s, d.strftime('%Y-%m-%d') + '.zip')) and os.path.exists(p)]
    if parts:
        df = pd.concat(parts); df = df[~df.index.duplicated()].reindex(NEW)
        for k, c in names.items(): new[k][s] = pd.to_numeric(df[c], errors='coerce')
    parts = [read(p) for d in DAYS if (p := os.path.join(TMP, 'premiumIndexKlines', s, d.strftime('%Y-%m-%d') + '.zip')) and os.path.exists(p)]
    if parts:
        df = pd.concat(parts); df = df[~df.index.duplicated()].reindex(NEW); newp[s] = pd.to_numeric(df['close'], errors='coerce')
fu = pd.read_csv(os.path.join(ROOT, 'research/vps_dump/funding_bybit.csv.gz'))
fu['t'] = pd.to_datetime(fu.ts_ms, unit='ms', utc=True).dt.floor('h')
nf = fu.pivot_table(index='t', columns='symbol', values='rate', aggfunc='sum').reindex(index=NEW, columns=alive).fillna(0.0)
for k in list(names) + ['funding', 'premium']:
    old = pd.read_parquet(os.path.join(ORIG, f'{k}.parquet'))
    add = nf if k == 'funding' else newp if k == 'premium' else new[k]
    add = add.reindex(columns=old.columns).astype(old.dtypes.iloc[0] if k != 'funding' else 'float32')
    if k == 'funding': add = add.fillna(0.0)
    out = pd.concat([old[old.index < NEW[0]], add]); out.to_parquet(os.path.join(PAN, f'{k}.parquet'))
    print(k, out.shape, out.index[-1], 'заполнено в окт: %.2f' % add.notna().mean().mean(), flush=True)
print('DONE')
