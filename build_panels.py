#!/usr/bin/env python3
"""Build wide hourly panels from Binance Vision raw monthly zips.
Outputs data/panels/*.parquet (dir: env PANELS_OUT):
  close (f64), open/high/low (f32), qvol (quote volume USDT, f32), vol (base, f32),
  ntrades (f32), tbq (taker buy quote vol, f32), funding (f32, rate at the hour it is paid, 0 elsewhere)
  meta.json: first/last valid bar per symbol
Run: python3 build_panels.py "<END>" core|ohlc|premium|freeze   (freeze last, after the other three)

PREREG reg. 4 (audit 2026-10-31):
  * holes of truncated monthly archives are filled from daily archives (data/raw/daily, `python3 dl_binance.py gaps`);
    a daily row is used only for an hour that no monthly file has;
  * mode `freeze`: a run of >= FREEZE_H (24) bars with zero quote volume and an unchanged close (Binance Vision keeps publishing
    a frozen price after a delisting) is NOT a price -> all price/volume/premium panels set to NaN there (funding untouched).
    The research engine then sees a gap > 24 h: delisting (force close + shock) and a NEW listing segment if the ticker trades
    again (re-used ticker = new coin, its age restarts).
"""
import os, io, sys, json, zipfile, glob, time
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(ROOT, 'data', 'raw')
OUT = os.environ.get('PANELS_OUT') or os.path.join(ROOT, 'data', 'panels')
DAILY = os.path.join(RAW, 'daily')
FREEZE_H = 24
os.makedirs(OUT, exist_ok=True)
START = pd.Timestamp('2020-01-01', tz='UTC')
END = pd.Timestamp(sys.argv[1] if len(sys.argv) > 1 else '2026-08-31 23:00', tz='UTC')
IDX = pd.date_range(START, END, freq='h')
H = len(IDX)
KCOLS = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_volume',
         'count', 'taker_buy_volume', 'taker_buy_quote_volume', 'ignore']


def read_zip_csv(path, names):
    with zipfile.ZipFile(path) as z:
        n = z.namelist()[0]
        raw = z.read(n)
    # some files have header row, some don't
    first = raw[:200].split(b'\n')[0]
    skip = 1 if (b'open_time' in first or b'calc_time' in first) else 0
    return pd.read_csv(io.BytesIO(raw), header=None, names=names, skiprows=skip)


def _read_klines(files):
    parts = []
    for f in files:
        try:
            parts.append(read_zip_csv(f, KCOLS))
        except Exception as e:
            print(f'  bad zip {f}: {e}')
    if not parts:
        return None
    df = pd.concat(parts, ignore_index=True)
    df = df[pd.to_numeric(df['open_time'], errors='coerce').notna()]
    ot = df['open_time'].astype(np.int64)
    # Binance switched some archives to microseconds in 2025
    ot = np.where(ot > 1e14, ot // 1000, ot)
    df.index = pd.to_datetime(ot, unit='ms', utc=True)
    return df[~df.index.duplicated(keep='last')].sort_index()


def load_kline_kind(kind, sym):
    """Monthly archives, holes filled from daily archives (only hours absent from every monthly file)."""
    df = _read_klines(sorted(glob.glob(os.path.join(RAW, kind, sym, '1h', '*.zip'))))
    dd = _read_klines(sorted(glob.glob(os.path.join(DAILY, kind, sym, '1h', '*.zip'))))
    if dd is not None and len(dd):
        dd = dd[~dd.index.isin(df.index)] if df is not None else dd
        if len(dd):
            df = dd if df is None else pd.concat([df, dd]).sort_index()
    return df


def load_klines(sym):
    df = load_kline_kind('klines', sym)
    if df is None:
        return None
    df = df[(df.index >= START) & (df.index <= END)]
    for c in ['open', 'high', 'low', 'close', 'volume', 'quote_volume', 'count',
              'taker_buy_quote_volume']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    return df


def load_funding(sym):
    files = sorted(glob.glob(os.path.join(RAW, 'fundingRate', sym, '*.zip')))
    if not files:
        return None
    parts = []
    for f in files:
        try:
            parts.append(read_zip_csv(f, ['calc_time', 'interval_h', 'rate']))
        except Exception as e:
            print(f'  bad zip {f}: {e}')
    if not parts:
        return None
    df = pd.concat(parts, ignore_index=True)
    df = df[pd.to_numeric(df['calc_time'], errors='coerce').notna()]
    ct = df['calc_time'].astype(np.int64)
    ct = np.where(ct > 1e14, ct // 1000, ct)
    ts = pd.to_datetime(ct, unit='ms', utc=True).floor('h')
    s = pd.Series(pd.to_numeric(df['rate'], errors='coerce').values, index=ts)
    s = s.groupby(level=0).sum(min_count=1)
    return s[(s.index >= START) & (s.index <= END)]


def main():
    syms = sorted(d for d in os.listdir(os.path.join(RAW, 'klines')) if d.endswith('USDT'))
    print(f'{len(syms)} symbols, grid {H} hours {IDX[0]} -> {IDX[-1]}', flush=True)
    mode = sys.argv[2] if len(sys.argv) > 2 else 'core'
    if mode == 'freeze':
        return freeze()
    if mode == 'premium':
        # premium index klines: same layout under raw/premiumIndexKlines/<SYM>/1h/; store close as f32
        rawp = os.path.join(ROOT, 'data', 'raw', 'premiumIndexKlines')
        syms = sorted(d for d in os.listdir(rawp) if d.endswith('USDT'))
        arr = np.full((H, len(syms)), np.nan, dtype=np.float32)
        keep = []
        for j, sym in enumerate(syms):
            df = load_kline_kind('premiumIndexKlines', sym)
            if df is None:
                continue
            pos = IDX.get_indexer(df.index); ok = pos >= 0
            arr[pos[ok], j] = pd.to_numeric(df['close'], errors='coerce').values[ok]
            keep.append(j)
            if j % 100 == 0:
                print(f'  {j}/{len(syms)} {sym}', flush=True)
        cols = [syms[j] for j in keep]
        pd.DataFrame(arr[:, keep], index=IDX, columns=cols).to_parquet(os.path.join(OUT, 'premium.parquet'))
        print(f'DONE premium: {len(cols)} symbols', flush=True)
        return
    kinds = ['close', 'qvol'] if mode == 'core' else ['open', 'high', 'low', 'vol', 'ntrades', 'tbq']
    panels = {k: np.full((H, len(syms)), np.nan, dtype=(np.float64 if k == 'close' else np.float32))
              for k in kinds}
    fund = np.zeros((H, len(syms)), dtype=np.float32) if mode == 'core' else None
    meta = {}
    t0 = time.time()
    keep = []
    for j, sym in enumerate(syms):
        df = load_klines(sym)
        if df is None or len(df) < 24:
            continue
        pos = IDX.get_indexer(df.index)
        ok = pos >= 0
        pos = pos[ok]
        df = df.iloc[np.flatnonzero(ok)]
        src = {'close': 'close', 'open': 'open', 'high': 'high', 'low': 'low', 'qvol': 'quote_volume',
               'vol': 'volume', 'ntrades': 'count', 'tbq': 'taker_buy_quote_volume'}
        for k in kinds:
            panels[k][pos, j] = df[src[k]].values
        nf = 0
        if fund is not None:
            fs = load_funding(sym)
            if fs is not None and len(fs):
                fp = IDX.get_indexer(fs.index)
                okf = fp >= 0
                fund[fp[okf], j] = fs.values[okf]
                nf = int(okf.sum())
        meta[sym] = {'first': str(df.index[0]), 'last': str(df.index[-1]), 'bars': int(len(df)),
                     'n_funding': nf}
        keep.append(j)
        if j % 50 == 0:
            print(f'  {j}/{len(syms)} {sym} ({time.time()-t0:.0f}s)', flush=True)
    cols = [syms[j] for j in keep]
    for k in list(panels):
        arr = panels.pop(k)
        pd.DataFrame(arr[:, keep], index=IDX, columns=cols).to_parquet(os.path.join(OUT, f'{k}.parquet'))
        del arr
    if fund is not None:
        pd.DataFrame(fund[:, keep], index=IDX, columns=cols).to_parquet(os.path.join(OUT, 'funding.parquet'))
        json.dump(meta, open(os.path.join(OUT, 'meta.json'), 'w'), indent=0)
    print(f'DONE: {len(cols)} symbols written in {time.time()-t0:.0f}s', flush=True)


def freeze_mask(close, qvol, min_h=FREEZE_H):
    """(T, N) bool: bars inside a run of >= min_h consecutive bars with valid close, quote volume 0 and close == previous
    bar's close (a frozen post-delisting price, not a traded one)."""
    c = np.asarray(close, dtype=np.float64); q = np.asarray(qvol, dtype=np.float64)
    prev = np.vstack([np.full((1, c.shape[1]), np.nan), c[:-1]])
    flat = np.isfinite(c) & (c > 0) & (q == 0) & (c == prev)
    out = np.zeros_like(flat)
    for j in np.flatnonzero(flat.sum(0) >= min_h):
        x = np.concatenate(([0], flat[:, j].astype(np.int8), [0])); dx = np.diff(x)
        for a, b in zip(np.flatnonzero(dx == 1), np.flatnonzero(dx == -1)):
            if b - a >= min_h:
                out[a:b, j] = True
    return out


def freeze():
    t0 = time.time()
    close = pd.read_parquet(os.path.join(OUT, 'close.parquet')); qvol = pd.read_parquet(os.path.join(OUT, 'qvol.parquet'))
    assert list(close.columns) == list(qvol.columns) and close.index.equals(qvol.index)
    fm = freeze_mask(close.values, qvol.values)
    n_sym = int((fm.sum(0) > 0).sum())
    print(f'freeze: {int(fm.sum())} bars in {n_sym} symbols set to NaN (runs >= {FREEZE_H} h of zero volume at a constant price)', flush=True)
    for k in ('close', 'open', 'high', 'low', 'qvol', 'vol', 'ntrades', 'tbq', 'premium'):
        p = os.path.join(OUT, f'{k}.parquet')
        if not os.path.exists(p):
            continue
        df = close if k == 'close' else qvol if k == 'qvol' else pd.read_parquet(p)
        m = pd.DataFrame(fm, index=close.index, columns=close.columns).reindex(index=df.index, columns=df.columns, fill_value=False).values
        arr = df.values.copy(); arr[m] = np.nan
        pd.DataFrame(arr, index=df.index, columns=df.columns).to_parquet(p)
    mp = os.path.join(OUT, 'meta.json')
    meta = json.load(open(mp)) if os.path.exists(mp) else {}
    cv = close.values.copy(); cv[fm] = np.nan; ok = np.isfinite(cv) & (cv > 0)
    for j, sym in enumerate(close.columns):
        ix = np.flatnonzero(ok[:, j])
        if len(ix):
            meta.setdefault(sym, {}).update(first=str(close.index[ix[0]]), last=str(close.index[ix[-1]]), bars=int(len(ix)),
                                            frozen_bars=int(fm[:, j].sum()))
    json.dump(meta, open(mp, 'w'), indent=0)
    print(f'DONE freeze in {time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
