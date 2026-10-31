"""D_ audit: panel-level checks (funding timing, segment starts/gaps, delistings, stale prices). Read-only."""
import os, sys, json, numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); PAN = os.path.join(ROOT, 'data', 'panels')
close = pd.read_parquet(os.path.join(PAN, 'close.parquet'))
fund = pd.read_parquet(os.path.join(PAN, 'funding.parquet'))
qvol = pd.read_parquet(os.path.join(PAN, 'qvol.parquet'))
idx = close.index; T, N = close.shape
print('panel', T, N, idx[0], idx[-1])
# 1. funding hour distribution
F = fund.values; nz = np.isfinite(F) & (F != 0)
hrs = idx.hour.values
cnt = pd.Series(nz.sum(1), index=hrs).groupby(level=0).sum()
print('nonzero funding count by hour-of-day:'); print(cnt.to_string())
# funding on bars where close invalid
cv = close.values; valid = np.isfinite(cv) & (cv > 0)
print('funding nonzero where close invalid:', int((nz & ~valid).sum()), 'of', int(nz.sum()))
# 2. segments
t24 = qvol.rolling(24, min_periods=6).sum()
rows = []
for j, s in enumerate(close.columns):
    v = np.flatnonzero(valid[:, j])
    if len(v) == 0: continue
    g = np.diff(v); brk = np.flatnonzero(g > 24)
    for b in brk:
        rows.append((s, str(idx[v[b]]), str(idx[v[b + 1]]), int(g[b])))
seg = pd.DataFrame(rows, columns=['sym', 'last_before_gap', 'restart', 'gap_h'])
print('segment restarts (gap>24h):', len(seg)); print(seg.sort_values('restart').to_string())
seg.to_csv(os.path.join(ROOT, 'logs', 'D_segments.csv'), index=False)
# 3. delistings / first bars
first = pd.Series({s: idx[np.flatnonzero(valid[:, j])[0]] for j, s in enumerate(close.columns) if valid[:, j].any()})
last = pd.Series({s: idx[np.flatnonzero(valid[:, j])[-1]] for j, s in enumerate(close.columns) if valid[:, j].any()})
print('symbols with first bar == panel start:', int((first == idx[0]).sum()))
print('symbols ending > 30d before panel end (delisted):', int((last < idx[-1] - pd.Timedelta(days=30)).sum()))
# 4. stale prices: runs of >=48 consecutive valid bars with zero qvol
Q = qvol.values; stale = []
for j, s in enumerate(close.columns):
    z = valid[:, j] & (np.nan_to_num(Q[:, j]) == 0)
    if z.sum() < 48: continue
    # longest run
    d = np.diff(np.concatenate(([0], z.astype(np.int8), [0]))); st = np.flatnonzero(d == 1); en = np.flatnonzero(d == -1)
    L = en - st; k = int(np.argmax(L))
    if L[k] >= 24: stale.append((s, str(idx[st[k]]), int(L[k]), int(z.sum())))
print('symbols with zero-volume valid runs >=24h:', len(stale)); print(pd.DataFrame(stale, columns=['sym', 'start', 'longest_run_h', 'total_zero_h']).sort_values('longest_run_h', ascending=False).head(30).to_string())
# 5. last-bar behaviour of delisted names: return over last 7d before last bar
dl = last[last < idx[-1] - pd.Timedelta(days=30)].index
r7 = []
for s in dl:
    j = close.columns.get_loc(s); v = np.flatnonzero(valid[:, j]); e = v[-1]
    if e - 168 >= v[0]: r7.append(cv[e, j] / cv[e - 168, j] - 1)
r7 = np.array(r7); print(f'delisted names: n={len(r7)} median 7d ret before last bar {np.median(r7):+.3f}, mean {r7.mean():+.3f}')
json.dump({'first': {k: str(v) for k, v in first.items()}, 'last': {k: str(v) for k, v in last.items()}}, open(os.path.join(ROOT, 'logs', 'D_first_last.json'), 'w'))
