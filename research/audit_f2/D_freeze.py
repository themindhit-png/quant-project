"""D_ audit: delistings that appear as a frozen-price zero-volume TAIL (no close_event -> no delist_shock).
Logs the F2 book's |position|/equity in each such symbol at the first rebalance after volume dies."""
import os, sys, numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, ROOT); os.chdir(ROOT)
import exp_common as X
from exp_common import d, N, F2, BASE8
from bt import run
q = pd.read_parquet('data/panels/qvol.parquet', columns=list(d.cols)).loc[d.idx[0]:d.idx[-1]].values
T = len(d.idx); frz = np.full(N, -1)
for j in range(N):
    z = d.valid[:, j] & (np.nan_to_num(q[:, j]) == 0)
    if not z[-1] and not z[-169:-1].all():
        # tail must be zero-volume until the end of the grid (or for >= 30 days)
        pass
    dz = np.diff(np.concatenate(([0], z.astype(np.int8), [0]))); st = np.flatnonzero(dz == 1); en = np.flatnonzero(dz == -1)
    long_ = [(a, b) for a, b in zip(st, en) if b - a >= 24 * 30]
    if long_: frz[j] = long_[0][0]
print('symbols with >=30d frozen zero-volume run:', int((frz >= 0).sum()))
rec = {}
f0 = F2()
def fn(dd, i, mask):
    p = dd.cur_pos; e = dd.cur_eq
    for j in np.flatnonzero((frz >= 0) & (frz <= i) & (frz > i - 48)):
        if j not in rec: rec[j] = (str(dd.idx[i]), float(p[j] / e))
    return f0(dd, i, mask)
d.start_i = int(d.idx.searchsorted(pd.Timestamp(X.OOS, tz='UTC')))
m, eq = run(d, fn, **dict(BASE8, label='freeze'))
r = pd.DataFrame([(d.cols[j], a, b) for j, (a, b) in rec.items()], columns=['sym', 'at', 'pos_over_eq'])
r = r[r['at'] >= '2022-01-01']
print(r[r.pos_over_eq != 0].sort_values('at').to_string())
print(f'n events in OOS {len(r)}, with position {int((r.pos_over_eq != 0).sum())}, sum|pos|/eq {r.pos_over_eq.abs().sum():.4f}, '
      f'5% shock cost = {0.05 * r.pos_over_eq.abs().sum() * 100:.3f}% of equity in total; net short share {(r.pos_over_eq < 0).mean():.2f}')
