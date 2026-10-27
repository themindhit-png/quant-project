#!/usr/bin/env python3
# Review 4.1 (run from repo root, log: logs/phase4_check_oldF2.log)
# F2 lag0 on the ORIGINAL predictions/grid (END 2026-08-31, _p0/_noage) -> daily returns, to cross-check phase4 base
import os, sys; sys.path.insert(0, os.getcwd())
for k in ('END','PRED_SUF8','PRED_SUF37'): os.environ.pop(k, None)
import pandas as pd, exp_common as X
from bt import run
X.d.start_i = int(X.d.idx.searchsorted(pd.Timestamp(X.OOS, tz='UTC')))
for lag in (0, 1):
    f0 = X.F2(); fn = f0 if lag == 0 else (lambda dd, i, m, f0=f0: f0(dd, i-1, m))
    a = dict(X.BASE8); a['reb_offset'] = lag; a['label'] = f'old F2 lag{lag}'
    m, eq, hp = run(X.d, fn, ret_diag=True, **a)
    de = eq.groupby(eq.index.floor('D')).last(); r = de.pct_change().dropna(); r.index = r.index.tz_localize(None)
    r.to_csv(f'research/phase4_out/check_oldpred_F2_lag{lag}.csv'); print(lag, m['sharpe'], m['cagr'], m['mdd'], flush=True)
