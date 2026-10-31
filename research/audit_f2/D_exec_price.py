"""D_ audit: execution-price sensitivity of F2 without changing bt.py. A patched COPY of bt.run (exec'd here) charges every
rebalance trade the move from close[i] to an execution price inside the next bar: EXEC=close (baseline, no change),
vwap (bar i+1 VWAP = qvol/vol, ~ +30 min average), c1 (close of bar i+1, ~ 1h lag of the trade only)."""
import os, sys, inspect, numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, ROOT); os.chdir(ROOT)
import bt
if os.environ.get('PAN'): bt.PAN = os.environ['PAN']
import exp_common as X
from exp_common import d, N, F2, BASE8, Held, sl_listing, sl_core, ML3n, ML7n, ML8, fund_change, fund_level
src = inspect.getsource(bt.run)
old = "                pos = newpos\n"
assert src.count(old) == 1
src = src.replace(old, "                pos = newpos\n                if EXECPX is not None:\n                    xp = EXECPX[i + 1]; adj = float(np.nansum(np.where(np.isfinite(xp) & (xp > 0), dpos * (xp / px - 1.0), 0.0)))\n                    eq -= adj; EXADJ[0] += adj\n")
ns = dict(bt.__dict__); exec(src, ns); run2 = ns['run']
vol = pd.read_parquet(os.path.join(bt.PAN, 'vol.parquet'), columns=list(d.cols)).loc[d.idx[0]:d.idx[-1]].values.astype(np.float64)
qv = pd.read_parquet(os.path.join(bt.PAN, 'qvol.parquet'), columns=list(d.cols)).loc[d.idx[0]:d.idx[-1]].values.astype(np.float64)
VW = np.where(vol > 0, qv / vol, np.nan); del vol, qv
C1 = d.cff.copy()
SLEEVES = dict(listing=Held(sl_listing), core=Held(sl_core), ml3=Held(ML3n), ml7=Held(ML7n), ml8=ML8, fchg=Held(fund_change), fcarry=fund_level)
for book in os.environ.get('BOOKS', 'F2').split(','):
    for ex in os.environ.get('EXEC', 'close,vwap,c1').split(','):
        ns['EXECPX'] = {'close': None, 'vwap': VW, 'c1': C1}[ex]; ns['EXADJ'] = [0.0]
        fn = F2() if book == 'F2' else SLEEVES[book]
        d.start_i = int(d.idx.searchsorted(pd.Timestamp(X.OOS, tz='UTC')))
        m, eq = run2(d, fn, **dict(BASE8, label=book))
        de = eq.groupby(eq.index.floor('D')).last(); r = de.pct_change().dropna()
        print(f'{book:<8} exec={ex:<6} Sh {m["sharpe"]:5.2f} SEL {X.sh(r[:"2024-12-31"]):5.2f} CONF {X.sh(r["2025-01-01":]):5.2f} CAGR {m["cagr"]*100:5.1f}% MDD {m["mdd"]*100:5.1f}% '
              f'exec-adj ${ns["EXADJ"][0]:8.0f} | ' + ' '.join(f'{y}:{v[1]:.2f}' for y, v in m['by_year'].items()), flush=True)
