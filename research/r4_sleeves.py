#!/usr/bin/env python3
"""Reg. 4: each F2 sleeve alone through the research engine with the R2 lag-1 configuration (signal on bar i-1, execution at bar i,
BASE8 risk layer, costs x1) -> daily returns research/phase4_out/<OUT_PREFIX>sleeves.csv. Old vs new: run with
PANELS_DIR / PRED_SUF8 / PRED_SUF37 / FUND_TIEBREAK of each version, e.g.
  END="2026-10-26 23:00" PANELS_DIR=<old panels> PRED_SUF8=_oct_noconst PRED_SUF37=_noage_oct FUND_TIEBREAK=0 OUT_PREFIX=R4_old_ python3 research/r4_sleeves.py"""
import os, sys, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import exp_common as X
from exp_common import d, BASE8, Held, sl_listing, sl_core, ML3n, ML7n, ML8, fund_change, fund_level
from bt import run
SL = dict(listing=Held(sl_listing), core=Held(sl_core), ml3=Held(ML3n), ml7=Held(ML7n), ml8=ML8, fchg=Held(fund_change), fcarry=fund_level)
out = {}
for name, f0 in SL.items():
    def fn(dd, i, mask, f0=f0): return f0(dd, i - 1, mask)
    args = dict(BASE8); args['reb_offset'] = 1; args['label'] = name
    d.start_i = int(d.idx.searchsorted(pd.Timestamp(X.OOS, tz='UTC')))
    m, eq, hp = run(d, fn, ret_diag=True, **args)
    de = eq.groupby(eq.index.floor('D')).last(); r = de.pct_change().dropna(); r.index = r.index.tz_localize(None)
    out[name] = r; print(f'{name}: Sh {m["sharpe"]:.2f} CAGR {m["cagr"]*100:.1f}% MDD {m["mdd"]*100:.1f}%', flush=True)
pd.DataFrame(out).to_csv(f'research/phase4_out/{os.environ.get("OUT_PREFIX", "R4_")}sleeves.csv')
