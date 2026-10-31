#!/usr/bin/env python3
"""Audit E: the project's own engine (bt.run + exp_common sleeves) on the same runs as E_indep.py.
Usage: python3 E_proj.py core_raw fcarry_raw core_full fcarry_full combo_full  -> logs/E_proj_<run>.pkl"""
import os, sys, time, pickle, numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT); os.chdir(ROOT)
import exp_common as X
from exp_common import d, BASE8, Held, sl_core, fund_level, combo
from bt import run

RAW = dict(BASE8); RAW.update(band=0.0, smooth=0.0, vol_target=None, pos_cap=1e9, cap_short=1e9, cap_exempt=None,
                              max_gross_x=None, min_trade=0.0)
def book(name):
    if name.startswith('combo'): return combo([Held(sl_core), fund_level], [1, 1])
    return Held(sl_core) if name.startswith('core') else fund_level

for name in sys.argv[1:]:
    t0 = time.time(); args = dict(RAW if name.endswith('raw') else BASE8); args['label'] = name
    d.start_i = int(d.idx.searchsorted(pd.Timestamp(X.OOS, tz='UTC')))
    m, eq, hp = run(d, book(name), ret_diag=True, **args)
    keep = {k: v for k, v in m.items() if k not in ('attrib', 'bad_days')}
    pickle.dump(dict(m=keep, eq=eq, hp=hp), open(os.path.join(ROOT, 'logs', f'E_proj_{name}.pkl'), 'wb'))
    print(f'{name}: Sharpe {m["sharpe"]:.2f} end {eq.iloc[-1]:.0f} turn {m["turnover_x"]:.0f}x ({time.time()-t0:.0f}s)', flush=True)
