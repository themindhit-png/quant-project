#!/usr/bin/env python3
"""Audit E debug: run bt.run (project) on core_raw from 2022-02-14 to 2022-02-16 and dump hourly equity."""
import os, sys, numpy as np, pandas as pd, pickle
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, ROOT); os.chdir(ROOT)
from bt import Data, run, quantile_ls, slip_model
from strategies import core_signal
d = Data(start='2021-01-01', end='2022-03-01', min_turn_ever=5e6)
class Held:
    def __init__(s): s.w = np.zeros(len(d.cols))
    def __call__(s, dd, i, mask):
        if dd.hh[i] % 24 == 0: s.w = quantile_ls(core_signal(dd, i, dd.universe(i, min_age_h=720, min_turn=1e7, top_n=100), 336), 0.2)
        return s.w
d.start_i = int(d.idx.get_loc(pd.Timestamp('2022-02-14 00:00', tz='UTC')))
m, eq, hp = run(d, Held(), reb_h=8, band=0.0, smooth=0.0, fee_bps=5.5, maker_share=0.7, slip_fn=slip_model(scale=0.3), gross=1.0, pos_cap=1e9, cap_short=1e9,
                uni_kwargs=dict(min_age_h=0, min_turn=0, top_n=0), verbose=False, vol_target=None, min_trade=0.0, ret_diag=True,
                end_i=int(d.idx.get_loc(pd.Timestamp('2022-02-16 00:00', tz='UTC'))))
print(eq['2022-02-15 06:00':'2022-02-15 10:00'].round(4).to_string())
