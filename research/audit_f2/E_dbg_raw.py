#!/usr/bin/env python3
"""Audit E debug: first core_raw divergence (2022-02-15 08:00) - compare bt-side weights/mask with the independent ones."""
import os, sys, numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, ROOT); os.chdir(ROOT)
from bt import Data, quantile_ls
from strategies import core_signal
d = Data(start='2021-01-01', end='2022-03-01', min_turn_ever=5e6)
i = int(d.idx.get_loc(pd.Timestamp('2022-02-15 08:00', tz='UTC'))); i0 = int(d.idx.get_loc(pd.Timestamp('2022-02-15 00:00', tz='UTC')))
m0 = d.universe(i0, min_age_h=720, min_turn=1e7, top_n=100); w = quantile_ls(core_signal(d, i0, m0, 336), 0.2)
outer = d.universe(i, min_age_h=0, min_turn=0, top_n=0)
bad = np.flatnonzero((w != 0) & ~outer)
print('held names masked out at 08:00:', [(d.cols[j], w[j], bool(d.valid[i, j]), d.t24[i, j], int(d.age[i, j]), bool(d.close_event[i0:i+1, j].any())) for j in bad])
