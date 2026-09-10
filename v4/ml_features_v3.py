#!/usr/bin/env python3
"""ML feature spec v3 (Bybit-feasible): v2 features + realised skew, funding change, market-regime features.
No order-flow / trade-count features (not available from Bybit klines). Same conventions as ml_features.py:
arrays end at the decision bar (inclusive); cross-sectional rank normalisation by the caller.
Regime features are identical across names on a day (trees use them as interactions)."""
import numpy as np
from ml_features import compute_features as _v2, rank_norm, NEED_H  # noqa: F401

FEATURES_V3 = ['r24', 'r72', 'r168', 'r336', 'r720', 'r1440', 'r2160', 'r720_skip24', 'r168_skip24', 'rel168', 'rel720',
               'vol168', 'vol720', 'volratio', 'fund24', 'fund72', 'fund168', 'fund720', 'fund_chg', 'prem8', 'prem24', 'prem168',
               'premz', 'logturn', 'turnratio', 'hl168', 'dist_hi720', 'dist_lo720', 'age_d', 'beta720', 'skew168',
               'reg_btcvol', 'reg_btcret', 'reg_disp24', 'reg_breadth']


def compute_features_v3(cff, fund, t24, prem, high, low, age_h, btc_j):
    F = _v2(cff, fund, t24, prem, high, low, age_h, btc_j)
    H = cff.shape[0]; i = H - 1; N = cff.shape[1]
    F['fund_chg'] = F['fund24'] * 7 - F['fund168']
    ret = np.full_like(cff, np.nan); ret[1:] = cff[1:] / cff[:-1] - 1.0
    a = ret[i - 167:i + 1]; a = np.where(np.isfinite(a), a, 0.0); m = a.mean(0); s = a.std(0) + 1e-12
    F['skew168'] = (((a - m) / s) ** 3).mean(0)
    rb = ret[i - 719:i + 1, btc_j]
    F['reg_btcvol'] = np.full(N, float(np.nanstd(rb)))
    F['reg_btcret'] = np.full(N, float(np.log(cff[i, btc_j] / cff[i - 720, btc_j])))
    r24 = F['r24']
    F['reg_disp24'] = np.full(N, float(np.nanstd(r24)))
    F['reg_breadth'] = np.full(N, float(np.nanmean(np.where(np.isfinite(r24), (r24 > 0).astype(float), np.nan))))
    return F
