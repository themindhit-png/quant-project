#!/usr/bin/env python3
"""Feature spec for the intraday (8h) ML sleeve: ml_features_v3 (Bybit-feasible) + intraday extras.
Decision bars: 00:00 / 08:00 / 16:00 UTC open bars (closed at 01/09/17) — same grid as the daily sleeves.
Arrays end at the decision bar (inclusive). Requires >= 2161 rows."""
import numpy as np
from ml_features_v3 import compute_features_v3, FEATURES_V3, rank_norm, NEED_H  # noqa: F401

EXTRA_8H = ['r8', 'vol24', 'fund8', 'prem_last', 'hl24', 'hour']
FEATURES_8H = FEATURES_V3 + EXTRA_8H


def compute_features_8h(cff, fund, t24, prem, high, low, age_h, btc_j, hour_utc):
    F = compute_features_v3(cff, fund, t24, prem, high, low, age_h, btc_j)
    i = cff.shape[0] - 1; N = cff.shape[1]
    ret = np.full_like(cff, np.nan); ret[1:] = cff[1:] / cff[:-1] - 1.0
    F['r8'] = np.log(cff[i] / cff[i - 8])
    F['vol24'] = np.nanstd(ret[i - 23:i + 1], 0)
    F['fund8'] = np.nansum(fund[i - 7:i + 1], 0, dtype=np.float64)
    F['prem_last'] = prem[i].astype(np.float64) if prem is not None else np.full(N, np.nan)
    F['hl24'] = np.log(np.nanmax(high[i - 23:i + 1], 0) / np.nanmin(low[i - 23:i + 1], 0))
    F['hour'] = np.full(N, float(hour_utc))
    return F
