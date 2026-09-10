#!/usr/bin/env python3
"""Shared ML feature spec for research (exp6) and the live bot (v4).
All features are computed from hourly arrays ending at the decision bar (inclusive):
  cff  : forward-filled close (H, N)        fund : funding rate per hour (0 where none) (H, N)
  t24  : rolling 24h quote turnover (H, N)  prem : premium index close (H, N) (may be None)
  high, low : bar high/low (H, N)           age_h: hours since listing (N,)
  btc_j: column index of BTCUSDT
The decision bar is the LAST row. Returns dict name -> (N,) float array. Cross-sectional rank
normalisation (rank_norm) is applied by the caller within the eligible universe.
Requires at least 2160+1 rows of history."""
import numpy as np

FEATURES = ['r24', 'r72', 'r168', 'r336', 'r720', 'r1440', 'r2160', 'r720_skip24', 'r168_skip24', 'rel168', 'rel720',
            'vol168', 'vol720', 'volratio', 'fund24', 'fund72', 'fund168', 'fund720', 'prem8', 'prem24', 'prem168', 'premz',
            'logturn', 'turnratio', 'hl168', 'dist_hi720', 'dist_lo720', 'age_d', 'beta720']
NEED_H = 2161


def _nanstd(a):
    return np.nanstd(a, 0, ddof=0)


def compute_features(cff, fund, t24, prem, high, low, age_h, btc_j):
    H = cff.shape[0]
    assert H >= NEED_H, 'need >= 2161 hourly rows'
    i = H - 1
    c = cff
    ret = np.full_like(c, np.nan)
    ret[1:] = c[1:] / c[:-1] - 1.0
    F = {}
    for k in (24, 72, 168, 336, 720, 1440, 2160):
        F[f'r{k}'] = np.log(c[i] / c[i - k])
    F['r720_skip24'] = np.log(c[i - 24] / c[i - 720])
    F['r168_skip24'] = np.log(c[i - 24] / c[i - 168])
    F['rel168'] = F['r168'] - np.log(c[i, btc_j] / c[i - 168, btc_j])
    F['rel720'] = F['r720'] - np.log(c[i, btc_j] / c[i - 720, btc_j])
    F['vol168'] = _nanstd(ret[i - 167:i + 1])
    F['vol720'] = _nanstd(ret[i - 719:i + 1])
    F['volratio'] = F['vol168'] / F['vol720']
    for k in (24, 72, 168, 720):
        F[f'fund{k}'] = np.nansum(fund[i - k + 1:i + 1], 0, dtype=np.float64)
    if prem is not None:
        p = prem.astype(np.float64)
        F['prem8'] = np.nanmean(p[i - 7:i + 1], 0)
        F['prem24'] = np.nanmean(p[i - 23:i + 1], 0)
        F['prem168'] = np.nanmean(p[i - 167:i + 1], 0)
        F['premz'] = (F['prem8'] - F['prem168']) / (np.nanstd(p[i - 167:i + 1], 0) + 1e-9)
    else:
        for k in ('prem8', 'prem24', 'prem168', 'premz'):
            F[k] = np.full(c.shape[1], np.nan)
    tt = t24[i].astype(np.float64)
    F['logturn'] = np.log(np.where(tt > 0, tt, np.nan))
    tavg30 = np.nanmean(t24[i - 24 * 30 + 24:i + 1:24].astype(np.float64), 0)
    F['turnratio'] = np.log(tt / tavg30)
    F['hl168'] = np.log(np.nanmax(high[i - 167:i + 1], 0) / np.nanmin(low[i - 167:i + 1], 0))
    F['dist_hi720'] = np.log(c[i] / np.nanmax(high[i - 719:i + 1], 0))
    F['dist_lo720'] = np.log(c[i] / np.nanmin(low[i - 719:i + 1], 0))
    F['age_d'] = age_h / 24.0
    R = ret[i - 719:i + 1]
    rb = R[:, btc_j]; ok = np.isfinite(rb)
    Rm = np.where(np.isfinite(R[ok]), R[ok], 0.0); rbb = rb[ok]
    vb = rbb.var()
    F['beta720'] = ((Rm - Rm.mean(0)) * (rbb - rbb.mean())[:, None]).mean(0) / vb if vb > 0 else np.full(c.shape[1], np.nan)
    return F


def rank_norm(X):
    """Column-wise cross-sectional rank normalisation to [-0.5, 0.5]; NaN preserved. X: (n, f)."""
    Xr = np.full(X.shape, np.nan, dtype=np.float32)
    for c in range(X.shape[1]):
        col = X[:, c]; ok = np.isfinite(col)
        if ok.sum() > 2:
            rr = np.empty(ok.sum()); rr[np.argsort(col[ok])] = np.arange(ok.sum())
            Xr[ok, c] = rr / (ok.sum() - 1) - 0.5
    return Xr
