#!/usr/bin/env python3
"""Signal functions for bt.run. Each: fn(data, i, mask) -> weights (N,), sides sum to 1.
All use only data with index <= i (bar i closed)."""
import numpy as np
from bt import quantile_ls, rank_weights


def _win(data, i, h):
    return data.ret[i - h + 1:i + 1]


def core_signal(data, i, mask, lb=336, min_cov=0.6):
    """Bot's core: mean/std of hourly returns over lb hours."""
    w = _win(data, i, lb)
    cnt = np.isfinite(w).sum(0)
    mu = np.nanmean(w, 0); sd = np.nanstd(w, 0, ddof=1)
    sig = np.where((cnt >= lb * min_cov) & (sd > 0) & mask, mu / sd, np.nan)
    return sig


def mom_signal(data, i, mask, lb=720, skip=24):
    a = data.cff[i - skip]; b = data.cff[i - lb]
    sig = np.where(mask & np.isfinite(a) & np.isfinite(b) & (b > 0), a / b - 1.0, np.nan)
    return sig


def vol_h(data, i, lb):
    w = _win(data, i, lb)
    return np.nanstd(w, 0, ddof=1)


def bot_baseline(data, i, mask, core_w=0.8, mom_w=0.2, top=0.2):
    """Replica of MUBITE-LADDER targets (before vol-target/caps): 80% core EW quantile L/S,
    20% momentum inverse-vol quantile L/S. NOTE: bot rebalances mom weekly and core 12h; here
    both are recomputed at every rebalance (approximation)."""
    wc = quantile_ls(core_signal(data, i, mask), top)
    wm = quantile_ls(mom_signal(data, i, mask), top, inv_vol=vol_h(data, i, 720))
    return core_w * wc + mom_w * wm


def make_quantile(sig_fn, top=0.2, inv_vol_lb=None, **kw):
    def f(data, i, mask):
        s = sig_fn(data, i, mask, **kw)
        iv = vol_h(data, i, inv_vol_lb) if inv_vol_lb else None
        return quantile_ls(s, top, inv_vol=iv)
    return f


def make_rank(sig_fn, **kw):
    def f(data, i, mask):
        s = sig_fn(data, i, mask, **kw)
        return rank_weights(s, mask)
    return f


# ---------------- additional candidate signals ----------------
def funding_signal(data, i, mask, lb_h=72):
    """Average funding rate over last lb_h hours (per 8h event). High funding -> crowded longs ->
    short candidate (carry + reversal). Sign flipped so that HIGH signal = LONG."""
    f = data.fund[i - lb_h + 1:i + 1]
    s = np.nansum(f, 0)
    cnt = (f != 0).sum(0)
    return np.where(mask & (cnt > 0), -s, np.nan)


def reversal_signal(data, i, mask, lb=24):
    a = data.cff[i]; b = data.cff[i - lb]
    return np.where(mask & (b > 0), -(a / b - 1.0), np.nan)


def lowvol_signal(data, i, mask, lb=720):
    v = vol_h(data, i, lb)
    return np.where(mask & (v > 0), -v, np.nan)


def volume_shock_signal(data, i, mask, short_h=24, long_h=24 * 30):
    """log(24h turnover / 30d average daily turnover), sign flipped: volume spikes -> short."""
    s = data.t24[i]
    l = np.nanmean(data.t24[i - long_h + 24:i + 1:24], 0)
    return np.where(mask & (l > 0) & (s > 0), -np.log(s / l), np.nan)


def zscore(x):
    m = np.nanmean(x); s = np.nanstd(x)
    return (x - m) / s if s > 0 else x * 0


def combo(weights_dict):
    """Combine z-scored signals: weights_dict = {signal_fn: weight}. Returns signal fn."""
    def f(data, i, mask):
        tot = np.zeros(len(mask))
        for fn, wt in weights_dict.items():
            s = fn(data, i, mask)
            z = zscore(np.where(mask, s, np.nan))
            tot += wt * np.where(np.isfinite(z), z, 0.0)
        return np.where(mask, tot, np.nan)
    return f
