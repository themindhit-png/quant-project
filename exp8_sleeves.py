#!/usr/bin/env python3
"""Experiment 8: make the two other candidate sleeves executable.
 S1 predicted-funding carry (gross Sharpe ~1.4 pre-cost at 8h/q10) with turnover control:
    maker execution, bands, smoothing, 24h rebalance, extreme-only (q10/q5), universe top-50.
 S2 time-series trend ensemble on top-10 majors by turnover: mean sign over lookbacks
    (14/30/60/90d), vol-scaled, weekly & daily; long-short (can be net long/short) — check
    market-neutral variant (subtract mean sign) too."""
import time, numpy as np, pandas as pd
from bt import Data, run, quantile_ls, rank_weights, slip_model
from strategies import funding_signal, zscore

START, END = '2021-06-01', '2026-08-31 23:00'
d = Data(start=START, end=END, min_turn_ever=5e6, load_premium=True)
UNI = dict(min_age_h=720, min_turn=1e7, top_n=100)
UNI50 = dict(min_age_h=720, min_turn=3e7, top_n=50)
rows = []
N = len(d.cols)


def report(label, m, t0):
    by = m['by_year']
    rows.append(dict(label=label, sharpe=m['sharpe'], cagr=m['cagr'], dvol=m['dvol'], mdd=m['mdd'], calmar=m['calmar'],
                     worst=m['worst_day'], turnover=m['turnover_x'], fees=m['fees'], slip=m['slip'], funding=m['funding'],
                     npos=m['avg_npos'], **{f'sh{y}': v[1] for y, v in by.items()}, **{f'r{y}': v[0] for y, v in by.items()}))
    print(f'{label:<58} Sh {m["sharpe"]:5.2f} CAGR {m["cagr"]*100:6.1f}% dvol {m["dvol"]*100:.2f}% MDD {m["mdd"]*100:6.1f}% '
          f'wd {m["worst_day"]*100:5.1f}% turn {m["turnover_x"]:4.0f}x cost {(m["fees"]+m["slip"])*100:.1f}% fund {-m["funding"]*100:+.1f}% npos {m["avg_npos"]:.0f} | '
          + ' '.join(f'{y}:{v[1]:.1f}' for y, v in by.items()) + f'  ({time.time()-t0:.0f}s)', flush=True)


def go(label, fn, maker=0.0, **kw):
    t0 = time.time()
    args = dict(reb_h=8, band=0.0, fee_bps=5.5, slip_fn=slip_model(scale=1.0 - maker), maker_share=maker,
                gross=1.0, pos_cap=0.02, cap_short=0.015, uni_kwargs=UNI, verbose=False, label=label)
    args.update(kw)
    m, eq = run(d, fn, **args)
    report(label, m, t0)
    return m


def pred_funding(dd, i, mask, h=8):
    pav = np.nanmean(dd.prem[i - h + 1:i + 1].astype(np.float64), 0)
    f = pav + np.clip(0.0001 - pav, -0.0005, 0.0005)
    return np.where(mask & np.isfinite(f), -f, np.nan)


def combo(dd, i, mask):
    z1 = np.nan_to_num(zscore(np.where(mask, pred_funding(dd, i, mask), np.nan)))
    z2 = np.nan_to_num(zscore(np.where(mask, funding_signal(dd, i, mask, 72), np.nan)))
    return np.where(mask, z1 + z2, np.nan)


# --- S1 carry with turnover control
go('S1a pred-fund q10 8h taker (ref)', lambda dd, i, m: quantile_ls(pred_funding(dd, i, m), 0.1))
go('S1b pred-fund q10 8h maker70', lambda dd, i, m: quantile_ls(pred_funding(dd, i, m), 0.1), maker=0.7)
go('S1c pred-fund q10 8h maker70 band50 smooth0.7', lambda dd, i, m: quantile_ls(pred_funding(dd, i, m), 0.1), maker=0.7, band=0.5, smooth=0.7)
go('S1d pred-fund q10 24h maker70 band30 smooth0.5', lambda dd, i, m: quantile_ls(pred_funding(dd, i, m), 0.1), maker=0.7, band=0.3, smooth=0.5, reb_h=24)
go('S1e pred+realised72 q10 8h maker70 smooth0.7 band50', lambda dd, i, m: quantile_ls(combo(dd, i, m), 0.1), maker=0.7, band=0.5, smooth=0.7)
go('S1f pred-fund q5 8h maker70 smooth0.7 band50', lambda dd, i, m: quantile_ls(pred_funding(dd, i, m), 0.05), maker=0.7, band=0.5, smooth=0.7)
go('S1g pred-fund q10 uni50 8h maker70 smooth0.7 band50', lambda dd, i, m: quantile_ls(pred_funding(dd, i, m), 0.1), maker=0.7, band=0.5, smooth=0.7, uni_kwargs=UNI50)
go('S1h pred-fund short-only q10 vs long BTC, maker70 smooth0.7', lambda dd, i, m: (lambda w: (np.where(w < 0, w, 0.0) + np.where(dd.cols == 'BTCUSDT', -np.where(w < 0, w, 0.0).sum(), 0.0)))(quantile_ls(pred_funding(dd, i, m), 0.1)),
   maker=0.7, band=0.5, smooth=0.7, pos_cap=1.0)


# --- S2 TS trend ensemble on majors
def ts_ens(dd, i, mask, lbs=(336, 720, 1440, 2160), neutral=False, n_top=10):
    t = dd.t24[i]
    ok = dd.valid[i] & np.isfinite(t) & (dd.age[i] >= 24 * 120)
    idx = np.flatnonzero(ok)
    idx = idx[np.argsort(-t[idx])][:n_top]
    w = np.zeros(N)
    if len(idx) == 0:
        return w
    sig = np.zeros(len(idx))
    for lb in lbs:
        a, b = dd.cff[i, idx], dd.cff[i - lb, idx]
        sig += np.sign(np.log(a / b))
    sig /= len(lbs)
    if neutral:
        sig = sig - sig.mean()
    vol = np.nanstd(dd.ret[i - 720:i + 1][:, idx], 0)
    w[idx] = sig / np.maximum(vol, 1e-6)
    s = np.abs(w).sum()
    return w / s if s > 0 else w


go('S2a TS ens majors10 weekly', ts_ens, reb_h=168, pos_cap=0.5, cap_short=0.5, uni_kwargs=dict(min_age_h=0, min_turn=0, top_n=0))
go('S2b TS ens majors10 daily band30', ts_ens, reb_h=24, band=0.3, pos_cap=0.5, cap_short=0.5, uni_kwargs=dict(min_age_h=0, min_turn=0, top_n=0))
go('S2c TS ens majors10 weekly market-neutral', lambda dd, i, m: ts_ens(dd, i, m, neutral=True), reb_h=168, pos_cap=0.5, cap_short=0.5, uni_kwargs=dict(min_age_h=0, min_turn=0, top_n=0))
go('S2d TS ens top20 weekly', lambda dd, i, m: ts_ens(dd, i, m, n_top=20), reb_h=168, pos_cap=0.5, cap_short=0.5, uni_kwargs=dict(min_age_h=0, min_turn=0, top_n=0))
go('S2e TS ens majors10 weekly maker70', ts_ens, reb_h=168, maker=0.7, pos_cap=0.5, cap_short=0.5, uni_kwargs=dict(min_age_h=0, min_turn=0, top_n=0))
pd.DataFrame(rows).to_csv('out_exp8_sleeves.csv', index=False)
print('DONE')
