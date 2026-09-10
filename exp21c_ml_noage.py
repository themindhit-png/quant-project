#!/usr/bin/env python3
"""Experiment 21c — Opus test: retrain the daily ML rankers WITHOUT the age feature and with the universe restricted
to names >= 90 days old (so that missing long-lookback features no longer encode youth). Same pipeline as exp6
(walk-forward monthly, expanding window, 7-day embargo, target = fwd return − funding, demeaned). Outputs
out_exp21c_pred_h{3,7}_noage.npy on the standard Data(2021-01-01) grid + IC by year + feature importance."""
import os, sys, time, json, numpy as np, pandas as pd
import lightgbm as lgb
from bt import Data
START, END = '2021-01-01', '2026-08-31 23:00'
MIN_AGE_H = int(os.environ.get('MIN_AGE_D', '90')) * 24
DROP = [f for f in os.environ.get('DROP_FEATS', 'age_d').split(',') if f]
SUF = os.environ.get('SUF', '_noage')
d = Data(start=START, end=END, min_turn_ever=5e6, load_hl=True, load_premium=True)
T, N = d.ret.shape
dec = np.flatnonzero(d.hh % 24 == 0); dec = dec[(dec >= 24 * 100) & (dec < T - 24 * 8)]; D = len(dec)
print(f'decision days: {D}, symbols {N}, min age {MIN_AGE_H/24:.0f}d, dropped {DROP}', flush=True)
cff = d.cff; t24 = d.t24; fund = d.fund; prem = d.prem; high = d.high; low = d.low
jbtc = int(np.flatnonzero(d.cols == 'BTCUSDT')[0])
def ret_h(k): return np.log(cff[dec] / cff[dec - k])
def roll_std_hourly(k):
    out = np.full((D, N), np.nan, dtype=np.float32)
    for n, i in enumerate(dec): out[n] = np.nanstd(d.ret[i - k + 1:i + 1], 0)
    return out
def roll_sum(arr, k):
    out = np.full((D, N), np.nan, dtype=np.float32)
    for n, i in enumerate(dec): out[n] = np.nansum(arr[i - k + 1:i + 1], 0, dtype=np.float64)
    return out
def roll_mean(arr, k):
    out = np.full((D, N), np.nan, dtype=np.float32)
    for n, i in enumerate(dec): out[n] = np.nanmean(arr[i - k + 1:i + 1], 0, dtype=np.float64)
    return out
def roll_max(arr, k):
    out = np.full((D, N), np.nan, dtype=np.float32)
    for n, i in enumerate(dec): out[n] = np.nanmax(arr[i - k + 1:i + 1], 0)
    return out
def roll_min(arr, k):
    out = np.full((D, N), np.nan, dtype=np.float32)
    for n, i in enumerate(dec): out[n] = np.nanmin(arr[i - k + 1:i + 1], 0)
    return out
t0 = time.time(); F = {}
for k in (24, 72, 168, 336, 720, 1440, 2160): F[f'r{k}'] = ret_h(k).astype(np.float32)
F['r720_skip24'] = np.log(cff[dec - 24] / cff[dec - 720]); F['r168_skip24'] = np.log(cff[dec - 24] / cff[dec - 168])
btc = np.log(cff[dec, jbtc] / cff[dec - 168, jbtc])[:, None]; F['rel168'] = F['r168'] - btc
btc30 = np.log(cff[dec, jbtc] / cff[dec - 720, jbtc])[:, None]; F['rel720'] = F['r720'] - btc30
F['vol168'] = roll_std_hourly(168); F['vol720'] = roll_std_hourly(720); F['volratio'] = F['vol168'] / F['vol720']
for k in (24, 72, 168, 720): F[f'fund{k}'] = roll_sum(fund, k)
if prem is not None:
    F['prem8'] = roll_mean(prem, 8); F['prem24'] = roll_mean(prem, 24); F['prem168'] = roll_mean(prem, 168)
    F['premz'] = (F['prem8'] - F['prem168']) / (np.nanstd(prem[dec[0] - 168:dec[0]], 0) + 1e-9)
F['logturn'] = np.log(np.where(t24[dec] > 0, t24[dec], np.nan)); tavg30 = roll_mean(t24, 24 * 30); F['turnratio'] = np.log(t24[dec] / tavg30)
F['hl168'] = np.log(roll_max(high, 168) / roll_min(low, 168)); F['dist_hi720'] = np.log(cff[dec] / roll_max(high, 720)); F['dist_lo720'] = np.log(cff[dec] / roll_min(low, 720))
F['age_d'] = d.age[dec] / 24.0
beta = np.full((D, N), np.nan)
for n, i in enumerate(dec):
    R = d.ret[i - 720 + 1:i + 1]; rb = R[:, jbtc]; ok = np.isfinite(rb)
    Rm = np.where(np.isfinite(R[ok]), R[ok], 0.0); rbb = rb[ok]; vb = rbb.var()
    beta[n] = ((Rm - Rm.mean(0)) * (rbb - rbb.mean())[:, None]).mean(0) / vb if vb > 0 else np.nan
F['beta720'] = beta
F = {k: v.astype(np.float32) for k, v in F.items() if k not in DROP}
del beta, tavg30; d.high = d.low = d.prem = None; high = low = prem = None
import gc; gc.collect(); print(f'features built: {len(F)} ({sorted(F)}) in {time.time()-t0:.0f}s', flush=True)
fwd = {}
for hzn in (3, 7):
    k = 24 * hzn; fpaid = np.zeros((D, N))
    for n, i in enumerate(dec): fpaid[n] = np.nansum(fund[i + 1:i + k + 1], 0, dtype=np.float64)
    fwd[hzn] = np.log(cff[dec + k] / cff[dec]) - fpaid
masks = np.zeros((D, N), dtype=bool); d.prev_mask = None
for n, i in enumerate(dec): masks[n] = d.universe(i, min_age_h=MIN_AGE_H, min_turn=1e7, top_n=150)
feat_names = list(F); rows = []
for n in range(D):
    m = masks[n]
    if m.sum() < 20: continue
    X = np.column_stack([F[f][n][m] for f in feat_names]).astype(np.float32); Xr = np.empty_like(X)
    for c in range(X.shape[1]):
        col = X[:, c]; ok = np.isfinite(col); r = np.full(len(col), np.nan, dtype=np.float32)
        if ok.sum() > 2:
            rr = np.empty(ok.sum()); rr[np.argsort(col[ok])] = np.arange(ok.sum()); r[ok] = rr / (ok.sum() - 1) - 0.5
        Xr[:, c] = r
    df = pd.DataFrame(Xr, columns=feat_names); df['day'] = n; df['sym'] = np.flatnonzero(m)
    for hzn, arr in fwd.items():
        v = arr[n][m]; df[f'y{hzn}'] = v - np.nanmean(v)
    rows.append(df)
tab = pd.concat(rows, ignore_index=True); tab['date'] = d.idx[dec[tab.day.values]]; del rows
print(f'table: {tab.shape}, {tab.date.min().date()} -> {tab.date.max().date()}', flush=True)
params = dict(objective='regression', learning_rate=0.03, num_leaves=31, min_data_in_leaf=200, feature_fraction=0.7, bagging_fraction=0.7, bagging_freq=1, lambda_l2=10.0, verbose=-1, num_threads=2, seed=1)
months = sorted(set(tab.date.dt.strftime('%Y-%m'))); first_test = months.index('2022-01'); mth_col = tab.date.dt.strftime('%Y-%m').values
for HZ in (3, 7):
    ycol = f'y{HZ}'; okrow = np.isfinite(tab[ycol].values); pred = np.full(len(tab), np.nan, dtype=np.float32); ics = []; imp = np.zeros(len(feat_names))
    for k in range(first_test, len(months)):
        mth = months[k]; test = (mth_col == mth) & okrow
        if test.sum() == 0: continue
        t_start = tab.date.values[test].min(); train = (tab.date.values < (t_start - np.timedelta64(HZ + 7, 'D'))) & okrow
        Xtr, ytr = tab.loc[train, feat_names].values, tab.loc[train, ycol].values
        lo, hi = np.nanpercentile(ytr, [0.5, 99.5]); ytr = np.clip(ytr, lo, hi)
        mdl = lgb.train(params, lgb.Dataset(Xtr, ytr, free_raw_data=True), num_boost_round=400)
        p = mdl.predict(tab.loc[test, feat_names].values); pred[test] = p; imp += mdl.feature_importance('gain')
        sub = tab.loc[test, ['day', ycol]].copy(); sub['p'] = p
        ics.append((mth, sub.groupby('day').apply(lambda g: g[[ycol, 'p']].corr(method='spearman').iloc[0, 1]).mean(), int(train.sum())))
        if k % 6 == 0: print(f'  h{HZ} {mth}: train {int(train.sum()):>7} rows  IC {ics[-1][1]:+.3f} ({time.time()-t0:.0f}s)', flush=True)
    ic_df = pd.DataFrame(ics, columns=['month', 'ic', 'ntrain'])
    print(f'\nh{HZ}{SUF} mean IC by year:'); print(ic_df.assign(y=ic_df.month.str[:4]).groupby('y').ic.mean().round(4).to_string())
    fi = pd.Series(imp, index=feat_names).sort_values(ascending=False); print(f'h{HZ}{SUF} feature importance (gain):'); print((fi / fi.sum()).round(3).head(12).to_string(), flush=True)
    P = np.full((T, N), np.nan, dtype=np.float32); okp = np.isfinite(pred); P[dec[tab.day.values[okp]], tab.sym.values[okp]] = pred[okp]
    np.save(f'out_exp21c_pred_h{HZ}{SUF}.npy', P); ic_df.to_csv(f'out_exp21c_ic_h{HZ}{SUF}.csv', index=False)
print('DONE')
