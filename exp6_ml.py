#!/usr/bin/env python3
"""Experiment 6: walk-forward ML cross-sectional ranker (LightGBM) on daily features.
Daily panels are sampled from hourly data at 00:00 UTC boundaries (bar open_time 00:00 is bar D*24;
"day D closes" at bar D*24+23). Decision at the close of bar D*24+23 (= 00:00 UTC of D+1), which is
exactly bt.run's reb_h=24 rebalance bar (hh % 24 == 23? no: bt rebalances when hh%24==0, i.e. the bar
with open_time 00:00, closed at 01:00). To stay consistent we build features on the last 24 bars ending
at the 00:00-open bar inclusive and predict the following 24h/72h/168h.
Target: cross-sectionally demeaned forward return (3d default). Walk-forward monthly retraining with
an expanding window and a 7-day embargo. Outputs prediction panel -> portfolio via bt.run."""
import os, sys, time, json, numpy as np, pandas as pd
import lightgbm as lgb
from bt import Data, run, rank_weights, quantile_ls, slip_model

START, END = '2021-01-01', '2026-08-31 23:00'
HORIZON_D = int(os.environ.get('ML_H', '3'))
d = Data(start=START, end=END, min_turn_ever=5e6, load_hl=True, load_premium=True)
T, N = d.ret.shape
hh0 = d.hh[0]
# day boundaries: bars where hh % 24 == 0 (open_time 00:00). Use them as decision bars (bt convention).
dec = np.flatnonzero(d.hh % 24 == 0)
dec = dec[(dec >= 24 * 100) & (dec < T - 24 * 8)]
D = len(dec)
print(f'decision days: {D}, symbols {N}', flush=True)

cff = d.cff; t24 = d.t24; fund = d.fund; prem = d.prem          # keep f32 panels as-is (memory)
high = d.high; low = d.low
jbtc = int(np.flatnonzero(d.cols == 'BTCUSDT')[0])


def px(k):      # close k hours before the decision bar close
    return cff[dec - k]


def ret_h(k):
    return np.log(cff[dec] / cff[dec - k])


def roll_std_hourly(k):
    out = np.full((D, N), np.nan, dtype=np.float32)
    for n, i in enumerate(dec):
        w = d.ret[i - k + 1:i + 1]
        out[n] = np.nanstd(w, 0)
    return out


def roll_sum(arr, k):
    out = np.full((D, N), np.nan, dtype=np.float32)
    for n, i in enumerate(dec):
        out[n] = np.nansum(arr[i - k + 1:i + 1], 0, dtype=np.float64)
    return out


def roll_mean(arr, k):
    out = np.full((D, N), np.nan, dtype=np.float32)
    for n, i in enumerate(dec):
        out[n] = np.nanmean(arr[i - k + 1:i + 1], 0, dtype=np.float64)
    return out


def roll_max(arr, k):
    out = np.full((D, N), np.nan, dtype=np.float32)
    for n, i in enumerate(dec):
        out[n] = np.nanmax(arr[i - k + 1:i + 1], 0)
    return out


def roll_min(arr, k):
    out = np.full((D, N), np.nan, dtype=np.float32)
    for n, i in enumerate(dec):
        out[n] = np.nanmin(arr[i - k + 1:i + 1], 0)
    return out


t0 = time.time()
F = {}
for k in (24, 72, 168, 336, 720, 1440, 2160):
    F[f'r{k}'] = ret_h(k).astype(np.float32)
F['r720_skip24'] = np.log(cff[dec - 24] / cff[dec - 720])
F['r168_skip24'] = np.log(cff[dec - 24] / cff[dec - 168])
btc = np.log(cff[dec, jbtc] / cff[dec - 168, jbtc])[:, None]
F['rel168'] = F['r168'] - btc
btc30 = np.log(cff[dec, jbtc] / cff[dec - 720, jbtc])[:, None]
F['rel720'] = F['r720'] - btc30
F['vol168'] = roll_std_hourly(168)
F['vol720'] = roll_std_hourly(720)
F['volratio'] = F['vol168'] / F['vol720']
F['fund24'] = roll_sum(fund, 24)
F['fund72'] = roll_sum(fund, 72)
F['fund168'] = roll_sum(fund, 168)
F['fund720'] = roll_sum(fund, 720)
if prem is not None:
    F['prem8'] = roll_mean(prem, 8)
    F['prem24'] = roll_mean(prem, 24)
    F['prem168'] = roll_mean(prem, 168)
    F['premz'] = (F['prem8'] - F['prem168']) / (np.nanstd(prem[dec[0] - 168:dec[0]], 0) + 1e-9)
F['logturn'] = np.log(np.where(t24[dec] > 0, t24[dec], np.nan))
tavg30 = roll_mean(t24[:, :], 24 * 30)
F['turnratio'] = np.log(t24[dec] / tavg30)
F['hl168'] = np.log(roll_max(high, 168) / roll_min(low, 168))
F['dist_hi720'] = np.log(cff[dec] / roll_max(high, 720))
F['dist_lo720'] = np.log(cff[dec] / roll_min(low, 720))
F['age_d'] = d.age[dec] / 24.0
# beta to BTC over 720h
beta = np.full((D, N), np.nan)
for n, i in enumerate(dec):
    R = d.ret[i - 720 + 1:i + 1]
    rb = R[:, jbtc]; ok = np.isfinite(rb)
    Rm = np.where(np.isfinite(R[ok]), R[ok], 0.0); rbb = rb[ok]
    vb = rbb.var()
    beta[n] = ((Rm - Rm.mean(0)) * (rbb - rbb.mean())[:, None]).mean(0) / vb if vb > 0 else np.nan
F['beta720'] = beta
F = {k: v.astype(np.float32) for k, v in F.items()}
del beta, tavg30
d.high = d.low = d.prem = None; high = low = prem = None      # free ~0.5 GB
import gc; gc.collect()
print(f'features built: {len(F)} in {time.time()-t0:.0f}s', flush=True)

# ---- targets: forward log return over HORIZON_D days (from decision bar close) MINUS funding paid by a
# long position over the horizon (so the model learns total return incl. carry), demeaned within universe
fwd = {}
for hzn in (1, 3, 7):
    k = 24 * hzn
    fpaid = np.zeros((D, N))
    for n, i in enumerate(dec):
        fpaid[n] = np.nansum(fund[i + 1:i + k + 1], 0, dtype=np.float64)
    fwd[hzn] = np.log(cff[dec + k] / cff[dec]) - fpaid

# ---- universe mask per decision day (same as bt: age>=30d, t24>=1e7, top 150)
masks = np.zeros((D, N), dtype=bool)
d.prev_mask = None
for n, i in enumerate(dec):
    masks[n] = d.universe(i, min_age_h=720, min_turn=1e7, top_n=150)

# ---- assemble long table
feat_names = list(F)
rows = []
for n in range(D):
    m = masks[n]
    if m.sum() < 20:
        continue
    X = np.column_stack([F[f][n][m] for f in feat_names]).astype(np.float32)
    # cross-sectional rank-normalise features (robust to scale, regime)
    Xr = np.empty_like(X)
    for c in range(X.shape[1]):
        col = X[:, c]
        ok = np.isfinite(col)
        r = np.full(len(col), np.nan, dtype=np.float32)
        if ok.sum() > 2:
            rr = np.empty(ok.sum()); rr[np.argsort(col[ok])] = np.arange(ok.sum())
            r[ok] = rr / (ok.sum() - 1) - 0.5
        Xr[:, c] = r
    y = {}
    for hzn, arr in fwd.items():
        v = arr[n][m]
        y[hzn] = v - np.nanmean(v)
    df = pd.DataFrame(Xr, columns=feat_names)
    df['day'] = n; df['sym'] = np.flatnonzero(m)
    for hzn in fwd:
        df[f'y{hzn}'] = y[hzn]
    rows.append(df)
tab = pd.concat(rows, ignore_index=True)
tab['date'] = d.idx[dec[tab.day.values]]
print(f'table: {tab.shape}, {tab.date.min().date()} -> {tab.date.max().date()}', flush=True)
del rows

# ---- walk-forward monthly, for horizons 3 and 7 days
params = dict(objective='regression', learning_rate=0.03, num_leaves=31, min_data_in_leaf=200,
              feature_fraction=0.7, bagging_fraction=0.7, bagging_freq=1, lambda_l2=10.0, verbose=-1,
              num_threads=2, seed=1)
months = sorted(set(tab.date.dt.strftime('%Y-%m')))
first_test = months.index('2022-01')
mth_col = tab.date.dt.strftime('%Y-%m').values
PANELS = {}
for HZ in (3, 7):
    ycol = f'y{HZ}'
    okrow = np.isfinite(tab[ycol].values)
    pred = np.full(len(tab), np.nan, dtype=np.float32)
    ics = []
    imp = np.zeros(len(feat_names))
    for k in range(first_test, len(months)):
        mth = months[k]
        test = (mth_col == mth) & okrow
        if test.sum() == 0:
            continue
        t_start = tab.date.values[test].min()
        train = (tab.date.values < (t_start - np.timedelta64(HZ + 7, 'D'))) & okrow     # embargo
        Xtr, ytr = tab.loc[train, feat_names].values, tab.loc[train, ycol].values
        lo, hi = np.nanpercentile(ytr, [0.5, 99.5]); ytr = np.clip(ytr, lo, hi)   # tame squeezes
        mdl = lgb.train(params, lgb.Dataset(Xtr, ytr, free_raw_data=True), num_boost_round=400)
        p = mdl.predict(tab.loc[test, feat_names].values)
        pred[test] = p
        imp += mdl.feature_importance('gain')
        sub = tab.loc[test, ['day', ycol]].copy(); sub['p'] = p
        ic = sub.groupby('day').apply(lambda g: g[[ycol, 'p']].corr(method='spearman').iloc[0, 1]).mean()
        ics.append((mth, ic, int(train.sum())))
        if k % 6 == 0:
            print(f'  h{HZ} {mth}: train {int(train.sum()):>7} rows  mean daily rank-IC {ic:+.3f}  ({time.time()-t0:.0f}s)', flush=True)
    ic_df = pd.DataFrame(ics, columns=['month', 'ic', 'ntrain'])
    print(f'\nh{HZ} mean IC by year:'); print(ic_df.assign(y=ic_df.month.str[:4]).groupby('y').ic.mean().round(4).to_string())
    fi = pd.Series(imp, index=feat_names).sort_values(ascending=False)
    print(f'h{HZ} feature importance (gain):'); print((fi / fi.sum()).round(3).head(12).to_string(), flush=True)
    P = np.full((T, N), np.nan, dtype=np.float32)
    okp = np.isfinite(pred)
    P[dec[tab.day.values[okp]], tab.sym.values[okp]] = pred[okp]
    np.save(f'out_exp6_pred_h{HZ}.npy', P)
    PANELS[HZ] = P
    pd.DataFrame({'date': tab.date.values, 'sym': tab.sym.values, 'pred': pred, 'y': tab[ycol].values}).to_parquet(f'out_exp6_tab_h{HZ}.parquet')
P = PANELS[HORIZON_D]


def ml_sig(dd, i, mask, top=0.2, use_rank=True):
    s = P[i]
    s = np.where(mask & np.isfinite(s), s, np.nan)
    return rank_weights(s, mask) if use_rank else quantile_ls(s, top)


UNI = dict(min_age_h=720, min_turn=1e7, top_n=150)
for label, kw in [
    ('ML rank daily taker', dict(reb_h=24)),
    ('ML q20 daily taker', dict(reb_h=24, sig=lambda dd, i, m: ml_sig(dd, i, m, 0.2, False))),
    ('ML rank daily band30 smooth0.5 maker70', dict(reb_h=24, band=0.3, smooth=0.5, maker_share=0.7, slip_fn=slip_model(scale=0.3))),
    ('ML rank daily band30 smooth0.5 maker70 VT0.6', dict(reb_h=24, band=0.3, smooth=0.5, maker_share=0.7, slip_fn=slip_model(scale=0.3), vol_target=0.006, max_lev=2.5)),
]:
    sig = kw.pop('sig', ml_sig)
    args = dict(fee_bps=5.5, slip_fn=slip_model(), gross=1.0, pos_cap=0.02, cap_short=0.015, uni_kwargs=UNI, label=label)
    args.update(kw)
    # restrict evaluation to test period
    d.start_i = int(d.idx.searchsorted(pd.Timestamp('2022-01-01', tz='UTC')))
    run(d, sig, **args)
print('DONE')
