#!/usr/bin/env python3
"""Experiment 24b — slow age-free ML sleeves (horizons 14 and 30 days; universe >= 90d; walk-forward monthly, embargo
H+7) as low-turnover diversifiers, standalone and added to F."""
import os, sys, time, json, math, numpy as np, pandas as pd
import lightgbm as lgb
from bt import Data, run, quantile_ls, slip_model
from strategies import core_signal
from betautil import rolling_std
START, END, OOS = '2021-01-01', '2026-08-31 23:00', '2022-01-01'
HZS = [int(x) for x in os.environ.get('HZS', '14,30').split(',')]; MIN_AGE_H = 90 * 24; DROP = ['age_d']
d = Data(start=START, end=END, min_turn_ever=5e6, load_hl=True, load_premium=True); T, N = d.ret.shape
dec = np.flatnonzero(d.hh % 24 == 0); dec = dec[(dec >= 24 * 100) & (dec < T - 24 * (max(HZS) + 1))]; D = len(dec)
print(f'decision days: {D}, horizons {HZS}', flush=True)
cff = d.cff; t24 = d.t24; fund = d.fund; prem = d.prem; high = d.high; low = d.low; jbtc = int(np.flatnonzero(d.cols == 'BTCUSDT')[0])
def ret_h(k): return np.log(cff[dec] / cff[dec - k])
def roll(fn, arr, k):
    out = np.full((D, N), np.nan, dtype=np.float32)
    for n, i in enumerate(dec): out[n] = fn(arr[i - k + 1:i + 1], 0)
    return out
t0 = time.time(); F = {}
for k in (24, 72, 168, 336, 720, 1440, 2160): F[f'r{k}'] = ret_h(k).astype(np.float32)
F['r720_skip24'] = np.log(cff[dec - 24] / cff[dec - 720]); F['r168_skip24'] = np.log(cff[dec - 24] / cff[dec - 168])
F['rel168'] = F['r168'] - np.log(cff[dec, jbtc] / cff[dec - 168, jbtc])[:, None]; F['rel720'] = F['r720'] - np.log(cff[dec, jbtc] / cff[dec - 720, jbtc])[:, None]
F['vol168'] = roll(np.nanstd, d.ret, 168); F['vol720'] = roll(np.nanstd, d.ret, 720); F['volratio'] = F['vol168'] / F['vol720']
for k in (24, 72, 168, 720): F[f'fund{k}'] = roll(lambda a, ax: np.nansum(a, ax, dtype=np.float64), fund, k)
F['prem8'] = roll(np.nanmean, prem, 8); F['prem24'] = roll(np.nanmean, prem, 24); F['prem168'] = roll(np.nanmean, prem, 168)
F['premz'] = (F['prem8'] - F['prem168']) / (np.nanstd(prem[dec[0] - 168:dec[0]], 0) + 1e-9)
F['logturn'] = np.log(np.where(t24[dec] > 0, t24[dec], np.nan)); F['turnratio'] = np.log(t24[dec] / roll(np.nanmean, t24, 24 * 30))
F['hl168'] = np.log(roll(np.nanmax, high, 168) / roll(np.nanmin, low, 168)); F['dist_hi720'] = np.log(cff[dec] / roll(np.nanmax, high, 720)); F['dist_lo720'] = np.log(cff[dec] / roll(np.nanmin, low, 720))
beta = np.full((D, N), np.nan)
for n, i in enumerate(dec):
    R = d.ret[i - 720 + 1:i + 1]; rb = R[:, jbtc]; ok = np.isfinite(rb); Rm = np.where(np.isfinite(R[ok]), R[ok], 0.0); rbb = rb[ok]; vb = rbb.var()
    beta[n] = ((Rm - Rm.mean(0)) * (rbb - rbb.mean())[:, None]).mean(0) / vb if vb > 0 else np.nan
F['beta720'] = beta; F = {k: v.astype(np.float32) for k, v in F.items() if k not in DROP}
d.high = d.low = d.prem = None; high = low = prem = None; import gc; gc.collect(); print(f'features {len(F)} in {time.time()-t0:.0f}s', flush=True)
fwd = {}
for hz in HZS:
    k = 24 * hz; fp = np.zeros((D, N))
    for n, i in enumerate(dec): fp[n] = np.nansum(fund[i + 1:i + k + 1], 0, dtype=np.float64)
    fwd[hz] = np.log(cff[dec + k] / cff[dec]) - fp
masks = np.zeros((D, N), dtype=bool); d.prev_mask = None
for n, i in enumerate(dec): masks[n] = d.universe(i, min_age_h=MIN_AGE_H, min_turn=1e7, top_n=150)
feats = list(F); rows = []
for n in range(D):
    m = masks[n]
    if m.sum() < 20: continue
    X = np.column_stack([F[f][n][m] for f in feats]).astype(np.float32); Xr = np.empty_like(X)
    for c in range(X.shape[1]):
        col = X[:, c]; ok = np.isfinite(col); r = np.full(len(col), np.nan, dtype=np.float32)
        if ok.sum() > 2: rr = np.empty(ok.sum()); rr[np.argsort(col[ok])] = np.arange(ok.sum()); r[ok] = rr / (ok.sum() - 1) - 0.5
        Xr[:, c] = r
    df = pd.DataFrame(Xr, columns=feats); df['day'] = n; df['sym'] = np.flatnonzero(m)
    for hz in HZS: v = fwd[hz][n][m]; df[f'y{hz}'] = v - np.nanmean(v)
    rows.append(df)
tab = pd.concat(rows, ignore_index=True); tab['date'] = d.idx[dec[tab.day.values]]; del rows, F; gc.collect()
params = dict(objective='regression', learning_rate=0.03, num_leaves=31, min_data_in_leaf=200, feature_fraction=0.7, bagging_fraction=0.7, bagging_freq=1, lambda_l2=10.0, verbose=-1, num_threads=2, seed=1)
months = sorted(set(tab.date.dt.strftime('%Y-%m'))); first_test = months.index('2022-01'); mth_col = tab.date.dt.strftime('%Y-%m').values
PAN = {}
for hz in HZS:
    ycol = f'y{hz}'; okrow = np.isfinite(tab[ycol].values); pred = np.full(len(tab), np.nan, dtype=np.float32); ics = []
    for kk in range(first_test, len(months)):
        mth = months[kk]; test = (mth_col == mth) & okrow
        if test.sum() == 0: continue
        t_start = tab.date.values[test].min(); train = (tab.date.values < (t_start - np.timedelta64(hz + 7, 'D'))) & okrow
        ytr = tab.loc[train, ycol].values; lo, hi = np.nanpercentile(ytr, [0.5, 99.5])
        mdl = lgb.train(params, lgb.Dataset(tab.loc[train, feats].values, np.clip(ytr, lo, hi), free_raw_data=True), num_boost_round=400)
        p = mdl.predict(tab.loc[test, feats].values); pred[test] = p; sub = tab.loc[test, ['day', ycol]].copy(); sub['p'] = p
        ics.append((mth, sub.groupby('day').apply(lambda g: g[[ycol, 'p']].corr(method='spearman').iloc[0, 1]).mean()))
    ic = pd.DataFrame(ics, columns=['m', 'ic']); ic['y'] = ic.m.str[:4]
    print(f'h{hz} IC by year: ' + ' '.join(f'{y}:{v:.3f}' for y, v in ic.groupby('y').ic.mean().items()) + f' ({time.time()-t0:.0f}s)', flush=True)
    P = np.full((T, N), np.nan, dtype=np.float32); okp = np.isfinite(pred); P[dec[tab.day.values[okp]], tab.sym.values[okp]] = pred[okp]
    np.save(f'out_exp24b_pred_h{hz}_noage.npy', P); PAN[hz] = P
del tab; gc.collect()
# ---- portfolio
P8 = np.load('out_exp17_pred_8h_p0.npy'); Q3, Q7 = np.load('out_exp21c_pred_h3_noage.npy'), np.load('out_exp21c_pred_h7_noage.npy'); VOL = rolling_std(d.ret, 168, 72)
instr = json.load(open('instruments.json')); launch_h = np.full(N, np.nan)
for j, s in enumerate(d.cols):
    mm = instr.get(str(s))
    if mm and mm.get('launch'): launch_h[j] = mm['launch'] / 1000 / 3600
MAJ5 = [s for s in ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT'] if s in d.cols]; jm = np.array([int(np.flatnonzero(d.cols == s)[0]) for s in MAJ5])
def sl_listing(dd, i, mask):
    age = np.maximum(dd.age[i].astype(float), np.where(np.isfinite(launch_h), dd.hh[i] - launch_h, -np.inf)); t = dd.t24[i]
    cand = (age >= 24 * 7) & (age <= 24 * 90) & dd.valid[i] & np.isfinite(t) & (t >= 2e7) & (dd.age[i] >= 72); cand[jm] = False
    r24 = dd.cff[i] / dd.cff[i - 24] - 1; cand &= ~(r24 > 0.30); idx = np.flatnonzero(cand); w = np.zeros(N)
    if len(idx) == 0: return w
    idx = idx[np.argsort(-t[idx])][:30]; ws = min(1.0 / len(idx), 0.05); w[idx] = -ws; w[jm] += ws * len(idx) / len(jm); return w
def sl_core(dd, i, mask):
    m = dd.universe(i, min_age_h=720, min_turn=1e7, top_n=100); return quantile_ls(core_signal(dd, i, m, 336), 0.2)
def make_ml(P, top=0.3, min_age_h=720, iv=False):
    def f(dd, i, mask):
        m = dd.universe(i, min_age_h=min_age_h, min_turn=1e7, top_n=150)
        return quantile_ls(np.where(m & np.isfinite(P[i]), P[i], np.nan), top, inv_vol=VOL[i] if iv else None)
    return f
class Held:
    def __init__(self, fn): self.fn = fn; self.w = np.zeros(N)
    def __call__(self, dd, i, mask):
        if dd.hh[i] % 24 == 0: self.w = self.fn(dd, i, mask)
        return self.w
class Weekly:
    def __init__(self, fn): self.fn = fn; self.w = np.zeros(N); self.last = None
    def __call__(self, dd, i, mask):
        wk = (dd.hh[i] // 24) // 7
        if wk != self.last: self.w = self.fn(dd, i, mask); self.last = wk
        return self.w
def combo(parts, weights):
    def fn(dd, i, mask):
        w = np.zeros(N)
        for f, s in zip(parts, weights): w += s * f(dd, i, mask)
        return w / sum(weights)
    return fn
ML3n, ML7n, ML8 = make_ml(Q3, min_age_h=MIN_AGE_H), make_ml(Q7, min_age_h=MIN_AGE_H), make_ml(P8, 0.2, 180 * 24, True)
MLS = {hz: make_ml(PAN[hz], min_age_h=MIN_AGE_H) for hz in HZS}
BASE8 = dict(reb_h=8, reb_offset=0, band=0.5, smooth=0.5, fee_bps=5.5, maker_share=0.7, slip_fn=slip_model(scale=0.3), gross=1.0, pos_cap=0.02, cap_short=0.015,
             cap_exempt=MAJ5, cap_exempt_val=0.30, uni_kwargs=dict(min_age_h=0, min_turn=0, top_n=0), verbose=False, vol_target=0.006, max_lev=3.0, max_gross_x=2.0)
BASE24 = dict(BASE8); BASE24.update(reb_h=24, band=0.3)
def sh(x): x = np.asarray(x, float); return float(x.mean() / x.std(ddof=1) * math.sqrt(365)) if len(x) > 30 and x.std(ddof=1) > 0 else float('nan')
def go(label, fn, base=BASE8, **kw):
    t1 = time.time(); d.start_i = int(d.idx.searchsorted(pd.Timestamp(OOS, tz='UTC'))); args = dict(base); args.update(kw); args['label'] = label
    m, eq, hp = run(d, fn, ret_diag=True, **args); de = eq.groupby(eq.index.floor('D')).last(); r = de.pct_change().dropna()
    print(f'{label:<56} Sh {m["sharpe"]:5.2f} | SEL {sh(r[:"2024-12-31"]):5.2f} CONF {sh(r["2025-01-01":]):5.2f} | CAGR {m["cagr"]*100:5.1f}% MDD {m["mdd"]*100:6.1f}% wd {m["worst_day"]*100:5.1f}% turn {m["turnover_x"]:4.0f}x cost {(m["fees"]+m["slip"])*100:4.1f}% | '
          + ' '.join(f'{y}:{v[1]:.1f}' for y, v in m['by_year'].items()) + f' ({time.time()-t1:.0f}s)', flush=True)
print('\n=== slow ML sleeves ===', flush=True)
for hz in HZS:
    go(f'ML{hz}n alone daily', MLS[hz], BASE24); go(f'ML{hz}n alone weekly rebalance', Weekly(MLS[hz]), BASE24)
Fp = [Held(sl_listing), Held(sl_core), Held(ML3n), Held(ML7n), ML8]
go('F reference', combo(Fp, [1, 1, 1, 1, 2]))
for hz in HZS: go(f'F + ML{hz}n (w1)', combo(Fp + [Held(MLS[hz])], [1, 1, 1, 1, 2, 1]))
go('F + all slow (w1 each)', combo(Fp + [Held(MLS[hz]) for hz in HZS], [1, 1, 1, 1, 2] + [1] * len(HZS)))
go('F with ML7n replaced by ML14n', combo([Held(sl_listing), Held(sl_core), Held(ML3n), Held(MLS[HZS[0]]), ML8], [1, 1, 1, 1, 2]))
print('DONE')
