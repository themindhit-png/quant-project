#!/usr/bin/env python3
"""Reverse test 2020-21 (docs/REVERSE_2020_21.md): train the F2 ML rankers ONLY on 2022-01..2026-10 data and predict 2020-04..2021-12
out of sample ("backwards"). Feature code, universe, labels and LightGBM params are those of the final-model scripts
(exp6b_noage_final.py for h3/h7 — ml_features.compute_features without age_d, universe >= 90 d; exp17b_final_8h.py for 8h —
v4/ml_features_8h, universe >= 30 d for ranks). Embargo: training rows start 2022-01-01 + (horizon + 7 d) for h3/h7, + 8 d for 8h
(mirrors the walk-forward embargo). Predictions are saved on the grid of Data(start=2020-01-01, end=2026-10-26 23:00) — the grid
exp_common uses with START=2020-01-01 — as out_exp21c_pred_h{3,7}_rev.npy / out_exp17_pred_8h_rev.npy (only rows < 2022-01-01 filled).
Run: python3 research/rev_ml.py h37   |   python3 research/rev_ml.py 8h"""
import os, sys, time, gc, numpy as np, pandas as pd, lightgbm as lgb
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'v4')); os.chdir(ROOT)
from bt import Data
MODE = sys.argv[1]
START, END, SPLIT = '2020-01-01', os.environ.get('END', '2026-10-26 23:00'), pd.Timestamp('2022-01-01', tz='UTC')
d = Data(start=START, end=END, min_turn_ever=5e6, load_hl=True, load_premium=True)
T, N = d.ret.shape; jbtc = int(np.flatnonzero(d.cols == 'BTCUSDT')[0]); t0 = time.time(); rows = []; d.prev_mask = None

def ic_by(tab, ycol, p, key):
    sub = tab[[key, ycol, 'date']].copy(); sub['p'] = p; ok = np.isfinite(sub[ycol].values)
    s = sub[ok].groupby(key).apply(lambda g: g[[ycol, 'p']].corr(method='spearman').iloc[0, 1])
    dt = sub[ok].groupby(key).date.first(); return s.groupby(dt.dt.tz_localize(None).dt.to_period('Q').astype(str).values).mean()

if MODE == 'h37':
    from ml_features import compute_features, rank_norm, FEATURES, NEED_H
    FEATS = [f for f in FEATURES if f != 'age_d']; HZS = [3, 7]; MIN_AGE_H = 90 * 24
    dec = np.flatnonzero(d.hh % 24 == 0); dec = dec[(dec >= NEED_H) & (dec < T - 24 * (max(HZS) + 1))]
    for n, i in enumerate(dec):
        m = d.universe(i, min_age_h=MIN_AGE_H, min_turn=1e7, top_n=150)
        if m.sum() < 20: continue
        F = compute_features(d.cff[i - NEED_H + 1:i + 1], d.fund[i - NEED_H + 1:i + 1], d.t24[i - NEED_H + 1:i + 1], d.prem[i - NEED_H + 1:i + 1],
                             d.high[i - NEED_H + 1:i + 1], d.low[i - NEED_H + 1:i + 1], d.age[i].astype(float), jbtc)
        X = np.column_stack([F[f][m] for f in FEATS]).astype(np.float32); df = pd.DataFrame(rank_norm(X), columns=FEATS); df['day'] = n; df['sym'] = np.flatnonzero(m)
        for hz in HZS:
            k = 24 * hz; fpaid = np.nansum(d.fund[i + 1:i + k + 1], 0, dtype=np.float64); y = np.log(d.cff[i + k] / d.cff[i]) - fpaid
            df[f'y{hz}'] = (y - np.nanmean(y[m]))[m]
        rows.append(df)
        if n % 300 == 0: print(f'  {n}/{len(dec)} ({time.time()-t0:.0f}s)', flush=True)
    tab = pd.concat(rows, ignore_index=True); del rows; tab['date'] = d.idx[dec[tab.day.values]]
    params = dict(objective='regression', learning_rate=0.03, num_leaves=31, min_data_in_leaf=200, feature_fraction=0.7, bagging_fraction=0.7, bagging_freq=1, lambda_l2=10.0, verbose=-1, num_threads=2, seed=1)
    print('table', tab.shape, tab.date.min(), tab.date.max(), flush=True)
    for hz in HZS:
        y = tab[f'y{hz}'].values; ok = np.isfinite(y); tr = ok & (tab.date >= SPLIT + pd.Timedelta(days=hz + 7)).values; te = (tab.date < SPLIT).values
        lo, hi = np.nanpercentile(y[tr], [0.5, 99.5])
        mdl = lgb.train(params, lgb.Dataset(tab.loc[tr, FEATS].values, np.clip(y[tr], lo, hi)), num_boost_round=400)
        p = mdl.predict(tab.loc[te, FEATS].values)
        print(f'h{hz}: train {tr.sum()} rows {tab.date[tr].min().date()}..{tab.date[tr].max().date()}; test {te.sum()} rows {tab.date[te].min().date()}..{tab.date[te].max().date()}', flush=True)
        print(f'h{hz} reverse OOS IC by quarter:\n' + ic_by(tab[te], f'y{hz}', p, 'day').round(3).to_string(), flush=True)
        P = np.full((T, N), np.nan, dtype=np.float32); P[dec[tab.day.values[te]], tab.sym.values[te]] = p
        np.save(f'out_exp21c_pred_h{hz}_rev.npy', P)
elif MODE == '8h':
    from ml_features_8h import compute_features_8h, FEATURES_8H, rank_norm, NEED_H
    dec = np.flatnonzero(d.hh % 8 == 0); dec = dec[(dec >= NEED_H) & (dec < T - 16)]
    for n, i in enumerate(dec):
        m = d.universe(i, min_age_h=720, min_turn=1e7, top_n=150)
        if m.sum() < 20: continue
        sl = slice(i - NEED_H + 1, i + 1)
        F = compute_features_8h(d.cff[sl], d.fund[sl], d.t24[sl], d.prem[sl], d.high[sl], d.low[sl], d.age[i].astype(float), jbtc, (d.hh[i] + 1) % 24)
        X = np.column_stack([F[f][m] for f in FEATURES_8H]).astype(np.float32); Xr = rank_norm(X)
        fp = np.nansum(d.fund[i + 1:i + 9], 0, dtype=np.float64); y = (np.log(d.cff[i + 8] / d.cff[i]) - fp)[m]; y = y - np.nanmean(y)
        df = pd.DataFrame(Xr, columns=FEATURES_8H); df['dec'] = n; df['sym'] = np.flatnonzero(m); df['y'] = y.astype(np.float32); rows.append(df)
        if n % 1000 == 0: print(f'  {n}/{len(dec)} ({time.time()-t0:.0f}s)', flush=True)
    tab = pd.concat(rows, ignore_index=True); del rows; gc.collect(); tab['date'] = d.idx[dec[tab.dec.values]]
    params = dict(objective='regression', learning_rate=0.03, num_leaves=31, min_data_in_leaf=400, feature_fraction=0.7, bagging_fraction=0.7, bagging_freq=1, lambda_l2=10.0, verbose=-1, num_threads=2, seed=1)
    y = tab['y'].values; ok = np.isfinite(y); tr = ok & (tab.date >= SPLIT + pd.Timedelta(days=8)).values; te = (tab.date < SPLIT).values
    lo, hi = np.nanpercentile(y[tr], [0.5, 99.5])
    mdl = lgb.train(params, lgb.Dataset(tab.loc[tr, FEATURES_8H].values, np.clip(y[tr], lo, hi)), num_boost_round=300)
    p = mdl.predict(tab.loc[te, FEATURES_8H].values)
    print(f'8h: train {tr.sum()} rows {tab.date[tr].min()}..{tab.date[tr].max()}; test {te.sum()} rows {tab.date[te].min()}..{tab.date[te].max()}', flush=True)
    print('8h reverse OOS IC by quarter:\n' + ic_by(tab[te], 'y', p, 'dec').round(3).to_string(), flush=True)
    P = np.full((T, N), np.nan, dtype=np.float32); P[dec[tab.dec.values[te]], tab.sym.values[te]] = p
    np.save('out_exp17_pred_8h_rev.npy', P)
print(f'DONE ({time.time()-t0:.0f}s)')
