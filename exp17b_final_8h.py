#!/usr/bin/env python3
"""Train the FINAL deployable 8h model (grid 00/08/16, regression on clipped funding-adjusted 8h return, demeaned)
on all data through 2026-08 with the shared v4/ml_features_8h spec; save models/ml_8h.txt + models/spec_8h.json.
Also sanity-check: correlation with the walk-forward panel out_exp17_pred_8h_p0.npy on the last 200 decisions."""
import os, sys, time, json, gc, numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, 'v4')
from bt import Data
from ml_features_8h import compute_features_8h, FEATURES_8H, rank_norm, NEED_H

START, END = '2021-01-01', '2026-08-31 23:00'
d = Data(start=START, end=END, min_turn_ever=5e6, load_hl=True, load_premium=True)
T, N = d.ret.shape; jbtc = int(np.flatnonzero(d.cols == 'BTCUSDT')[0])
dec = np.flatnonzero(d.hh % 8 == 0); dec = dec[(dec >= NEED_H) & (dec < T - 16)]
t0 = time.time(); rows = []; d.prev_mask = None
for n, i in enumerate(dec):
    m = d.universe(i, min_age_h=720, min_turn=1e7, top_n=150)
    if m.sum() < 20:
        continue
    sl = slice(i - NEED_H + 1, i + 1)
    F = compute_features_8h(d.cff[sl], d.fund[sl], d.t24[sl], d.prem[sl], d.high[sl], d.low[sl], d.age[i].astype(float), jbtc, (d.hh[i] + 1) % 24)
    X = np.column_stack([F[f][m] for f in FEATURES_8H]).astype(np.float32); Xr = rank_norm(X)
    fp = np.nansum(d.fund[i + 1:i + 9], 0, dtype=np.float64)
    y = (np.log(d.cff[i + 8] / d.cff[i]) - fp)[m]; y = y - np.nanmean(y)
    df = pd.DataFrame(Xr, columns=FEATURES_8H); df['dec'] = n; df['sym'] = np.flatnonzero(m); df['y'] = y.astype(np.float32); rows.append(df)
    if n % 1000 == 0:
        print(f'  {n}/{len(dec)} ({time.time()-t0:.0f}s)', flush=True)
tab = pd.concat(rows, ignore_index=True); del rows; gc.collect()
print('table', tab.shape, flush=True)
params = dict(objective='regression', learning_rate=0.03, num_leaves=31, min_data_in_leaf=400, feature_fraction=0.7, bagging_fraction=0.7, bagging_freq=1,
              lambda_l2=10.0, verbose=-1, num_threads=2, seed=1)
ok = np.isfinite(tab['y'].values); y = tab['y'].values[ok]; lo, hi = np.nanpercentile(y, [0.5, 99.5])
mdl = lgb.train(params, lgb.Dataset(tab.loc[ok, FEATURES_8H].values, np.clip(y, lo, hi)), num_boost_round=300)
os.makedirs('models', exist_ok=True); mdl.save_model('models/ml_8h.txt')
json.dump(dict(features=FEATURES_8H, need_h=NEED_H, universe=dict(min_age_h=720, min_turn=1e7, top_n=150), horizon_h=8,
               target='log 8h fwd return minus funding paid (long), demeaned', grid='decision bars open 00/08/16 UTC (closed 01/09/17)',
               train_end=str(d.idx[dec[-1]].date()), params=params), open('models/spec_8h.json', 'w'), indent=1)
fi = pd.Series(mdl.feature_importance('gain'), index=FEATURES_8H).sort_values(ascending=False)
print('top features: ' + ', '.join(f'{k} {v/fi.sum():.3f}' for k, v in fi.head(10).items()), flush=True)
P8 = np.load('out_exp17_pred_8h_p0.npy')
last = tab[tab.dec >= tab.dec.max() - 200]
p_new = mdl.predict(last[FEATURES_8H].values); p_old = P8[dec[last.dec.values], last.sym.values]; okc = np.isfinite(p_old)
print(f'corr(final 8h model, walk-forward panel) last 200 decisions: {np.corrcoef(p_new[okc], p_old[okc])[0,1]:.3f}')
print('DONE')
