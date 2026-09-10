#!/usr/bin/env python3
"""Train FINAL deployable LightGBM rankers (h=3 and h=7, funding-aware targets) on all data through
2026-08 using the shared ml_features spec, and verify the shared feature code reproduces exp6's
features. Saves models/ml_h3.txt, models/ml_h7.txt and models/spec.json."""
import os, json, time, numpy as np, pandas as pd, lightgbm as lgb
from bt import Data
from ml_features import compute_features, rank_norm, FEATURES, NEED_H

START, END = '2021-01-01', '2026-08-31 23:00'
d = Data(start=START, end=END, min_turn_ever=5e6, load_hl=True, load_premium=True)
T, N = d.ret.shape
jbtc = int(np.flatnonzero(d.cols == 'BTCUSDT')[0])
dec = np.flatnonzero(d.hh % 24 == 0)
dec = dec[(dec >= NEED_H) & (dec < T - 24 * 8)]
os.makedirs('models', exist_ok=True)
t0 = time.time()
rows = []
d.prev_mask = None
for n, i in enumerate(dec):
    m = d.universe(i, min_age_h=720, min_turn=1e7, top_n=150)
    if m.sum() < 20:
        continue
    F = compute_features(d.cff[i - NEED_H + 1:i + 1], d.fund[i - NEED_H + 1:i + 1], d.t24[i - NEED_H + 1:i + 1],
                         d.prem[i - NEED_H + 1:i + 1] if d.prem is not None else None,
                         d.high[i - NEED_H + 1:i + 1], d.low[i - NEED_H + 1:i + 1], d.age[i].astype(float), jbtc)
    X = np.column_stack([F[f][m] for f in FEATURES]).astype(np.float32)
    Xr = rank_norm(X)
    df = pd.DataFrame(Xr, columns=FEATURES)
    df['day'] = n; df['sym'] = np.flatnonzero(m)
    for hz in (3, 7):
        k = 24 * hz
        fpaid = np.nansum(d.fund[i + 1:i + k + 1], 0, dtype=np.float64)
        y = np.log(d.cff[i + k] / d.cff[i]) - fpaid
        df[f'y{hz}'] = (y - np.nanmean(y[m]))[m]
    rows.append(df)
    if n % 200 == 0:
        print(f'  {n}/{len(dec)} ({time.time()-t0:.0f}s)', flush=True)
tab = pd.concat(rows, ignore_index=True)
print('table', tab.shape, flush=True)
params = dict(objective='regression', learning_rate=0.03, num_leaves=31, min_data_in_leaf=200,
              feature_fraction=0.7, bagging_fraction=0.7, bagging_freq=1, lambda_l2=10.0, verbose=-1, num_threads=2, seed=1)
spec = dict(features=FEATURES, need_h=NEED_H, universe=dict(min_age_h=720, min_turn=1e7, top_n=150),
            target='log fwd return minus funding paid (long), demeaned cross-sectionally', horizons=[3, 7],
            train_end=str(d.idx[dec[-1]].date()), params=params, rank_norm='cross-sectional rank to [-0.5,0.5]')
for hz in (3, 7):
    y = tab[f'y{hz}'].values; ok = np.isfinite(y)
    lo, hi = np.nanpercentile(y[ok], [0.5, 99.5])
    mdl = lgb.train(params, lgb.Dataset(tab.loc[ok, FEATURES].values, np.clip(y[ok], lo, hi)), num_boost_round=400)
    mdl.save_model(f'models/ml_h{hz}.txt')
    fi = pd.Series(mdl.feature_importance('gain'), index=FEATURES).sort_values(ascending=False)
    print(f'h{hz}: trained on {ok.sum()} rows; top features: ' + ', '.join(f'{k} {v/fi.sum():.3f}' for k, v in fi.head(8).items()), flush=True)
    # sanity: in-sample IC on last 90 days
    last = tab.day >= tab.day.max() - 90
    p = mdl.predict(tab.loc[last & ok, FEATURES].values)
    sub = tab.loc[last & ok, ['day', f'y{hz}']].copy(); sub['p'] = p
    ic = sub.groupby('day').apply(lambda g: g[[f'y{hz}', 'p']].corr(method='spearman').iloc[0, 1]).mean()
    print(f'   in-sample IC last 90d: {ic:.3f}')
json.dump(spec, open('models/spec.json', 'w'), indent=1)
# consistency check vs exp6 predictions: correlation between exp6 h7 panel and the new h7 model on the last 200 days
P7 = np.load('out_exp6_pred_h7.npy')
mdl7 = lgb.Booster(model_file='models/ml_h7.txt')
last = tab[tab.day >= tab.day.max() - 200]
p_new = mdl7.predict(last[FEATURES].values)
p_old = P7[dec[last.day.values], last.sym.values]
okc = np.isfinite(p_old)
print(f'corr(new final h7 model, exp6 walk-forward h7 preds) on last 200 days: {np.corrcoef(p_new[okc], p_old[okc])[0,1]:.3f}')
print('DONE')
