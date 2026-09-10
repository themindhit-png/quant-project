#!/usr/bin/env python3
"""Train FINAL deployable age-free LightGBM rankers (h=3, h=7): features = ml_features.FEATURES minus age_d,
universe = names >= 90 days old (exp21c/21d). Saves models/ml_h3_noage.txt, models/ml_h7_noage.txt, models/spec_noage.json.
Deploy with ML_DROP_FEATS=age_d ML_MIN_AGE_D=90 and MODEL files renamed/pointed accordingly."""
import os, json, time, numpy as np, pandas as pd, lightgbm as lgb
from bt import Data
from ml_features import compute_features, rank_norm, FEATURES, NEED_H
START, END = '2021-01-01', '2026-08-31 23:00'
DROP = ['age_d']; FEATS = [f for f in FEATURES if f not in DROP]; MIN_AGE_H = 90 * 24
HZS = [int(x) for x in os.environ.get('HZS', '3,7').split(',')]
d = Data(start=START, end=END, min_turn_ever=5e6, load_hl=True, load_premium=True)
T, N = d.ret.shape; jbtc = int(np.flatnonzero(d.cols == 'BTCUSDT')[0])
dec = np.flatnonzero(d.hh % 24 == 0); dec = dec[(dec >= NEED_H) & (dec < T - 24 * (max(HZS) + 1))]
os.makedirs('models', exist_ok=True); t0 = time.time(); rows = []; d.prev_mask = None
for n, i in enumerate(dec):
    m = d.universe(i, min_age_h=MIN_AGE_H, min_turn=1e7, top_n=150)
    if m.sum() < 20: continue
    F = compute_features(d.cff[i - NEED_H + 1:i + 1], d.fund[i - NEED_H + 1:i + 1], d.t24[i - NEED_H + 1:i + 1],
                         d.prem[i - NEED_H + 1:i + 1] if d.prem is not None else None, d.high[i - NEED_H + 1:i + 1], d.low[i - NEED_H + 1:i + 1], d.age[i].astype(float), jbtc)
    X = np.column_stack([F[f][m] for f in FEATS]).astype(np.float32); df = pd.DataFrame(rank_norm(X), columns=FEATS); df['day'] = n; df['sym'] = np.flatnonzero(m)
    for hz in HZS:
        k = 24 * hz; fpaid = np.nansum(d.fund[i + 1:i + k + 1], 0, dtype=np.float64); y = np.log(d.cff[i + k] / d.cff[i]) - fpaid
        df[f'y{hz}'] = (y - np.nanmean(y[m]))[m]
    rows.append(df)
    if n % 200 == 0: print(f'  {n}/{len(dec)} ({time.time()-t0:.0f}s)', flush=True)
tab = pd.concat(rows, ignore_index=True); print('table', tab.shape, flush=True)
params = dict(objective='regression', learning_rate=0.03, num_leaves=31, min_data_in_leaf=200, feature_fraction=0.7, bagging_fraction=0.7, bagging_freq=1, lambda_l2=10.0, verbose=-1, num_threads=2, seed=1)
spec = dict(features=FEATS, dropped=DROP, need_h=NEED_H, universe=dict(min_age_h=MIN_AGE_H, min_turn=1e7, top_n=150), target='log fwd return minus funding paid (long), demeaned cross-sectionally',
            horizons=HZS, train_end=str(d.idx[dec[-1]].date()), params=params, rank_norm='cross-sectional rank to [-0.5,0.5]', env=dict(ML_DROP_FEATS='age_d', ML_MIN_AGE_D=90))
for hz in HZS:
    y = tab[f'y{hz}'].values; ok = np.isfinite(y); lo, hi = np.nanpercentile(y[ok], [0.5, 99.5])
    mdl = lgb.train(params, lgb.Dataset(tab.loc[ok, FEATS].values, np.clip(y[ok], lo, hi)), num_boost_round=400); mdl.save_model(f'models/ml_h{hz}_noage.txt')
    fi = pd.Series(mdl.feature_importance('gain'), index=FEATS).sort_values(ascending=False)
    print(f'h{hz}: trained on {ok.sum()} rows; top features: ' + ', '.join(f'{k} {v/fi.sum():.3f}' for k, v in fi.head(8).items()), flush=True)
    last = tab.day >= tab.day.max() - 90; p = mdl.predict(tab.loc[last & ok, FEATS].values)
    sub = tab.loc[last & ok, ['day', f'y{hz}']].copy(); sub['p'] = p
    print(f'   in-sample IC last 90d: {sub.groupby("day").apply(lambda g: g[[f"y{hz}", "p"]].corr(method="spearman").iloc[0, 1]).mean():.3f}')
json.dump(spec, open('models/spec_noage.json' if HZS == [3, 7] else f'models/spec_noage_h{"_".join(map(str, HZS))}.json', 'w'), indent=1)
chk = 7 if 7 in HZS else HZS[-1]; pth = 'out_exp21c_pred_h7_noage.npy' if chk == 7 else f'out_exp24b_pred_h{chk}_noage.npy'
if os.path.exists(pth):
    Q = np.load(pth); mdlc = lgb.Booster(model_file=f'models/ml_h{chk}_noage.txt')
    last = tab[tab.day >= tab.day.max() - 200]; p_new = mdlc.predict(last[FEATS].values); p_old = Q[dec[last.day.values], last.sym.values]; okc = np.isfinite(p_old)
    print(f'corr(final noage h{chk} model, walk-forward preds) on last 200 days: {np.corrcoef(p_new[okc], p_old[okc])[0,1]:.3f}')
print('DONE')
