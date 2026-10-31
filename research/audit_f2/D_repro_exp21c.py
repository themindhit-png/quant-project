"""D_ audit: re-run exp21c (h3, no-age) for two test months only, from the unchanged source (text-patched copy, exec'd),
and compare with the saved walk-forward panel out_exp21c_pred_h3_noage.npy. Also an IN-SAMPLE control: a model whose
training set includes the test month, to show what a leak would look like in IC terms."""
import os, sys, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, ROOT); os.chdir(ROOT)
src = open('exp21c_ml_noage.py').read()
rep = [("for hzn in (3, 7):", "for hzn in (3,):"), ("for HZ in (3, 7):", "for HZ in (3,):"),
       ("for k in range(first_test, len(months)):", "for k in [first_test, first_test + 30]:"),
       ("np.save(f'out_exp21c_pred_h{HZ}{SUF}.npy', P); ic_df.to_csv(f'out_exp21c_ic_h{HZ}{SUF}.csv', index=False)",
        "np.save('logs/D_repro_pred_h3.npy', P); globals().update(tab=tab, feat_names=feat_names, months=months, first_test=first_test, mth_col=mth_col, params=params)")]
for a, b in rep:
    assert src.count(a) == 1, a; src = src.replace(a, b)
g = {'__name__': 'repro'}; exec(compile(src, 'exp21c_patched', 'exec'), g)
import pandas as pd, lightgbm as lgb
P = np.load('logs/D_repro_pred_h3.npy'); Q = np.load('out_exp21c_pred_h3_noage.npy', mmap_mode='r')
ok = np.isfinite(P); print('repro cells', int(ok.sum()), ' corr(repro, saved) =', np.corrcoef(P[ok], Q[ok])[0, 1], ' max|diff| =', float(np.nanmax(np.abs(P[ok] - Q[ok]))))
tab, F_, months, ft, mc, prm = g['tab'], g['feat_names'], g['months'], g['first_test'], g['mth_col'], g['params']
for k in [ft, ft + 30]:
    mth = months[k]; okr = np.isfinite(tab['y3'].values); test = (mc == mth) & okr
    tr = okr & ((tab.date.values <= tab.date.values[test].max()))     # leak control: train includes the test month
    y = tab.loc[tr, 'y3'].values; lo, hi = np.nanpercentile(y, [0.5, 99.5])
    mdl = lgb.train(prm, lgb.Dataset(tab.loc[tr, F_].values, np.clip(y, lo, hi)), num_boost_round=400)
    sub = tab.loc[test, ['day', 'y3']].copy(); sub['p'] = mdl.predict(tab.loc[test, F_].values)
    ic = sub.groupby('day').apply(lambda x: x[['y3', 'p']].corr(method='spearman').iloc[0, 1]).mean()
    sub['q'] = Q[g['dec'][sub.day.values], tab.loc[test, 'sym'].values]
    ic0 = sub.groupby('day').apply(lambda x: x[['y3', 'q']].corr(method='spearman').iloc[0, 1]).mean()
    print(f'{mth}: IC walk-forward (saved) {ic0:+.3f} | IC with test month IN training (leak control) {ic:+.3f}')
