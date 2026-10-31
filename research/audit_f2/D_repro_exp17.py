"""D_ audit: re-run exp17 (8h ML, DEC_PHASE=0, monthly) for two test months from the unchanged source (text-patched copy)
and compare with the saved walk-forward panel out_exp17_pred_8h_p0.npy; plus an in-sample (leak) control IC."""
import os, sys, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, ROOT); os.chdir(ROOT)
os.environ['DEC_PHASE'] = '0'; os.environ['PRED_ONLY'] = '1'; os.environ['SUF'] = '_D_repro'
src = open('exp17_intraday_ml.py').read()
rep = [("for kk in range(first_test, len(months)):", "for kk in [first_test, first_test + 30]:"),
       ("np.save(f'out_exp17_pred_8h{SUF}.npy', P8); del tab; gc.collect()", "np.save('logs/D_repro_pred_8h.npy', P8); globals().update(tab=tab, FEATS=FEATS, months=months, first_test=first_test, mth_col=mth_col, params=params)"),
       ("    print('DONE (pred only)'); raise SystemExit", "    print('DONE (pred only)')")]
for a, b in rep:
    assert src.count(a) == 1, a; src = src.replace(a, b)
src = src.split("\nP3, P7 = np.load")[0]
g = {'__name__': 'repro'}; exec(compile(src, 'exp17_patched', 'exec'), g)
import lightgbm as lgb
P = np.load('logs/D_repro_pred_8h.npy'); Q = np.load('out_exp17_pred_8h_p0.npy', mmap_mode='r')
ok = np.isfinite(P); print('repro cells', int(ok.sum()), ' corr(repro, saved) =', np.corrcoef(P[ok], Q[ok])[0, 1], ' max|diff| =', float(np.nanmax(np.abs(P[ok] - Q[ok]))))
tab, F_, months, ft, mc, prm, dec = g['tab'], g['FEATS'], g['months'], g['first_test'], g['mth_col'], g['params'], g['dec']
for k in [ft, ft + 30]:
    mth = months[k]; okr = np.isfinite(tab['y'].values); test = (mc == mth) & okr; tr = okr & (tab.date.values <= tab.date.values[test].max())
    y = tab.loc[tr, 'y'].values; lo, hi = np.nanpercentile(y, [0.5, 99.5])
    mdl = lgb.train(prm, lgb.Dataset(tab.loc[tr, F_].values, np.clip(y, lo, hi)), num_boost_round=300)
    sub = tab.loc[test, ['dec', 'y']].copy(); sub['p'] = mdl.predict(tab.loc[test, F_].values); sub['q'] = Q[dec[sub.dec.values], tab.loc[test, 'sym'].values]
    ic = sub.groupby('dec').apply(lambda x: x[['y', 'p']].corr(method='spearman').iloc[0, 1]).mean(); ic0 = sub.groupby('dec').apply(lambda x: x[['y', 'q']].corr(method='spearman').iloc[0, 1]).mean()
    print(f'{mth}: IC walk-forward (saved) {ic0:+.3f} | IC with test month IN training (leak control) {ic:+.3f}')
