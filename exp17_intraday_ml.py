#!/usr/bin/env python3
"""Experiment 17 (memory-light): intraday ML — decisions every 8h at funding times, target = next-8h log return
minus funding paid at the end of the window (demeaned), regression on clipped target (rank target rejected in
exp16). Features: ml_features_v3 (Bybit-feasible) + intraday extras (r8, vol24, fund8, prem_last, hl24, hour).
Streaming feature computation per decision (no (D,N) matrices). Walk-forward monthly, embargo 7d.
Portfolio: intraday sleeve alone (reb 8h) and added to the daily v2 book."""
import os, time, json, gc, numpy as np, pandas as pd, lightgbm as lgb
from bt import Data, run, quantile_ls, slip_model
from strategies import core_signal
from ml_features_v3 import compute_features_v3, FEATURES_V3, rank_norm

START, END, OOS = '2021-01-01', '2026-08-31 23:00', '2022-01-01'
d = Data(start=START, end=END, min_turn_ever=5e6, load_hl=True, load_premium=True)
T, N = d.ret.shape
jbtc = int(np.flatnonzero(d.cols == 'BTCUSDT')[0])
HZ = 8; NEED = 2161
DEC_PHASE = int(os.environ.get('DEC_PHASE', '7'))        # 7: decisions at 07/15/23 bars (funding-aligned); 0: 00/08/16 (daily-grid aligned)
SUF = os.environ.get('SUF', '')
dec = np.flatnonzero(d.hh % 8 == DEC_PHASE)
dec = dec[(dec >= NEED) & (dec < T - 48)]
D = len(dec); print(f'decisions: {D}', flush=True)
EXTRA = ['r8', 'vol24', 'fund8', 'prem_last', 'hl24', 'hour']
FEATS = FEATURES_V3 + EXTRA
t0 = time.time(); rows = []; d.prev_mask = None
for n, i in enumerate(dec):
    m = d.universe(i, min_age_h=720, min_turn=1e7, top_n=150)
    if m.sum() < 20:
        continue
    sl = slice(i - NEED + 1, i + 1)
    F = compute_features_v3(d.cff[sl], d.fund[sl], d.t24[sl], d.prem[sl], d.high[sl], d.low[sl], d.age[i].astype(float), jbtc)
    F['r8'] = np.log(d.cff[i] / d.cff[i - 8])
    F['vol24'] = np.nanstd(d.ret[i - 23:i + 1], 0)
    F['fund8'] = np.nansum(d.fund[i - 7:i + 1], 0, dtype=np.float64)
    F['prem_last'] = d.prem[i].astype(np.float64)
    F['hl24'] = np.log(np.nanmax(d.high[i - 23:i + 1], 0) / np.nanmin(d.low[i - 23:i + 1], 0))
    F['hour'] = np.full(N, float((d.hh[i] + 1) % 24))
    X = np.column_stack([F[f][m] for f in FEATS]).astype(np.float32)
    Xr = rank_norm(X)
    fp = np.nansum(d.fund[i + 1:i + HZ + 1], 0, dtype=np.float64)
    y = (np.log(d.cff[i + HZ] / d.cff[i]) - fp)[m]; y = y - np.nanmean(y)
    df = pd.DataFrame(Xr, columns=FEATS); df['dec'] = n; df['sym'] = np.flatnonzero(m); df['y'] = y.astype(np.float32)
    rows.append(df)
    if n % 500 == 0:
        print(f'  {n}/{D} ({time.time()-t0:.0f}s)', flush=True)
tab = pd.concat(rows, ignore_index=True); del rows; gc.collect()
tab['date'] = d.idx[dec[tab.dec.values]]
d.high = d.low = None; gc.collect()
print(f'table {tab.shape} ({time.time()-t0:.0f}s)', flush=True)
params = dict(objective='regression', learning_rate=0.03, num_leaves=31, min_data_in_leaf=400, feature_fraction=0.7, bagging_fraction=0.7,
              bagging_freq=1, lambda_l2=10.0, verbose=-1, num_threads=2, seed=1)
months = sorted(set(tab.date.dt.strftime('%Y-%m'))); first_test = months.index('2022-01'); mth_col = tab.date.dt.strftime('%Y-%m').values
okrow = np.isfinite(tab['y'].values); pred = np.full(len(tab), np.nan, dtype=np.float32); ics = []
for kk in range(first_test, len(months)):
    mth = months[kk]; test = (mth_col == mth) & okrow
    if test.sum() == 0:
        continue
    t_start = tab.date.values[test].min(); train = (tab.date.values < (t_start - np.timedelta64(8, 'D'))) & okrow
    ytr = tab.loc[train, 'y'].values; lo, hi = np.nanpercentile(ytr, [0.5, 99.5])
    mdl = lgb.train(params, lgb.Dataset(tab.loc[train, FEATS].values, np.clip(ytr, lo, hi), free_raw_data=True), num_boost_round=300)
    p = mdl.predict(tab.loc[test, FEATS].values); pred[test] = p
    sub = tab.loc[test, ['dec', 'y']].copy(); sub['p'] = p
    ics.append((mth, sub.groupby('dec').apply(lambda g: g[['y', 'p']].corr(method='spearman').iloc[0, 1]).mean()))
    if kk % 6 == 0:
        print(f'  {mth} IC {ics[-1][1]:.3f} ({time.time()-t0:.0f}s)', flush=True)
ic = pd.DataFrame(ics, columns=['m', 'ic']); ic['y'] = ic.m.str[:4]
print('8h IC by year: ' + ' '.join(f'{y}:{v:.3f}' for y, v in ic.groupby('y').ic.mean().items()) + f' mean {ic.ic.mean():.3f}', flush=True)
P8 = np.full((T, N), np.nan, dtype=np.float32); okp = np.isfinite(pred)
P8[dec[tab.dec.values[okp]], tab.sym.values[okp]] = pred[okp]
np.save(f'out_exp17_pred_8h{SUF}.npy', P8); del tab; gc.collect()

P3, P7 = np.load('out_exp6_pred_h3.npy'), np.load('out_exp6_pred_h7.npy')
instr = json.load(open('instruments.json')); launch_h = np.full(N, np.nan)
for j, s in enumerate(d.cols):
    mm = instr.get(str(s))
    if mm and mm.get('launch'):
        launch_h[j] = mm['launch'] / 1000 / 3600
MAJ5 = [s for s in ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT'] if s in d.cols]; jm = np.array([int(np.flatnonzero(d.cols == s)[0]) for s in MAJ5])


def sl_listing(dd, i, mask):
    age = np.maximum(dd.age[i].astype(float), np.where(np.isfinite(launch_h), dd.hh[i] - launch_h, -np.inf)); t = dd.t24[i]
    cand = (age >= 24 * 7) & (age <= 24 * 90) & dd.valid[i] & np.isfinite(t) & (t >= 2e7) & (dd.age[i] >= 72); cand[jm] = False
    r24 = dd.cff[i] / dd.cff[i - 24] - 1; cand &= ~(r24 > 0.30)
    idx = np.flatnonzero(cand); w = np.zeros(N)
    if len(idx) == 0:
        return w
    idx = idx[np.argsort(-t[idx])][:30]; ws = min(1.0 / len(idx), 0.05); w[idx] = -ws; w[jm] += ws * len(idx) / len(jm)
    return w


def sl_core(dd, i, mask):
    m = dd.universe(i, min_age_h=720, min_turn=1e7, top_n=100); return quantile_ls(core_signal(dd, i, m, 336), 0.2)


def make_ml(P, top=0.3, hold_last=False):
    last = {'w': np.zeros(N)}
    def f(dd, i, mask):
        s = P[i]
        if not np.isfinite(s).any():
            return last['w'] if hold_last else np.zeros(N)
        m = dd.universe(i, min_age_h=720, min_turn=1e7, top_n=150)
        w = quantile_ls(np.where(m & np.isfinite(s), s, np.nan), top); last['w'] = w
        return w
    return f


def combo(parts):
    def fn(dd, i, mask):
        w = np.zeros(N)
        for f in parts:
            w += f(dd, i, mask)
        return w / len(parts)
    return fn


def go(label, fn, **kw):
    t1 = time.time(); d.start_i = int(d.idx.searchsorted(pd.Timestamp(OOS, tz='UTC')))
    args = dict(reb_h=24, band=0.3, smooth=0.5, fee_bps=5.5, maker_share=0.7, slip_fn=slip_model(scale=0.3), gross=1.0, pos_cap=0.02, cap_short=0.015,
                cap_exempt=MAJ5, cap_exempt_val=0.30, uni_kwargs=dict(min_age_h=0, min_turn=0, top_n=0), verbose=False, label=label,
                vol_target=0.006, max_lev=3.0, max_gross_x=2.0)
    args.update(kw)
    m, eq = run(d, fn, **args); by = m['by_year']
    print(f'{label:<50} Sh {m["sharpe"]:5.2f} CAGR {m["cagr"]*100:6.1f}% dvol {m["dvol"]*100:.2f}% MDD {m["mdd"]*100:6.1f}% wd {m["worst_day"]*100:5.1f}% turn {m["turnover_x"]:4.0f}x cost {(m["fees"]+m["slip"])*100:.1f}% | '
          + ' '.join(f'{y}:{v[1]:.1f}' for y, v in by.items()) + f' ({time.time()-t1:.0f}s)', flush=True)


ML8 = make_ml(P8, 0.3)
go('8h ML alone, reb 8h band0.3 smooth0.5 maker70', ML8, reb_h=8, reb_offset=DEC_PHASE)
go('8h ML alone, reb 8h band0.5 smooth0.7', ML8, reb_h=8, reb_offset=DEC_PHASE, band=0.5, smooth=0.7)
go('daily v2 book (ref)', combo([sl_listing, sl_core, make_ml(P3), make_ml(P7)]))
print('DONE')
