#!/usr/bin/env python3
"""Experiment (owner's idea, 2026-10-28): a SPECIALIST model for the 'jumpers' — the most volatile third of the ML universe — that predicts
which way they jump; compared with the current 8h model on exactly the same names and decision times.

Universe at each 8h decision (00/08/16 UTC, as the 8h model): top-150 by 24h turnover, >= 90 days old, turnover >= 1e7; 'jumpers' = top third
by 168h hourly-return std. Features: the live 8h feature set (v4/ml_features_8h) + jump features available live on Bybit (no taker-buy /
trade counts): max30 (largest daily log return in 30 d), min30, skew720 (hourly), r4, turn_shock (24h turnover / 30-day mean), dist_hi30
(close vs 30-day high). Target: forward 8h and 24h log return minus funding paid, demeaned within the jumpers. Walk-forward: retrained on the
1st of every month from 2022-07, expanding window, 2-day gap. LightGBM as exp17b.
Outputs: IC (Spearman) by year for specialist vs current 8h model (walk-forward panel out_exp17_pred_8h_oct_noconst.npy); sign hit-rate on
big moves (|y24| > 2 sigma); L/S within the jumpers (top / bottom 30 %, 8h hold, 6 bp one-way cost) Sharpe by year.
Run: python3 research/exp_hivol_model.py  (~1 h; log logs/hv_model.log)"""
import os, sys, time, gc, numpy as np, pandas as pd, lightgbm as lgb
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'v4'))
from bt import Data
from ml_features_8h import compute_features_8h, FEATURES_8H, rank_norm, NEED_H

EXTRA = ['max30', 'min30', 'skew720', 'r4', 'turn_shock', 'dist_hi30']
FEATS = list(FEATURES_8H) + EXTRA
COST = 0.0006


def extras(cff, t24, i, js):
    sub = cff[i - 720:i + 1][:, js]
    lr = np.log(sub[1:] / sub[:-1])
    dly = np.log(sub[24::24] / sub[:-24:24])
    m = np.nanmean(lr, 0); sd = np.nanstd(lr, 0)
    skew = np.nanmean(((lr - m) / np.where(sd > 0, sd, np.nan)) ** 3, 0)
    return np.column_stack([np.nanmax(dly, 0), np.nanmin(dly, 0), skew, np.log(cff[i, js] / cff[i - 4, js]),
                            t24[i, js] / np.nanmean(t24[i - 720:i + 1:24, js], 0), cff[i, js] / np.nanmax(sub, 0) - 1.0])


def main():
    t0 = time.time()
    d = Data(start='2021-01-01', end='2026-10-26 23:00', min_turn_ever=5e6, load_hl=True, load_premium=True)
    T, N = d.ret.shape; jbtc = int(np.flatnonzero(d.cols == 'BTCUSDT')[0])
    P8 = np.load(os.path.join(ROOT, 'out_exp17_pred_8h_oct_noconst.npy'), mmap_mode='r'); assert P8.shape == (T, N), P8.shape
    vol = pd.DataFrame(d.cff).pct_change().rolling(168, min_periods=72).std().values
    dec = np.flatnonzero(d.hh % 8 == 0); dec = dec[(dec >= max(NEED_H, 721)) & (dec < T - 25)]
    rows = []; d.prev_mask = None
    for n, i in enumerate(dec):
        m = d.universe(i, min_age_h=90 * 24, min_turn=1e7, top_n=150)
        js = np.flatnonzero(m & np.isfinite(vol[i]))
        if len(js) < 45:
            continue
        js = js[np.argsort(vol[i, js])[int(len(js) * 2 / 3):]]              # the most volatile third
        sl = slice(i - NEED_H + 1, i + 1)
        F = compute_features_8h(d.cff[sl], d.fund[sl], d.t24[sl], d.prem[sl], d.high[sl], d.low[sl], d.age[i].astype(float), jbtc, (d.hh[i] + 1) % 24)
        X = np.column_stack([np.column_stack([F[f][js] for f in FEATURES_8H]), extras(d.cff, d.t24, i, js)]).astype(np.float32)
        df = pd.DataFrame(rank_norm(X), columns=FEATS); df['i'] = i; df['j'] = js
        for hz, k in (('y8', 8), ('y24', 24)):
            fp = np.nansum(d.fund[i + 1:i + k + 1, js], 0, dtype=np.float64)
            y = np.log(d.cff[i + k, js] / d.cff[i, js]) - fp; df[hz] = (y - np.nanmean(y)).astype(np.float32); df[hz + '_raw'] = y.astype(np.float32)
        df['p_cur'] = np.asarray(P8[i, js], dtype=np.float32)
        rows.append(df)
        if n % 1000 == 0:
            print(f'  features {n}/{len(dec)} ({time.time() - t0:.0f}s)', flush=True)
    tab = pd.concat(rows, ignore_index=True); del rows; gc.collect()
    tab['t'] = d.idx[tab.i.values]; print('table', tab.shape, flush=True)
    params = dict(objective='regression', learning_rate=0.03, num_leaves=31, min_data_in_leaf=300, feature_fraction=0.7, bagging_fraction=0.7,
                  bagging_freq=1, lambda_l2=10.0, verbose=-1, num_threads=3, seed=1)
    tab['p_spec'] = np.nan
    for ms in pd.date_range('2022-07-01', '2026-10-01', freq='MS', tz='UTC'):
        tr = (tab.t < ms - pd.Timedelta(days=2)) & np.isfinite(tab.y24); te = (tab.t >= ms) & (tab.t < ms + pd.offsets.MonthBegin(1))
        if te.sum() == 0:
            continue
        y = tab.loc[tr, 'y24'].values; lo, hi = np.nanpercentile(y, [0.5, 99.5])
        mdl = lgb.train(params, lgb.Dataset(tab.loc[tr, FEATS].values, np.clip(y, lo, hi)), num_boost_round=300)
        tab.loc[te, 'p_spec'] = mdl.predict(tab.loc[te, FEATS].values)
        print(f'  {ms.date()} trained on {tr.sum()} rows ({time.time() - t0:.0f}s)', flush=True)
    fi = pd.Series(mdl.feature_importance('gain'), index=FEATS).sort_values(ascending=False)
    print('last model top features: ' + ', '.join(f'{k} {v / fi.sum():.3f}' for k, v in fi.head(12).items()))
    ev = tab[np.isfinite(tab.p_spec) & np.isfinite(tab.p_cur)].copy(); ev['year'] = ev.t.dt.year
    out = []
    for yr, g in list(ev.groupby('year')) + [('all', ev)]:
        r = dict(year=yr, n_dec=g.i.nunique())
        for p in ('p_spec', 'p_cur'):
            for hz in ('y8', 'y24'):
                r[f'IC_{p[2:]}_{hz}'] = round(g.groupby('i').apply(lambda x: x[[p, hz]].corr(method='spearman').iloc[0, 1]).mean(), 4)
            big = g[np.abs(g.y24) > 2 * g.y24.std()]
            r[f'hit_big_{p[2:]}'] = round((np.sign(big[p] - g[p].median()) == np.sign(big.y24)).mean(), 3)
            # L/S within jumpers: top / bottom 30 % by prediction, 8h hold, equal weight, cost on full turnover each decision
            def ls(x):
                q = x[p].rank(pct=True); lng = x.loc[q > 0.7, 'y8_raw'].mean(); sht = x.loc[q <= 0.3, 'y8_raw'].mean()
                return 0.5 * (lng - sht) - 2 * COST
            pnl = g.groupby('i').apply(ls)
            r[f'LS_{p[2:]}_ann'] = round(pnl.mean() * 3 * 365 * 100, 1); r[f'LS_{p[2:]}_sh'] = round(pnl.mean() / pnl.std() * np.sqrt(3 * 365), 2)
        out.append(r)
    res = pd.DataFrame(out); pd.set_option('display.width', 250); print(res.to_string(index=False))
    res.to_csv(os.path.join(ROOT, 'research', 'phase4_out', 'hivol_model.csv'), index=False)
    tab[['i', 'j', 'p_spec']].dropna().to_parquet(os.path.join(ROOT, 'research', 'phase4_out', 'hivol_model_preds.parquet'))
    print('DONE', f'{time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
