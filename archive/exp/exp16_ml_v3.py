#!/usr/bin/env python3
"""Experiment 16: ML v3 — richer features + target variants, walk-forward as exp6, then portfolio impact.
New features vs v2: order-flow (taker-buy ratio 24h/72h/168h), trade-count activity (24h vs 30d), realised
skew 7d, cross-sectional market regime (BTC 30d vol & return, dispersion of 24h returns, breadth), proper
premium z-score, funding change (24h vs 168h). Variants: (r) regression on clipped y (as v2),
(k) regression on cross-sectional rank of y (robust to squeezes). Horizons 3 and 7 days.
Outputs out_exp16_pred_h{H}_{variant}.npy panels + IC by year + portfolio Sharpe vs v2 panels."""
import os, time, json, numpy as np, pandas as pd, lightgbm as lgb
from bt import Data, run, quantile_ls, slip_model
from strategies import core_signal

START, END, OOS = '2021-01-01', '2026-08-31 23:00', '2022-01-01'
d = Data(start=START, end=END, min_turn_ever=5e6, load_hl=True, load_premium=True)
T, N = d.ret.shape
PAN = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'panels')
cols = [str(c) for c in d.cols]
s0 = pd.Timestamp(START, tz='UTC') - pd.Timedelta(days=60)
tbq = pd.read_parquet(os.path.join(PAN, 'tbq.parquet'), columns=cols).loc[s0:pd.Timestamp(END, tz='UTC')].values.astype(np.float32)
qv = pd.read_parquet(os.path.join(PAN, 'qvol.parquet'), columns=cols).loc[s0:pd.Timestamp(END, tz='UTC')].values.astype(np.float32)
ntr = pd.read_parquet(os.path.join(PAN, 'ntrades.parquet'), columns=cols).loc[s0:pd.Timestamp(END, tz='UTC')].values.astype(np.float32)
assert tbq.shape == (T, N)
jbtc = int(np.flatnonzero(d.cols == 'BTCUSDT')[0])
dec = np.flatnonzero(d.hh % 24 == 0)
dec = dec[(dec >= 24 * 100) & (dec < T - 24 * 8)]
D = len(dec)
cff, fund, t24, prem, high, low = d.cff, d.fund, d.t24, d.prem, d.high, d.low
t0 = time.time()


def roll(fn, arr, k, dtype=np.float32):
    out = np.full((D, N), np.nan, dtype=dtype)
    for n, i in enumerate(dec):
        out[n] = fn(arr[i - k + 1:i + 1])
    return out


F = {}
for k in (24, 72, 168, 336, 720, 1440, 2160):
    F[f'r{k}'] = np.log(cff[dec] / cff[dec - k]).astype(np.float32)
F['r720_skip24'] = np.log(cff[dec - 24] / cff[dec - 720]).astype(np.float32)
F['r168_skip24'] = np.log(cff[dec - 24] / cff[dec - 168]).astype(np.float32)
btc168 = np.log(cff[dec, jbtc] / cff[dec - 168, jbtc])[:, None]; btc720 = np.log(cff[dec, jbtc] / cff[dec - 720, jbtc])[:, None]
F['rel168'] = F['r168'] - btc168; F['rel720'] = F['r720'] - btc720
F['vol168'] = roll(lambda a: np.nanstd(a, 0), d.ret, 168); F['vol720'] = roll(lambda a: np.nanstd(a, 0), d.ret, 720)
F['volratio'] = F['vol168'] / F['vol720']
for k in (24, 72, 168, 720):
    F[f'fund{k}'] = roll(lambda a: np.nansum(a, 0, dtype=np.float64), fund, k)
F['fund_chg'] = F['fund24'] * 7 - F['fund168']
F['prem8'] = roll(lambda a: np.nanmean(a.astype(np.float64), 0), prem, 8)
F['prem24'] = roll(lambda a: np.nanmean(a.astype(np.float64), 0), prem, 24)
F['prem168'] = roll(lambda a: np.nanmean(a.astype(np.float64), 0), prem, 168)
F['premz'] = (F['prem8'] - F['prem168']) / (roll(lambda a: np.nanstd(a.astype(np.float64), 0), prem, 168) + 1e-9)
F['logturn'] = np.log(np.where(t24[dec] > 0, t24[dec], np.nan)).astype(np.float32)
tavg30 = roll(lambda a: np.nanmean(a[::24].astype(np.float64), 0), t24, 24 * 30)
F['turnratio'] = np.log(t24[dec] / tavg30).astype(np.float32)
F['hl168'] = np.log(roll(lambda a: np.nanmax(a, 0), high, 168) / roll(lambda a: np.nanmin(a, 0), low, 168))
F['dist_hi720'] = np.log(cff[dec] / roll(lambda a: np.nanmax(a, 0), high, 720))
F['dist_lo720'] = np.log(cff[dec] / roll(lambda a: np.nanmin(a, 0), low, 720))
F['age_d'] = (d.age[dec] / 24.0).astype(np.float32)
beta = np.full((D, N), np.nan, dtype=np.float32)
for n, i in enumerate(dec):
    R = d.ret[i - 719:i + 1]; rb = R[:, jbtc]; ok = np.isfinite(rb)
    Rm = np.where(np.isfinite(R[ok]), R[ok], 0.0); rbb = rb[ok]; vb = rbb.var()
    beta[n] = ((Rm - Rm.mean(0)) * (rbb - rbb.mean())[:, None]).mean(0) / vb if vb > 0 else np.nan
F['beta720'] = beta
# --- NEW: order flow, activity, skew, regime
for k in (24, 72, 168):
    num = roll(lambda a: np.nansum(a, 0, dtype=np.float64), tbq, k); den = roll(lambda a: np.nansum(a, 0, dtype=np.float64), qv, k)
    F[f'tbr{k}'] = (num / np.where(den > 0, den, np.nan) - 0.5).astype(np.float32)
ntr24 = roll(lambda a: np.nansum(a, 0, dtype=np.float64), ntr, 24)
ntr30 = roll(lambda a: np.nansum(a, 0, dtype=np.float64), ntr, 24 * 30) / 30.0
F['ntr_ratio'] = np.log(ntr24 / np.where(ntr30 > 0, ntr30, np.nan)).astype(np.float32)
F['log_ntr'] = np.log(np.where(ntr24 > 0, ntr24, np.nan)).astype(np.float32)
def _skew(a):
    a = np.where(np.isfinite(a), a, 0.0); m = a.mean(0); s = a.std(0) + 1e-12
    return (((a - m) / s) ** 3).mean(0)
F['skew168'] = roll(_skew, d.ret, 168)
# regime (same value for all names on a day): trees can interact with it
btc_vol30 = np.array([np.nanstd(d.ret[i - 719:i + 1, jbtc]) for i in dec], dtype=np.float32)
F['reg_btcvol'] = np.repeat(btc_vol30[:, None], N, 1)
F['reg_btcret'] = np.repeat(btc720.astype(np.float32), N, 1)
disp = np.nanstd(np.where(np.isfinite(F['r24']), F['r24'], np.nan), 1, keepdims=True).astype(np.float32)
F['reg_disp24'] = np.repeat(disp, N, 1)
breadth = np.nanmean(np.where(np.isfinite(F['r24']), (F['r24'] > 0).astype(np.float32), np.nan), 1, keepdims=True).astype(np.float32)
F['reg_breadth'] = np.repeat(breadth, N, 1)
FEATSET = os.environ.get('FEATSET', 'all')          # 'bybit' = only features computable from Bybit klines/funding/premium
if FEATSET == 'bybit':
    for k in [k for k in F if k.startswith('tbr') or k.startswith('ntr') or k == 'log_ntr']:
        F.pop(k)
VARIANTS = os.environ.get('VARIANTS', 'r,k').split(',')
SUFFIX = os.environ.get('SUFFIX', '')
feat_names = list(F)
del beta
d.high = d.low = d.prem = None; high = low = prem = None; tbq = qv = ntr = None
import gc; gc.collect()
print(f'features: {len(feat_names)} in {time.time()-t0:.0f}s', flush=True)

# --- targets (funding-aware) and masks
fwd = {}
for hz in (3, 7):
    k = 24 * hz
    fp = np.zeros((D, N))
    for n, i in enumerate(dec):
        fp[n] = np.nansum(fund[i + 1:i + k + 1], 0, dtype=np.float64)
    fwd[hz] = np.log(cff[dec + k] / cff[dec]) - fp
masks = np.zeros((D, N), dtype=bool); d.prev_mask = None
for n, i in enumerate(dec):
    masks[n] = d.universe(i, min_age_h=720, min_turn=1e7, top_n=150)
rows = []
for n in range(D):
    m = masks[n]
    if m.sum() < 20:
        continue
    X = np.column_stack([F[f][n][m] for f in feat_names]).astype(np.float32)
    Xr = np.full(X.shape, np.nan, dtype=np.float32)
    for c in range(X.shape[1]):
        col = X[:, c]; ok = np.isfinite(col)
        if ok.sum() > 2:
            rr = np.empty(ok.sum()); rr[np.argsort(col[ok])] = np.arange(ok.sum()); Xr[ok, c] = rr / (ok.sum() - 1) - 0.5
    df = pd.DataFrame(Xr, columns=feat_names); df['day'] = n; df['sym'] = np.flatnonzero(m)
    for hz in fwd:
        v = fwd[hz][n][m]; y = v - np.nanmean(v)
        df[f'y{hz}'] = y
        ok = np.isfinite(y); rk = np.full(len(y), np.nan)
        if ok.sum() > 2:
            rr = np.empty(ok.sum()); rr[np.argsort(y[ok])] = np.arange(ok.sum()); rk[ok] = rr / (ok.sum() - 1) - 0.5
        df[f'k{hz}'] = rk
    rows.append(df)
tab = pd.concat(rows, ignore_index=True); del rows
tab['date'] = d.idx[dec[tab.day.values]]
print(f'table {tab.shape} ({time.time()-t0:.0f}s)', flush=True)
params = dict(objective='regression', learning_rate=0.03, num_leaves=31, min_data_in_leaf=200, feature_fraction=0.7,
              bagging_fraction=0.7, bagging_freq=1, lambda_l2=10.0, verbose=-1, num_threads=2, seed=1)
months = sorted(set(tab.date.dt.strftime('%Y-%m'))); first_test = months.index('2022-01'); mth_col = tab.date.dt.strftime('%Y-%m').values
PANELS = {}
for hz in (3, 7):
    for variant in VARIANTS:
        ycol = f'y{hz}' if variant == 'r' else f'k{hz}'
        okrow = np.isfinite(tab[ycol].values)
        pred = np.full(len(tab), np.nan, dtype=np.float32); ics = []; imp = np.zeros(len(feat_names))
        for kk in range(first_test, len(months)):
            mth = months[kk]; test = (mth_col == mth) & okrow
            if test.sum() == 0:
                continue
            t_start = tab.date.values[test].min()
            train = (tab.date.values < (t_start - np.timedelta64(hz + 7, 'D'))) & okrow
            Xtr, ytr = tab.loc[train, feat_names].values, tab.loc[train, ycol].values
            if variant == 'r':
                lo, hi = np.nanpercentile(ytr, [0.5, 99.5]); ytr = np.clip(ytr, lo, hi)
            mdl = lgb.train(params, lgb.Dataset(Xtr, ytr, free_raw_data=True), num_boost_round=400)
            p = mdl.predict(tab.loc[test, feat_names].values); pred[test] = p; imp += mdl.feature_importance('gain')
            sub = tab.loc[test, ['day', f'y{hz}']].copy(); sub['p'] = p
            ics.append((mth, sub.groupby('day').apply(lambda g: g[[f'y{hz}', 'p']].corr(method='spearman').iloc[0, 1]).mean()))
        ic = pd.DataFrame(ics, columns=['m', 'ic']); ic['y'] = ic.m.str[:4]
        print(f'h{hz} variant {variant}: IC by year ' + ' '.join(f'{y}:{v:.3f}' for y, v in ic.groupby('y').ic.mean().items()) + f'  mean {ic.ic.mean():.3f} ({time.time()-t0:.0f}s)', flush=True)
        fi = pd.Series(imp, index=feat_names).sort_values(ascending=False); print('   top feats: ' + ', '.join(f'{k} {v/fi.sum():.3f}' for k, v in fi.head(10).items()), flush=True)
        P = np.full((T, N), np.nan, dtype=np.float32); okp = np.isfinite(pred)
        P[dec[tab.day.values[okp]], tab.sym.values[okp]] = pred[okp]
        np.save(f'out_exp16_pred_h{hz}_{variant}{SUFFIX}.npy', P); PANELS[(hz, variant)] = P

# --- portfolio impact: exp13b P0 structure with ML panels swapped
instr = json.load(open('instruments.json'))
launch_h = np.full(N, np.nan)
for j, s in enumerate(d.cols):
    mm = instr.get(str(s))
    if mm and mm.get('launch'):
        launch_h[j] = mm['launch'] / 1000 / 3600
MAJ5 = [s for s in ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT'] if s in d.cols]
jm = np.array([int(np.flatnonzero(d.cols == s)[0]) for s in MAJ5])


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


def make_ml(P, top=0.3):
    def f(dd, i, mask):
        m = dd.universe(i, min_age_h=720, min_turn=1e7, top_n=150)
        return quantile_ls(np.where(m & np.isfinite(P[i]), P[i], np.nan), top)
    return f


def combo(parts):
    def fn(dd, i, mask):
        w = np.zeros(N)
        for f in parts:
            w += f(dd, i, mask)
        return w / len(parts)
    return fn


def go(label, fn):
    t1 = time.time(); d.start_i = int(d.idx.searchsorted(pd.Timestamp(OOS, tz='UTC')))
    m, eq = run(d, fn, reb_h=24, band=0.3, smooth=0.5, fee_bps=5.5, maker_share=0.7, slip_fn=slip_model(scale=0.3), gross=1.0,
                pos_cap=0.02, cap_short=0.015, cap_exempt=MAJ5, cap_exempt_val=0.30, uni_kwargs=dict(min_age_h=0, min_turn=0, top_n=0),
                verbose=False, label=label, vol_target=0.006, max_lev=3.0, max_gross_x=2.0)
    by = m['by_year']
    print(f'{label:<44} Sh {m["sharpe"]:5.2f} CAGR {m["cagr"]*100:6.1f}% dvol {m["dvol"]*100:.2f}% MDD {m["mdd"]*100:6.1f}% wd {m["worst_day"]*100:5.1f}% turn {m["turnover_x"]:4.0f}x | '
          + ' '.join(f'{y}:{v[1]:.1f}' for y, v in by.items()) + f' ({time.time()-t1:.0f}s)', flush=True)
    return m


P3v2, P7v2 = np.load('out_exp6_pred_h3.npy'), np.load('out_exp6_pred_h7.npy')
go('v2 ML (ref P0): A+C+D3+D7', combo([sl_listing, sl_core, make_ml(P3v2), make_ml(P7v2)]))
if 'r' in VARIANTS:
    go(f'v3r{SUFFIX}: A+C+D3r+D7r', combo([sl_listing, sl_core, make_ml(PANELS[(3, 'r')]), make_ml(PANELS[(7, 'r')])]))
if 'k' in VARIANTS:
    go(f'v3k{SUFFIX}: A+C+D3k+D7k', combo([sl_listing, sl_core, make_ml(PANELS[(3, 'k')]), make_ml(PANELS[(7, 'k')])]))
    go(f'v3k{SUFFIX} ML only (D3k+D7k)', combo([make_ml(PANELS[(3, 'k')]), make_ml(PANELS[(7, 'k')])]))
if 'r' in VARIANTS and 'k' in VARIANTS:
    go('v3 r+k ensemble: A+C+4 ML', combo([sl_listing, sl_core, make_ml(PANELS[(3, 'r')]), make_ml(PANELS[(7, 'r')]), make_ml(PANELS[(3, 'k')]), make_ml(PANELS[(7, 'k')])]))
go('v2 ML only (D3+D7)', combo([make_ml(P3v2), make_ml(P7v2)]))
print('DONE')
