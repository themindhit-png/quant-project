#!/usr/bin/env python3
"""Experiment 23a — corrections requested by Opus round 2, on the final book F (listing + core + age-free ML3/ML7 (>=90d)
+ ML8 q.2 age180 inv-vol w2, band .5): (1) F under cost x1.5 / x2 and maker 50% / taker; (2) VT de-levering by
realised gross vs intended lev (clip share, lev distribution); (3) ML8 yearly attribution (gross price alpha, funding,
costs) to characterise the decay; (4) F with ml8 weight 1 and 0 (kill-rule scenarios); (5) Deflated Sharpe for F."""
import time, json, math, glob, numpy as np, pandas as pd
from scipy.stats import norm, skew, kurtosis
from bt import Data, run, quantile_ls, slip_model
from strategies import core_signal
from betautil import rolling_std
START, END, OOS = '2021-01-01', '2026-08-31 23:00', '2022-01-01'
d = Data(start=START, end=END, min_turn_ever=5e6, load_hl=False, load_premium=False); T, N = d.ret.shape
P8 = np.load('out_exp17_pred_8h_p0.npy'); Q3, Q7 = np.load('out_exp21c_pred_h3_noage.npy'), np.load('out_exp21c_pred_h7_noage.npy')
VOL = rolling_std(d.ret, 168, 72)
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
ML3n, ML7n = make_ml(Q3, min_age_h=90 * 24), make_ml(Q7, min_age_h=90 * 24); ML8 = make_ml(P8, 0.2, 180 * 24, True)
class Held:
    def __init__(self, fn): self.fn = fn; self.w = np.zeros(N)
    def __call__(self, dd, i, mask):
        if dd.hh[i] % 24 == 0: self.w = self.fn(dd, i, mask)
        return self.w
def combo(parts, weights):
    def fn(dd, i, mask):
        w = np.zeros(N)
        for f, s in zip(parts, weights): w += s * f(dd, i, mask)
        return w / sum(weights)
    return fn
def F(w8=2.0): return combo([Held(sl_listing), Held(sl_core), Held(ML3n), Held(ML7n), ML8], [1, 1, 1, 1, w8]) if w8 > 0 else combo([Held(sl_listing), Held(sl_core), Held(ML3n), Held(ML7n)], [1, 1, 1, 1])
BASE8 = dict(reb_h=8, reb_offset=0, band=0.5, smooth=0.5, fee_bps=5.5, maker_share=0.7, slip_fn=slip_model(scale=0.3), gross=1.0, pos_cap=0.02, cap_short=0.015,
             cap_exempt=MAJ5, cap_exempt_val=0.30, uni_kwargs=dict(min_age_h=0, min_turn=0, top_n=0), verbose=False, vol_target=0.006, max_lev=3.0, max_gross_x=2.0)
def sh(x): x = np.asarray(x, float); return float(x.mean() / x.std(ddof=1) * math.sqrt(365)) if len(x) > 30 and x.std(ddof=1) > 0 else float('nan')
SER = {}
def go(label, fn, keep=None, **kw):
    t1 = time.time(); d.start_i = int(d.idx.searchsorted(pd.Timestamp(OOS, tz='UTC'))); args = dict(BASE8); args.update(kw); args['label'] = label
    m, eq, hp = run(d, fn, ret_diag=True, **args); de = eq.groupby(eq.index.floor('D')).last(); r = de.pct_change().dropna()
    if keep: SER[keep] = r
    L = pd.DataFrame(m['lev_hist'], columns=['i', 'lev', 'gross', 'npos'])
    print(f'{label:<58} Sh {m["sharpe"]:5.2f} | SEL {sh(r[:"2024-12-31"]):5.2f} CONF {sh(r["2025-01-01":]):5.2f} | CAGR {m["cagr"]*100:5.1f}% dvol {m["dvol"]*100:.2f}% MDD {m["mdd"]*100:6.1f}% wd {m["worst_day"]*100:5.1f}% turn {m["turnover_x"]:4.0f}x cost {(m["fees"]+m["slip"])*100:4.1f}% | '
          f'lev mean {L.lev.mean():.2f} p5 {L.lev.quantile(.05):.2f} p95 {L.lev.quantile(.95):.2f} clip {m["clip_frac"]*100:.1f}% | ' + ' '.join(f'{y}:{v[1]:.1f}' for y, v in m['by_year'].items()) + f' ({time.time()-t1:.0f}s)', flush=True)
    return m
print('\n=== (1)+(2) F: cost scenarios and VT de-levering ===', flush=True)
go('F (realised de-lever, default)', F(), keep='F')
go('F (intended de-lever lev*gross, old)', F(), vt_delever='intended')
go('F cost x1.5', F(), fee_bps=8.25, maker_fee_bps=3.0, slip_fn=slip_model(scale=0.45))
go('F cost x2.0', F(), fee_bps=11.0, maker_fee_bps=4.0, slip_fn=slip_model(scale=0.6))
go('F maker 50%', F(), maker_share=0.5, slip_fn=slip_model(scale=0.5))
go('F taker only', F(), maker_share=0.0, slip_fn=slip_model(scale=1.0))
go('F max_gross 1.7x eq (firm cap in equity units)', F(), max_gross_x=1.7)
print('\n=== (4) kill-rule scenarios: ml8 weight 1 and 0 ===', flush=True)
go('F w8=1', F(1.0), keep='F1'); go('F w8=0 (daily age-free book on 8h grid)', F(0.0), keep='F0')
go('F w8=0 on the daily engine', combo([sl_listing, sl_core, ML3n, ML7n], [1, 1, 1, 1]), keep='D', reb_h=24, band=0.3)
print('\n=== (3) ML8 standalone attribution by year (q.2, age180, inv-vol, band .5) ===', flush=True)
for y in range(2022, 2027):
    d.start_i = int(d.idx.searchsorted(pd.Timestamp(f'{y}-01-01', tz='UTC'))); end_i = int(d.idx.searchsorted(pd.Timestamp(f'{y+1}-01-01', tz='UTC'))) if y < 2026 else T
    args = dict(BASE8); args['label'] = f'ml8 {y}'
    m, eq, hp = run(d, ML8, ret_diag=True, attrib=True, **args)
    a = m['attrib']; eqs = eq.dropna(); eqs = eqs[eqs.index < d.idx[min(end_i, T - 1)]]
    r = eqs.groupby(eqs.index.floor('D')).last().pct_change().dropna()
    price = a.price.sum(); fund = a.funding.sum(); cost = a.cost.sum(); e0 = 10000.0
    print(f'  {y}: Sharpe (year) {sh(r):5.2f} | gross price {price/e0*100:+6.1f}% funding {fund/e0*100:+5.1f}% costs {-cost/e0*100:6.1f}% -> net {(price+fund-cost)/e0*100:+6.1f}% of start equity | '
          f'turnover {m["turnover_x"]:.0f}x  (attrib over the whole run from {y}-01-01; year-only numbers for the first year)', flush=True)
print('\n=== (5) Deflated Sharpe for F ===', flush=True)
trials = []
for f in glob.glob('out_exp19*.csv') + glob.glob('out_exp20*.csv') + glob.glob('out_exp22*.csv'):
    try:
        df = pd.read_csv(f); trials += list(df.sharpe.dropna().values) if 'sharpe' in df else []
    except Exception: pass
trials = np.array(trials); v = SER['F'].values; n = len(v); sd = v.mean() / v.std(ddof=1); S_ann = sd * math.sqrt(365)
g3, g4 = skew(v), kurtosis(v, fisher=False); se_ann = math.sqrt((1 + 0.5 * S_ann ** 2) / (n / 365))
print(f'  trials N={len(trials)} (std of annual Sharpe {trials.std(ddof=1):.2f}) | F Sharpe {S_ann:.2f} SE {se_ann:.2f} CI [{S_ann-1.96*se_ann:.2f}, {S_ann+1.96*se_ann:.2f}] skew {g3:+.2f} kurt {g4:.1f}')
for Ntr in (len(trials), 100, 400):
    var_tr = (trials.std(ddof=1) / math.sqrt(365)) ** 2; em = 0.5772156649
    sr0 = math.sqrt(var_tr) * ((1 - em) * norm.ppf(1 - 1 / Ntr) + em * norm.ppf(1 - 1 / (Ntr * math.e)))
    dsr = norm.cdf((sd - sr0) * math.sqrt(n - 1) / math.sqrt(1 - g3 * sd + (g4 - 1) / 4 * sd ** 2))
    print(f'  DSR(N={Ntr}): SR0 ann {sr0*math.sqrt(365):.2f} -> {dsr:.3f}')
pd.DataFrame(SER).to_csv('out_exp23a_daily_returns.csv'); print('DONE')
