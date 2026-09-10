#!/usr/bin/env python3
"""Experiment 19d: is the 8h-combination fragility a small-account artefact (positions ~$40-120 vs min_trade $15,
band and dust rules dominating)? Re-run the smooth x w8 grid at start_eq=100k (min_trade negligible), at 10k with
min_trade 5, and at 10k with concentrated sleeves (ML q20, core q20, ml8 q20)."""
import time, json, numpy as np, pandas as pd
from bt import Data, run, quantile_ls, slip_model
from strategies import core_signal
START, END, OOS = '2021-01-01', '2026-08-31 23:00', '2022-01-01'
d = Data(start=START, end=END, min_turn_ever=5e6, load_hl=True, load_premium=True); T, N = d.ret.shape
P3, P7, P8 = np.load('out_exp6_pred_h3.npy'), np.load('out_exp6_pred_h7.npy'), np.load('out_exp17_pred_8h_p0.npy')
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
def sl_core(q=0.2):
    def f(dd, i, mask):
        m = dd.universe(i, min_age_h=720, min_turn=1e7, top_n=100); return quantile_ls(core_signal(dd, i, m, 336), q)
    return f
def make_ml(P, top=0.3):
    def f(dd, i, mask):
        m = dd.universe(i, min_age_h=720, min_turn=1e7, top_n=150); return quantile_ls(np.where(m & np.isfinite(P[i]), P[i], np.nan), top)
    return f
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
def book(w8, qd=0.3, q8=0.3, qc=0.2): return combo([Held(sl_listing), Held(sl_core(qc)), Held(make_ml(P3, qd)), Held(make_ml(P7, qd)), make_ml(P8, q8)], [1, 1, 1, 1, w8])
rows = []
def go(label, fn, **kw):
    t1 = time.time(); d.start_i = int(d.idx.searchsorted(pd.Timestamp(OOS, tz='UTC')))
    args = dict(reb_h=8, reb_offset=0, band=0.3, smooth=0.5, fee_bps=5.5, maker_share=0.7, slip_fn=slip_model(scale=0.3), gross=1.0, pos_cap=0.02, cap_short=0.015,
                cap_exempt=MAJ5, cap_exempt_val=0.30, uni_kwargs=dict(min_age_h=0, min_turn=0, top_n=0), verbose=False, label=label, vol_target=0.006, max_lev=3.0, max_gross_x=2.0)
    args.update(kw); m, eq = run(d, fn, **args); by = m['by_year']
    rows.append(dict(label=label, sharpe=m['sharpe'], cagr=m['cagr'], mdd=m['mdd'], turnover=m['turnover_x'], npos=m['avg_npos']))
    print(f'{label:<44} Sh {m["sharpe"]:5.2f} CAGR {m["cagr"]*100:6.1f}% MDD {m["mdd"]*100:6.1f}% wd {m["worst_day"]*100:5.1f}% turn {m["turnover_x"]:4.0f}x npos {m["avg_npos"]:.0f} | '
          + ' '.join(f'{y}:{v[1]:.1f}' for y, v in by.items()) + f' ({time.time()-t1:.0f}s)', flush=True)
def grid(tag, **kw):
    sh = []
    for sm in (0.4, 0.5, 0.6):
        for w8 in (1.5, 2.0, 2.5):
            m = go(f'{tag} smooth{sm} w8={w8}', book(w8, **{k: v for k, v in kw.items() if k in ('qd', 'q8', 'qc')}), smooth=sm, **{k: v for k, v in kw.items() if k not in ('qd', 'q8', 'qc')})
    s = pd.Series([r['sharpe'] for r in rows[-9:]]); print(f'>>> {tag}: Sharpe mean {s.mean():.2f} min {s.min():.2f} max {s.max():.2f} std {s.std():.2f}\n', flush=True)
grid('EQ100k', start_eq=100000.0)
grid('EQ10k min_trade5', min_trade=5.0)
grid('EQ10k concentrated q20/q20/q20', qd=0.2, q8=0.2, qc=0.2)
pd.DataFrame(rows).to_csv('out_exp19d_fragility.csv', index=False); print('DONE')
