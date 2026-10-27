#!/usr/bin/env python3
"""Experiment 21d — portfolio impact of the age-free ML (exp21c panels, universe >= 90d): ML3/ML7 standalone,
daily 4-sleeve book and the 5-sleeve R5 book with age-free ML3/ML7 (ML8 unchanged), vs the baseline panels."""
import time, json, os, numpy as np, pandas as pd
from bt import Data, run, quantile_ls, slip_model
from strategies import core_signal
START, END, OOS = '2021-01-01', '2026-08-31 23:00', '2022-01-01'
SUF = os.environ.get('SUF', '_noage'); AGE_ML = int(os.environ.get('MIN_AGE_D', '90')) * 24
d = Data(start=START, end=END, min_turn_ever=5e6, load_hl=True, load_premium=True); T, N = d.ret.shape
P3, P7, P8 = np.load('out_exp6_pred_h3.npy'), np.load('out_exp6_pred_h7.npy'), np.load('out_exp17_pred_8h_p0.npy')
Q3, Q7 = np.load(f'out_exp21c_pred_h3{SUF}.npy'), np.load(f'out_exp21c_pred_h7{SUF}.npy')
VOL = pd.DataFrame(d.ret).rolling(168, min_periods=72).std().values.astype(np.float32)
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
BASE8 = dict(reb_h=8, reb_offset=0, band=0.3, smooth=0.5, fee_bps=5.5, maker_share=0.7, slip_fn=slip_model(scale=0.3), gross=1.0, pos_cap=0.02, cap_short=0.015,
             cap_exempt=MAJ5, cap_exempt_val=0.30, uni_kwargs=dict(min_age_h=0, min_turn=0, top_n=0), verbose=False, vol_target=0.006, max_lev=3.0, max_gross_x=2.0)
BASE24 = dict(BASE8); BASE24.update(reb_h=24)
def go(label, fn, base=BASE8, **kw):
    t1 = time.time(); d.start_i = int(d.idx.searchsorted(pd.Timestamp(OOS, tz='UTC'))); args = dict(base); args.update(kw); args['label'] = label
    m, eq = run(d, fn, **args); by = m['by_year']
    print(f'{label:<52} Sh {m["sharpe"]:5.2f} CAGR {m["cagr"]*100:6.1f}% MDD {m["mdd"]*100:6.1f}% wd {m["worst_day"]*100:5.1f}% turn {m["turnover_x"]:4.0f}x | '
          + ' '.join(f'{y}:{v[1]:.1f}' for y, v in by.items()) + f' ({time.time()-t1:.0f}s)', flush=True)
ML8 = make_ml(P8, 0.3, 180 * 24, True)
print(f'\n=== age-free ML (universe >= {AGE_ML//24}d) vs baseline ===', flush=True)
go('ML3 baseline alone', make_ml(P3), BASE24); go(f'ML3{SUF} alone (uni>={AGE_ML//24}d)', make_ml(Q3, min_age_h=AGE_ML), BASE24)
go('ML7 baseline alone', make_ml(P7), BASE24); go(f'ML7{SUF} alone (uni>={AGE_ML//24}d)', make_ml(Q7, min_age_h=AGE_ML), BASE24)
go('ML3 baseline restricted to uni>=90d (no retrain)', make_ml(P3, min_age_h=AGE_ML), BASE24)
go('DAILY book baseline', combo([sl_listing, sl_core, make_ml(P3), make_ml(P7)], [1, 1, 1, 1]), BASE24)
go(f'DAILY book with ML{SUF}', combo([sl_listing, sl_core, make_ml(Q3, min_age_h=AGE_ML), make_ml(Q7, min_age_h=AGE_ML)], [1, 1, 1, 1]), BASE24)
go('R5 book baseline', combo([Held(sl_listing), Held(sl_core), Held(make_ml(P3)), Held(make_ml(P7)), ML8], [1, 1, 1, 1, 2]), BASE8)
go(f'R5 book with ML{SUF}', combo([Held(sl_listing), Held(sl_core), Held(make_ml(Q3, min_age_h=AGE_ML)), Held(make_ml(Q7, min_age_h=AGE_ML)), ML8], [1, 1, 1, 1, 2]), BASE8)
go(f'R5 book with ML{SUF}, no listing sleeve', combo([Held(sl_core), Held(make_ml(Q3, min_age_h=AGE_ML)), Held(make_ml(Q7, min_age_h=AGE_ML)), ML8], [1, 1, 1, 2]), BASE8)
print('DONE')
