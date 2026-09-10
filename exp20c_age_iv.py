#!/usr/bin/env python3
"""Experiment 20c: young-cohort exposure is the lottery. (A) does the same medicine (ML universes age>=180d,
inverse-vol weights) improve the PRODUCTION daily 4-sleeve book (daily engine)? grid over smooth.
(B) robustness of the 5-sleeve R5 book (ML8 age180+IV) to band, age threshold, q8, account size, min_trade.
(C) R7 = daily ML sleeves age180+IV + core IV + ML8 age180+IV on the smooth x w8 grid."""
import time, json, numpy as np, pandas as pd
from bt import Data, run, quantile_ls, slip_model
from strategies import core_signal
START, END, OOS = '2021-01-01', '2026-08-31 23:00', '2022-01-01'
d = Data(start=START, end=END, min_turn_ever=5e6, load_hl=True, load_premium=True); T, N = d.ret.shape
P3, P7, P8 = np.load('out_exp6_pred_h3.npy'), np.load('out_exp6_pred_h7.npy'), np.load('out_exp17_pred_8h_p0.npy')
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
def sl_core(iv=False, min_age_h=720):
    def f(dd, i, mask):
        m = dd.universe(i, min_age_h=min_age_h, min_turn=1e7, top_n=100); return quantile_ls(core_signal(dd, i, m, 336), 0.2, inv_vol=VOL[i] if iv else None)
    return f
def make_ml(P, top=0.3, min_age_h=720, min_turn=1e7, iv=False):
    def f(dd, i, mask):
        m = dd.universe(i, min_age_h=min_age_h, min_turn=min_turn, top_n=150)
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
A180 = 180 * 24
BASE8 = dict(reb_h=8, reb_offset=0, band=0.3, smooth=0.5, fee_bps=5.5, maker_share=0.7, slip_fn=slip_model(scale=0.3), gross=1.0, pos_cap=0.02, cap_short=0.015,
             cap_exempt=MAJ5, cap_exempt_val=0.30, uni_kwargs=dict(min_age_h=0, min_turn=0, top_n=0), verbose=False, vol_target=0.006, max_lev=3.0, max_gross_x=2.0)
BASE24 = dict(BASE8); BASE24.update(reb_h=24, reb_offset=0)
rows = []
def go(label, fn, base=BASE8, **kw):
    t1 = time.time(); d.start_i = int(d.idx.searchsorted(pd.Timestamp(OOS, tz='UTC'))); args = dict(base); args.update(kw); args['label'] = label
    m, eq = run(d, fn, **args); by = m['by_year']
    rows.append(dict(label=label, sharpe=m['sharpe'], cagr=m['cagr'], mdd=m['mdd'], worst_day=m['worst_day'], turnover=m['turnover_x'], **{f'y{y}': v[1] for y, v in by.items()}))
    print(f'{label:<50} Sh {m["sharpe"]:5.2f} CAGR {m["cagr"]*100:6.1f}% MDD {m["mdd"]*100:6.1f}% wd {m["worst_day"]*100:5.1f}% turn {m["turnover_x"]:4.0f}x cost {(m["fees"]+m["slip"])*100:.1f}% npos {m["avg_npos"]:.0f} | '
          + ' '.join(f'{y}:{v[1]:.1f}' for y, v in by.items()) + f' ({time.time()-t1:.0f}s)', flush=True)
    return m
def summary(tag, n):
    s = pd.Series([r['sharpe'] for r in rows[-n:]]); print(f'>>> {tag}: Sharpe mean {s.mean():.2f} min {s.min():.2f} max {s.max():.2f} std {s.std():.2f}\n', flush=True)
# ---------------- (A) production daily book variants on the daily engine
print('\n=== (A) daily 4-sleeve book (daily engine): age180 / inverse-vol in the ML (and core) sleeves ===', flush=True)
def daily_book(ml_age=720, ml_iv=False, core_iv=False, core_age=720):
    return combo([sl_listing, sl_core(core_iv, core_age), make_ml(P3, min_age_h=ml_age, iv=ml_iv), make_ml(P7, min_age_h=ml_age, iv=ml_iv)], [1, 1, 1, 1])
for sm in (0.4, 0.5, 0.6):
    go(f'D0 REF smooth{sm}', daily_book(), BASE24, smooth=sm)
summary('D0 REF', 3)
go('D1 ML age180 smooth0.5', daily_book(ml_age=A180), BASE24)
go('D2 ML IV smooth0.5', daily_book(ml_iv=True), BASE24)
for sm in (0.4, 0.5, 0.6):
    go(f'D3 ML age180+IV smooth{sm}', daily_book(ml_age=A180, ml_iv=True), BASE24, smooth=sm)
summary('D3 ML age180+IV', 3)
for sm in (0.4, 0.5, 0.6):
    go(f'D4 ML age180+IV, core IV smooth{sm}', daily_book(ml_age=A180, ml_iv=True, core_iv=True), BASE24, smooth=sm)
summary('D4 ML age180+IV core IV', 3)
go('D5 ML+core age180+IV smooth0.5', daily_book(ml_age=A180, ml_iv=True, core_iv=True, core_age=A180), BASE24)
# ---------------- (B) robustness of R5 (daily sleeves standard + ML8 age180+IV)
print('\n=== (B) R5 robustness: band / age threshold / q8 / account size / min_trade (smooth 0.5) ===', flush=True)
def r5(w8, age=A180, q8=0.3, daily_kw=None):
    dk = daily_kw or {}
    parts = [Held(sl_listing), Held(sl_core(dk.get('core_iv', False))), Held(make_ml(P3, min_age_h=dk.get('ml_age', 720), iv=dk.get('ml_iv', False))),
             Held(make_ml(P7, min_age_h=dk.get('ml_age', 720), iv=dk.get('ml_iv', False))), make_ml(P8, q8, min_age_h=age, iv=True)]
    return combo(parts, [1, 1, 1, 1, w8])
for w8 in (1.5, 2.0):
    for band in (0.2, 0.4): go(f'R5 w8={w8} band{band}', r5(w8), band=band)
    for age in (120, 240): go(f'R5 w8={w8} age{age}', r5(w8, age=age * 24))
    for q8 in (0.25, 0.35): go(f'R5 w8={w8} q8={q8}', r5(w8, q8=q8))
    go(f'R5 w8={w8} EQ100k', r5(w8), start_eq=100000.0)
    go(f'R5 w8={w8} min_trade5', r5(w8), min_trade=5.0)
summary('R5 robustness (16 perturbations)', 16)
# ---------------- (C) R7: daily ML age180+IV, core IV, ML8 age180+IV
print('\n=== (C) R7 = daily ML age180+IV + core IV + ML8 age180+IV, smooth x w8 grid ===', flush=True)
for sm in (0.4, 0.5, 0.6):
    for w8 in (1.5, 2.0, 2.5):
        go(f'R7 smooth{sm} w8={w8}', r5(w8, daily_kw=dict(ml_age=A180, ml_iv=True, core_iv=True)), smooth=sm)
summary('R7 grid', 9)
pd.DataFrame(rows).to_csv('out_exp20c.csv', index=False); print('DONE')
