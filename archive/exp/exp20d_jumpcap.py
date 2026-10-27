#!/usr/bin/env python3
"""Experiment 20d: the residual bifurcation is jump risk in young names (a handful of +100..300% 8h moves per year
decide the year). Test jump-risk controls on the perturbation set that exposed failures:
J1 cap_young 0.5% (age<180d), J2 cap_young 0.75%, J3 short stop 25% (all names, 48h cooldown), J4 = J1+J3.
Books: R5 (ML8 age180+IV) at w8 1.5/2.0, REF (ML8 plain) at w8 2.0/2.5, and the daily production book (daily engine)."""
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
A180 = 180 * 24
BASE8 = dict(reb_h=8, reb_offset=0, band=0.3, smooth=0.5, fee_bps=5.5, maker_share=0.7, slip_fn=slip_model(scale=0.3), gross=1.0, pos_cap=0.02, cap_short=0.015,
             cap_exempt=MAJ5, cap_exempt_val=0.30, uni_kwargs=dict(min_age_h=0, min_turn=0, top_n=0), verbose=False, vol_target=0.006, max_lev=3.0, max_gross_x=2.0)
BASE24 = dict(BASE8); BASE24.update(reb_h=24)
rows = []
def go(label, fn, base=BASE8, **kw):
    t1 = time.time(); d.start_i = int(d.idx.searchsorted(pd.Timestamp(OOS, tz='UTC'))); args = dict(base); args.update(kw); args['label'] = label
    m, eq = run(d, fn, **args); by = m['by_year']
    rows.append(dict(label=label, sharpe=m['sharpe'], cagr=m['cagr'], mdd=m['mdd'], worst_day=m['worst_day'], turnover=m['turnover_x'], n_stops=m['n_stops'], **{f'y{y}': v[1] for y, v in by.items()}))
    print(f'{label:<52} Sh {m["sharpe"]:5.2f} CAGR {m["cagr"]*100:6.1f}% MDD {m["mdd"]*100:6.1f}% wd {m["worst_day"]*100:5.1f}% turn {m["turnover_x"]:4.0f}x cost {(m["fees"]+m["slip"])*100:.1f}% stops {m["n_stops"]:3d} | '
          + ' '.join(f'{y}:{v[1]:.1f}' for y, v in by.items()) + f' ({time.time()-t1:.0f}s)', flush=True)
    return m
def summary(tag, n):
    s = pd.Series([r['sharpe'] for r in rows[-n:]]); y25 = pd.Series([r.get('y2025', np.nan) for r in rows[-n:]])
    print(f'>>> {tag}: Sharpe mean {s.mean():.2f} min {s.min():.2f} max {s.max():.2f} std {s.std():.2f} | 2025 Sharpe min {y25.min():.1f} | points<1.3: {(s<1.3).sum()}/{n}\n', flush=True)
def r5(w8, age=A180, q8=0.3): return combo([Held(sl_listing), Held(sl_core), Held(make_ml(P3)), Held(make_ml(P7)), make_ml(P8, q8, min_age_h=age, iv=True)], [1, 1, 1, 1, w8])
def ref(w8): return combo([Held(sl_listing), Held(sl_core), Held(make_ml(P3)), Held(make_ml(P7)), make_ml(P8)], [1, 1, 1, 1, w8])
# perturbation set (the ones that exposed failures) for R5
PERT = [('base', {}, {}), ('band0.2', {}, dict(band=0.2)), ('band0.4', {}, dict(band=0.4)), ('age120', dict(age=120 * 24), {}), ('q8=0.35', dict(q8=0.35), {}),
        ('smooth0.4', {}, dict(smooth=0.4)), ('smooth0.6', {}, dict(smooth=0.6))]
CTRL = {'J0 none': {}, 'J1 capY0.5%': dict(cap_young=0.005), 'J2 capY0.75%': dict(cap_young=0.0075),
        'J3 stopS25': dict(stop_pct=0.25, stop_side='short', stop_cooldown_h=48), 'J4 capY0.5%+stopS25': dict(cap_young=0.005, stop_pct=0.25, stop_side='short', stop_cooldown_h=48)}
for cname, ckw in CTRL.items():
    print(f'\n=== R5 book, control {cname} ===', flush=True)
    for w8 in (1.5, 2.0):
        for pname, bkw, rkw in PERT:
            kw = dict(rkw); kw.update(ckw); go(f'R5 {cname} w8={w8} {pname}', r5(w8, **bkw), **kw)
    summary(f'R5 {cname} (14 points)', 14)
for cname in ('J0 none', 'J1 capY0.5%', 'J4 capY0.5%+stopS25'):
    ckw = CTRL[cname]; print(f'\n=== REF book (plain ML8), control {cname} ===', flush=True)
    for w8 in (2.0, 2.5):
        for sm in (0.4, 0.5, 0.6):
            kw = dict(smooth=sm); kw.update(ckw); go(f'REF {cname} w8={w8} smooth{sm}', ref(w8), **kw)
    summary(f'REF {cname} (6 points)', 6)
print('\n=== daily production book (daily engine) with jump controls ===', flush=True)
daily = combo([sl_listing, sl_core, make_ml(P3), make_ml(P7)], [1, 1, 1, 1])
for cname in ('J0 none', 'J1 capY0.5%', 'J2 capY0.75%', 'J3 stopS25', 'J4 capY0.5%+stopS25'):
    ckw = CTRL[cname]
    for sm in (0.4, 0.5, 0.6):
        kw = dict(smooth=sm); kw.update(ckw); go(f'DAILY {cname} smooth{sm}', daily, BASE24, **kw)
    summary(f'DAILY {cname} (3 points)', 3)
pd.DataFrame(rows).to_csv('out_exp20d.csv', index=False); print('DONE')
