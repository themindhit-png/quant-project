#!/usr/bin/env python3
"""Experiment 20h: the research engine's vol-target was a bang-bang oscillator (lev at 0.05 clip 35% of the time, at
3.0 20%). Re-evaluate with the DIRECT estimator (lev = target / realised unlevered vol): daily book, daily-held on 8h,
ML8 alone, REF and R5 5-sleeve grids, path correlation of adjacent configs."""
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
             cap_exempt=MAJ5, cap_exempt_val=0.30, uni_kwargs=dict(min_age_h=0, min_turn=0, top_n=0), verbose=False, vol_target=0.006, max_lev=3.0, max_gross_x=2.0, vt_mode='direct')
BASE24 = dict(BASE8); BASE24.update(reb_h=24)
rows = []
def go(label, fn, base=BASE8, keep=False, **kw):
    t1 = time.time(); d.start_i = int(d.idx.searchsorted(pd.Timestamp(OOS, tz='UTC'))); args = dict(base); args.update(kw); args['label'] = label
    if keep: m, eq, hp = run(d, fn, ret_diag=True, **args)
    else: m, eq = run(d, fn, **args); hp = None
    by = m['by_year']; L = pd.DataFrame(m['lev_hist'], columns=['i', 'lev', 'gross', 'npos'])
    rows.append(dict(label=label, sharpe=m['sharpe'], cagr=m['cagr'], mdd=m['mdd'], dvol=m['dvol'], turnover=m['turnover_x'], lev_mean=L.lev.mean(), lev_p5=L.lev.quantile(.05), lev_p95=L.lev.quantile(.95), **{f'y{y}': v[1] for y, v in by.items()}))
    print(f'{label:<44} Sh {m["sharpe"]:5.2f} CAGR {m["cagr"]*100:6.1f}% dvol {m["dvol"]*100:.2f}% MDD {m["mdd"]*100:6.1f}% wd {m["worst_day"]*100:5.1f}% turn {m["turnover_x"]:4.0f}x cost {(m["fees"]+m["slip"])*100:.1f}% lev {L.lev.mean():.2f}[{L.lev.quantile(.05):.2f}-{L.lev.quantile(.95):.2f}] | '
          + ' '.join(f'{y}:{v[1]:.1f}' for y, v in by.items()) + f' ({time.time()-t1:.0f}s)', flush=True)
    return m, eq, hp
def summary(tag, n):
    s = pd.Series([r['sharpe'] for r in rows[-n:]]); y25 = pd.Series([r.get('y2025', np.nan) for r in rows[-n:]]); mn = pd.Series([min(v for k, v in r.items() if k.startswith('y')) for r in rows[-n:]])
    print(f'>>> {tag}: Sharpe mean {s.mean():.2f} min {s.min():.2f} max {s.max():.2f} std {s.std():.2f} | 2025 min {y25.min():.1f} | worst year-Sharpe min {mn.min():.1f}\n', flush=True)
daily = combo([sl_listing, sl_core, make_ml(P3), make_ml(P7)], [1, 1, 1, 1])
def daily8(): return combo([Held(sl_listing), Held(sl_core), Held(make_ml(P3)), Held(make_ml(P7))], [1, 1, 1, 1])
print('\n=== daily production book: VT recursive (legacy) vs direct ===', flush=True)
go('D recursive (legacy)', daily, BASE24, vt_mode='recursive')
go('D direct', daily, BASE24)
go('D direct vt_smooth .5', daily, BASE24, vt_smooth=0.5)
go('D direct win 20d', daily, BASE24, vol_win_d=20)
go('D direct win 60d', daily, BASE24, vol_win_d=60)
for sm in (0.4, 0.6): go(f'D direct smooth{sm}', daily, BASE24, smooth=sm)
print('\n=== 8h engine components (direct VT) ===', flush=True)
go('daily-held on 8h, smooth .794', daily8(), smooth=0.5 ** (1 / 3))
go('daily-held on 8h, smooth .5', daily8())
go('ML8 alone', make_ml(P8))
go('ML8 alone age180+IV', make_ml(P8, min_age_h=A180, iv=True))
def ref(w8): return combo([Held(sl_listing), Held(sl_core), Held(make_ml(P3)), Held(make_ml(P7)), make_ml(P8)], [1, 1, 1, 1, w8])
def r5(w8): return combo([Held(sl_listing), Held(sl_core), Held(make_ml(P3)), Held(make_ml(P7)), make_ml(P8, 0.3, min_age_h=A180, iv=True)], [1, 1, 1, 1, w8])
print('\n=== REF 5-sleeve grid (direct VT) ===', flush=True)
for sm in (0.4, 0.5, 0.6):
    for w8 in (1.5, 2.0, 2.5): go(f'REF smooth{sm} w8={w8}', ref(w8), smooth=sm)
summary('REF grid direct', 9)
print('\n=== R5 5-sleeve grid (direct VT) ===', flush=True)
for sm in (0.4, 0.5, 0.6):
    for w8 in (1.5, 2.0, 2.5): go(f'R5 smooth{sm} w8={w8}', r5(w8), smooth=sm)
summary('R5 grid direct', 9)
print('\n=== path correlation (direct VT): R5 w8=2 base vs band0.2 ===', flush=True)
_, eqa, hpa = go('R5 w8=2 base', r5(2.0), keep=True); _, eqb, hpb = go('R5 w8=2 band0.2', r5(2.0), keep=True, band=0.2)
ra = (hpa / eqa.shift(1)).dropna(); rb = (hpb / eqb.shift(1)).dropna()
da = (1 + ra).groupby(ra.index.floor('D')).prod() - 1; db = (1 + rb).groupby(rb.index.floor('D')).prod() - 1
for y in range(2022, 2027):
    a = da[da.index.year == y]; b = db[db.index.year == y]; print(f'  {y}: corr(daily rets) {a.corr(b):.2f}  sum {a.sum()*100:5.1f}%/{b.sum()*100:5.1f}%')
pd.DataFrame(rows).to_csv('out_exp20h.csv', index=False); print('DONE')
