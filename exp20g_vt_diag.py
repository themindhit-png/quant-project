#!/usr/bin/env python3
"""Experiment 20g: is the vol-target update (lev <- lev * target/dvol, recursive on LEVERED realised vol) unstable at
8h cadence? Print leverage path statistics for the daily book on the daily engine vs the 8h engine (hold-exact)."""
import time, json, numpy as np, pandas as pd
from bt import Data, run, quantile_ls, slip_model
from strategies import core_signal
START, END, OOS = '2021-01-01', '2026-08-31 23:00', '2022-01-01'
d = Data(start=START, end=END, min_turn_ever=5e6, load_hl=True, load_premium=True); T, N = d.ret.shape
P3, P7 = np.load('out_exp6_pred_h3.npy'), np.load('out_exp6_pred_h7.npy')
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
def make_ml(P, top=0.3):
    def f(dd, i, mask):
        m = dd.universe(i, min_age_h=720, min_turn=1e7, top_n=150); return quantile_ls(np.where(m & np.isfinite(P[i]), P[i], np.nan), top)
    return f
def combo(parts, weights):
    def fn(dd, i, mask):
        w = np.zeros(N)
        for f, s in zip(parts, weights): w += s * f(dd, i, mask)
        return w / sum(weights)
    return fn
BASE8 = dict(reb_h=8, reb_offset=0, band=0.3, smooth=0.0, fee_bps=5.5, maker_share=0.7, slip_fn=slip_model(scale=0.3), gross=1.0, pos_cap=0.02, cap_short=0.015,
             cap_exempt=MAJ5, cap_exempt_val=0.30, uni_kwargs=dict(min_age_h=0, min_turn=0, top_n=0), verbose=False, vol_target=0.006, max_lev=3.0, max_gross_x=2.0)
BASE24 = dict(BASE8); BASE24.update(reb_h=24)
daily = combo([sl_listing, sl_core, make_ml(P3), make_ml(P7)], [1, 1, 1, 1])
def hold_exact():
    st = {'w': None}
    def fn(dd, i, mask):
        if dd.hh[i] % 24 == 0:
            w = daily(dd, i, mask); st['w'] = w if st['w'] is None else 0.5 * st['w'] + 0.5 * w; return st['w']
        den = dd.cur_gross * dd.cur_lev * dd.cur_eq
        return dd.cur_pos / den if den > 0 else np.zeros(N)
    return fn
def levstats(label, m):
    L = pd.DataFrame(m['lev_hist'], columns=['i', 'lev', 'gross', 'npos']); L['ts'] = d.idx[L.i.values]
    lv = L.set_index('ts').lev
    daily_lev = lv.groupby(lv.index.floor('D')).last()
    chg = np.log(daily_lev).diff().abs()
    print(f'{label}: Sh {m["sharpe"]:.2f} | lev mean {lv.mean():.2f} p5 {lv.quantile(.05):.2f} p95 {lv.quantile(.95):.2f} | at clip 0.05: {(lv<=0.0501).mean()*100:.1f}%  at max 3.0: {(lv>=2.999).mean()*100:.1f}% | '
          f'median |dlog lev|/day {chg.median():.3f}  p95 {chg.quantile(.95):.3f} | days with lev change >25%: {(chg>0.25).sum()}')
    return daily_lev
for label, fn, base, kw in [('T1 daily engine', daily, BASE24, dict(smooth=0.5)), ('T2 8h hold-exact band.3', hold_exact(), BASE8, {}), ('T2b 8h hold-exact band.05', hold_exact(), BASE8, dict(band=0.05))]:
    d.start_i = int(d.idx.searchsorted(pd.Timestamp(OOS, tz='UTC'))); args = dict(base); args.update(kw); args['label'] = label
    m, eq = run(d, fn, **args); dl = levstats(label, m)
    print('   lev by half-year:', ' '.join(f'{p}:{v:.2f}' for p, v in dl.groupby(dl.index.to_period("Q")).mean().items()))
print('DONE')
