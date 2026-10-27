#!/usr/bin/env python3
"""Experiment 21b — the untested regime: 2020-H2 … 2021 (alt bull run). No ML predictions exist before 2022, so this
tests the listing and core sleeves (plus a weekly 720h momentum sleeve) with the fixed VT, daily engine, same costs."""
import time, json, numpy as np, pandas as pd
from bt import Data, run, quantile_ls, slip_model
from strategies import core_signal
START, END, OOS = '2020-04-01', '2021-12-31 23:00', '2020-08-01'
d = Data(start=START, end=END, min_turn_ever=5e6, load_hl=True, load_premium=False); T, N = d.ret.shape
instr = json.load(open('instruments.json')); launch_h = np.full(N, np.nan)
for j, s in enumerate(d.cols):
    mm = instr.get(str(s))
    if mm and mm.get('launch'): launch_h[j] = mm['launch'] / 1000 / 3600
MAJ5 = [s for s in ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT'] if s in d.cols]; jm = np.array([int(np.flatnonzero(d.cols == s)[0]) for s in MAJ5])
print('majors available:', MAJ5)
def sl_listing(dd, i, mask):
    age = np.maximum(dd.age[i].astype(float), np.where(np.isfinite(launch_h), dd.hh[i] - launch_h, -np.inf)); t = dd.t24[i]
    cand = (age >= 24 * 7) & (age <= 24 * 90) & dd.valid[i] & np.isfinite(t) & (t >= 2e7) & (dd.age[i] >= 72); cand[jm] = False
    r24 = dd.cff[i] / dd.cff[i - 24] - 1; cand &= ~(r24 > 0.30); idx = np.flatnonzero(cand); w = np.zeros(N)
    if len(idx) == 0: return w
    idx = idx[np.argsort(-t[idx])][:30]; ws = min(1.0 / len(idx), 0.05); w[idx] = -ws; w[jm] += ws * len(idx) / len(jm); return w
def sl_core(dd, i, mask):
    m = dd.universe(i, min_age_h=720, min_turn=1e7, top_n=100); return quantile_ls(core_signal(dd, i, m, 336), 0.2)
class Weekly:
    def __init__(self, fn): self.fn = fn; self.w = np.zeros(N); self.last = None
    def __call__(self, dd, i, mask):
        wk = (dd.hh[i] // 24) // 7
        if wk != self.last: self.w = self.fn(dd, i, mask); self.last = wk
        return self.w
def sl_mom(dd, i, mask):
    m = dd.universe(i, min_age_h=720 + 24, min_turn=1e7, top_n=100)
    with np.errstate(all='ignore'): sig = dd.cff[i - 24] / dd.cff[i - 720] - 1.0
    return quantile_ls(np.where(m & np.isfinite(sig), sig, np.nan), 0.2)
def combo(parts, weights):
    def fn(dd, i, mask):
        w = np.zeros(N)
        for f, s in zip(parts, weights): w += s * f(dd, i, mask)
        return w / sum(weights)
    return fn
BASE24 = dict(reb_h=24, reb_offset=0, band=0.3, smooth=0.5, fee_bps=5.5, maker_share=0.7, slip_fn=slip_model(scale=0.3), gross=1.0, pos_cap=0.02, cap_short=0.015,
              cap_exempt=MAJ5, cap_exempt_val=0.30, uni_kwargs=dict(min_age_h=0, min_turn=0, top_n=0), verbose=False, vol_target=0.006, max_lev=3.0, max_gross_x=2.0)
def go(label, fn, **kw):
    t1 = time.time(); d.start_i = int(d.idx.searchsorted(pd.Timestamp(OOS, tz='UTC'))); args = dict(BASE24); args.update(kw); args['label'] = label
    m, eq, hp = run(d, fn, ret_diag=True, **args)
    de = eq.groupby(eq.index.floor('D')).last(); r = de.pct_change().dropna()
    halves = {}
    for lab, a, b in (('2020H2', '2020-08-01', '2020-12-31'), ('2021H1', '2021-01-01', '2021-06-30'), ('2021H2', '2021-07-01', '2021-12-31')):
        x = r[a:b]
        if len(x) > 30: halves[lab] = (x.mean() / x.std(ddof=1) * np.sqrt(365), (1 + x).prod() - 1)
    print(f'{label:<40} Sh {m["sharpe"]:5.2f} CAGR {m["cagr"]*100:6.1f}% dvol {m["dvol"]*100:.2f}% MDD {m["mdd"]*100:6.1f}% wd {m["worst_day"]*100:5.1f}% turn {m["turnover_x"]:4.0f}x npos {m["avg_npos"]:.0f} | '
          + ' '.join(f'{k}: Sh {v[0]:.1f} ret {v[1]*100:+.0f}%' for k, v in halves.items()) + f' ({time.time()-t1:.0f}s)', flush=True)
    return m
print('\n=== 2020-08 → 2021-12 (alt bull run), direct VT, daily engine ===', flush=True)
go('listing alone', sl_listing)
go('listing alone, no pump filter, stop-short 40%', sl_listing, stop_pct=0.40, stop_side='short')
go('core alone', sl_core)
go('mom720 weekly alone', Weekly(sl_mom))
go('listing + core', combo([sl_listing, sl_core], [1, 1]))
go('listing + core + mom', combo([sl_listing, sl_core, Weekly(sl_mom)], [1, 1, 1]))
go('listing + core, VT legacy (recursive) for reference', combo([sl_listing, sl_core], [1, 1]), vt_mode='recursive')
print('DONE')
