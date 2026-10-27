#!/usr/bin/env python3
"""Experiment 19: proper combination of the daily book (listing + core + ML3 + ML7, recomputed once a day at the
00:00 bar and HELD in between) with the intraday 8h ML sleeve (recomputed every 8h). Engine rebalances every 8h.
Variants: 8h sleeve weight, band/smooth, execution assumptions; correlation of the 8h sleeve with the daily book."""
import time, json, numpy as np, pandas as pd
from bt import Data, run, quantile_ls, slip_model
from strategies import core_signal

START, END, OOS = '2021-01-01', '2026-08-31 23:00', '2022-01-01'
d = Data(start=START, end=END, min_turn_ever=5e6, load_hl=True, load_premium=True)
T, N = d.ret.shape
import os
P8FILE = os.environ.get('P8FILE', 'out_exp17_pred_8h_p0.npy')      # 8h panel on the 00/08/16 grid (DEC_PHASE=0)
P3, P7, P8 = np.load('out_exp6_pred_h3.npy'), np.load('out_exp6_pred_h7.npy'), np.load(P8FILE)
print('8h panel:', P8FILE, flush=True)
instr = json.load(open('instruments.json')); launch_h = np.full(N, np.nan)
for j, s in enumerate(d.cols):
    mm = instr.get(str(s))
    if mm and mm.get('launch'):
        launch_h[j] = mm['launch'] / 1000 / 3600
MAJ5 = [s for s in ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT'] if s in d.cols]; jm = np.array([int(np.flatnonzero(d.cols == s)[0]) for s in MAJ5])


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


class Held:
    """Recompute at bars where hh % 24 == 0 (daily decision bar), hold otherwise."""
    def __init__(self, fn):
        self.fn = fn; self.w = np.zeros(N); self.last_day = None
    def __call__(self, dd, i, mask):
        if dd.hh[i] % 24 == 0:
            self.w = self.fn(dd, i, mask)
        return self.w


def combo(parts, weights):
    def fn(dd, i, mask):
        w = np.zeros(N)
        for f, s in zip(parts, weights):
            w += s * f(dd, i, mask)
        return w / sum(weights)
    return fn


def daily_parts():
    return [Held(sl_listing), Held(sl_core), Held(make_ml(P3)), Held(make_ml(P7))]


ML8 = make_ml(P8, 0.3)
series = {}
def go(label, fn, keep=None, **kw):
    t1 = time.time(); d.start_i = int(d.idx.searchsorted(pd.Timestamp(OOS, tz='UTC')))
    args = dict(reb_h=8, reb_offset=0, band=0.3, smooth=0.5, fee_bps=5.5, maker_share=0.7, slip_fn=slip_model(scale=0.3), gross=1.0,
                pos_cap=0.02, cap_short=0.015, cap_exempt=MAJ5, cap_exempt_val=0.30, uni_kwargs=dict(min_age_h=0, min_turn=0, top_n=0),
                verbose=False, label=label, vol_target=0.006, max_lev=3.0, max_gross_x=2.0)
    args.update(kw)
    if keep:
        m, eq, hp = run(d, fn, ret_diag=True, **args); series[keep] = (hp / eq.shift(1)).dropna(); eq.to_csv(f'out_{keep}_eq.csv')
    else:
        m, eq = run(d, fn, **args)
    by = m['by_year']
    print(f'{label:<52} Sh {m["sharpe"]:5.2f} CAGR {m["cagr"]*100:6.1f}% dvol {m["dvol"]*100:.2f}% MDD {m["mdd"]*100:6.1f}% wd {m["worst_day"]*100:5.1f}% turn {m["turnover_x"]:4.0f}x cost {(m["fees"]+m["slip"])*100:.1f}% | '
          + ' '.join(f'{y}:{v[1]:.1f}' for y, v in by.items()) + f' ({time.time()-t1:.0f}s)', flush=True)
    return m


# reference: daily book on the 8h engine (held weights) — should match the daily engine result (~1.54)
go('daily book held, 8h engine (ref)', combo(daily_parts(), [1, 1, 1, 1]), keep='daily8')
go('8h ML alone', ML8, keep='ml8')
for w8 in (1.0, 2.0, 4.0):
    go(f'daily + 8h ML weight {w8:.0f} (of 4+{w8:.0f})', combo(daily_parts() + [ML8], [1, 1, 1, 1, w8]), keep=f'combo_w{int(w8)}' if w8 == 2.0 else None)
go('daily + 8h ML w2, band0.5 smooth0.7', combo(daily_parts() + [ML8], [1, 1, 1, 1, 2.0]), band=0.5, smooth=0.7)
go('daily + 8h ML w2, band0.2 smooth0.3', combo(daily_parts() + [ML8], [1, 1, 1, 1, 2.0]), band=0.2, smooth=0.3)
go('daily + 8h ML w2, maker50', combo(daily_parts() + [ML8], [1, 1, 1, 1, 2.0]), maker_share=0.5, slip_fn=slip_model(scale=0.5))
go('daily + 8h ML w2, taker only', combo(daily_parts() + [ML8], [1, 1, 1, 1, 2.0]), maker_share=0.0, slip_fn=slip_model(scale=1.0))
go('daily + 8h ML w2, 8h ML q20', combo(daily_parts() + [make_ml(P8, 0.2)], [1, 1, 1, 1, 2.0]))
if 'daily8' in series and 'ml8' in series:
    a = (1 + series['daily8']).groupby(series['daily8'].index.floor('D')).prod() - 1
    b = (1 + series['ml8']).groupby(series['ml8'].index.floor('D')).prod() - 1
    print(f'\ncorrelation(daily book, 8h ML) daily returns: {a.corr(b):.2f}')
print('DONE')
