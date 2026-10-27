#!/usr/bin/env python3
"""Experiment 14: explain the gap between research engine (exp13b P0: Sharpe 1.54) and the bot-in-the-loop
harness (1.11): (a) universe hysteresis exit_n=top+40 as in the bot, (b) listing sleeve with Binance-clock
7-90d (harness uses Binance age), (c) both."""
import time, json, numpy as np, pandas as pd
from bt import Data, run, quantile_ls, slip_model
from strategies import core_signal

START, END, OOS = '2021-01-01', '2026-08-31 23:00', '2022-01-01'
d = Data(start=START, end=END, min_turn_ever=5e6, load_hl=True, load_premium=True)
T, N = d.ret.shape
MAJ5 = [s for s in ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT'] if s in d.cols]
jm = np.array([int(np.flatnonzero(d.cols == s)[0]) for s in MAJ5])
P3 = np.load('out_exp6_pred_h3.npy'); P7 = np.load('out_exp6_pred_h7.npy')
instr = json.load(open('instruments.json'))
launch_h = np.full(N, np.nan)
for j, s in enumerate(d.cols):
    m = instr.get(str(s))
    if m and m.get('launch'):
        launch_h[j] = m['launch'] / 1000 / 3600


class Hyst:
    """separate hysteresis state per universe name (bt.Data keeps one prev_mask)"""
    def __init__(self):
        self.prev = {}

    def select(self, dd, i, name, min_turn, top_n, extra):
        t = dd.t24[i]
        m = (dd.age[i] >= 720) & dd.valid[i] & np.isfinite(t) & (t >= min_turn)
        if m.sum() > top_n:
            score = np.where(m, t, -np.inf); order = np.argsort(-score)
            rank = np.empty(N, dtype=np.int64); rank[order] = np.arange(N)
            keep = rank < top_n
            if extra and name in self.prev:
                keep |= self.prev[name] & (rank < top_n + extra)
            m &= keep
        self.prev[name] = m.copy()
        return m


H = Hyst()


def sl_listing(clock='two', lo=24 * 7, hi=24 * 90):
    def fn(dd, i, mask):
        age = dd.age[i].astype(float)
        if clock == 'two':
            age = np.maximum(age, np.where(np.isfinite(launch_h), dd.hh[i] - launch_h, -np.inf))
        t = dd.t24[i]
        cand = (age >= lo) & (age <= hi) & dd.valid[i] & np.isfinite(t) & (t >= 2e7) & (dd.age[i] >= 72)
        cand[jm] = False
        r24 = dd.cff[i] / dd.cff[i - 24] - 1; cand &= ~(r24 > 0.30)
        idx = np.flatnonzero(cand); w = np.zeros(N)
        if len(idx) == 0:
            return w
        idx = idx[np.argsort(-t[idx])][:30]; ws = min(1.0 / len(idx), 0.05)
        w[idx] = -ws; w[jm] += ws * len(idx) / len(jm)
        return w
    return fn


def sl_core(extra):
    def fn(dd, i, mask):
        m = H.select(dd, i, 'core', 1e7, 100, extra)
        return quantile_ls(core_signal(dd, i, m, 336), 0.2)
    return fn


def make_ml(P, extra):
    def f(dd, i, mask):
        m = H.select(dd, i, 'ml', 1e7, 150, extra)
        s = np.where(m & np.isfinite(P[i]), P[i], np.nan)
        return quantile_ls(s, 0.3)
    return f


def combo(parts):
    def fn(dd, i, mask):
        w = np.zeros(N)
        for f in parts:
            w += f(dd, i, mask)
        return w / len(parts)
    return fn


def go(label, fn, **kw):
    t0 = time.time(); H.prev.clear()
    d.start_i = int(d.idx.searchsorted(pd.Timestamp(OOS, tz='UTC')))
    args = dict(reb_h=24, band=0.3, smooth=0.5, fee_bps=5.5, maker_share=0.7, slip_fn=slip_model(scale=0.3), gross=1.0,
                pos_cap=0.02, cap_short=0.015, cap_exempt=MAJ5, cap_exempt_val=0.30, uni_kwargs=dict(min_age_h=0, min_turn=0, top_n=0),
                verbose=False, label=label, vol_target=0.006, max_lev=3.0, max_gross_x=2.0)
    args.update(kw)
    m, eq = run(d, fn, **args)
    by = m['by_year']
    print(f'{label:<48} Sh {m["sharpe"]:5.2f} CAGR {m["cagr"]*100:6.1f}% dvol {m["dvol"]*100:.2f}% MDD {m["mdd"]*100:6.1f}% wd {m["worst_day"]*100:5.1f}% '
          f'turn {m["turnover_x"]:4.0f}x cost {(m["fees"]+m["slip"])*100:.1f}% | ' + ' '.join(f'{y}:{v[1]:.1f}' for y, v in by.items()) + f' ({time.time()-t0:.0f}s)', flush=True)


go('P0 ref (two-clock, no hysteresis)', combo([sl_listing('two'), sl_core(0), make_ml(P3, 0), make_ml(P7, 0)]))
go('a) hysteresis +40', combo([sl_listing('two'), sl_core(40), make_ml(P3, 40), make_ml(P7, 40)]))
go('b) Binance-clock 7-90d', combo([sl_listing('bin'), sl_core(0), make_ml(P3, 0), make_ml(P7, 0)]))
go('c) both (≈ harness setup)', combo([sl_listing('bin'), sl_core(40), make_ml(P3, 40), make_ml(P7, 40)]))
go('d) c + taker share 27% (3 attempts)', combo([sl_listing('bin'), sl_core(40), make_ml(P3, 40), make_ml(P7, 40)]), maker_share=0.73, slip_fn=slip_model(scale=0.27))
print('DONE')
