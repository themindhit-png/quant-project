#!/usr/bin/env python3
"""Experiment 15: how do classic backtest errors inflate THIS strategy (core mean/std 336h + mom)?
 (a) clean point-in-time (as exp1 replica)         (b) 1-bar look-ahead in the signal (signal sees the bar
 being traded)  (c) survivorship universe (today's top-120 applied to all history)  (d) no funding, 2 bp costs
 (e) b+c+d together. Period 2021-06 -> 2026-08, 12h rebalance, VT 0.75%."""
import time, numpy as np, pandas as pd
from bt import Data, run, quantile_ls
from strategies import core_signal, mom_signal, vol_h

START, END = '2021-06-01', '2026-08-31 23:00'
d = Data(start=START, end=END, min_turn_ever=3e5)
T, N = d.ret.shape
# survivorship universe: top-120 by turnover at the END of the sample
t_end = np.where(np.isfinite(d.t24[T - 1]), d.t24[T - 1], -np.inf)
SURV = np.zeros(N, dtype=bool); SURV[np.argsort(-t_end)[:120]] = True
uni_pit = dict(min_age_h=720, min_turn=3e5, top_n=120)


def baseline(dd, i, mask, shift=0):
    """bot core+mom blend; shift=1 -> use data through bar i+1 (look-ahead)."""
    j = i + shift
    wc = quantile_ls(core_signal(dd, j, mask), 0.2)
    wm = quantile_ls(mom_signal(dd, j, mask), 0.2, inv_vol=vol_h(dd, j, 720))
    return 0.8 * wc + 0.2 * wm


def surv_uni(dd, i, mask):
    m = SURV & dd.valid[i] & (dd.age[i] >= 720)
    return baseline(dd, i, m)


def surv_uni_leak(dd, i, mask):
    m = SURV & dd.valid[i] & (dd.age[i] >= 720)
    return baseline(dd, i, m, shift=1)


def go(label, fn, **kw):
    t0 = time.time()
    args = dict(reb_h=12, band=0.05, fee_bps=4.7, slip_bps=2.0, vol_target=0.0075, max_lev=6.0, pos_cap=0.05,
                uni_kwargs=uni_pit, verbose=False, label=label)
    args.update(kw)
    m, eq = run(d, fn, **args)
    by = m['by_year']
    print(f'{label:<46} Sh {m["sharpe"]:5.2f} CAGR {m["cagr"]*100:6.1f}% MDD {m["mdd"]*100:6.1f}% turn {m["turnover_x"]:4.0f}x '
          f'fund {-m["funding"]*100:+.1f}% cost {(m["fees"]+m["slip"])*100:.1f}% | ' + ' '.join(f'{y}:{v[1]:.1f}' for y, v in by.items()) + f' ({time.time()-t0:.0f}s)', flush=True)


go('(a) clean point-in-time', lambda dd, i, m: baseline(dd, i, m))
go('(b) 1-bar look-ahead in signal', lambda dd, i, m: baseline(dd, i, m, shift=1))
go('(c) survivorship universe (today top-120)', surv_uni)
go('(d) no funding, 2bp cost only', lambda dd, i, m: baseline(dd, i, m), fee_bps=2.0, slip_bps=0.0)
d_nofund = d.fund.copy(); d.fund = np.zeros_like(d.fund)
go('(d2) no funding at all', lambda dd, i, m: baseline(dd, i, m), fee_bps=2.0, slip_bps=0.0)
go('(e) b + c + d2 together', surv_uni_leak, fee_bps=2.0, slip_bps=0.0)
d.fund = d_nofund
print('DONE')
