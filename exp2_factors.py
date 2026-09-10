#!/usr/bin/env python3
"""Experiment 2: factor landscape. Single cross-sectional signals, quantile 20% L/S equal-weight,
daily rebalance at 00:00 UTC, liquid universe (>= $10M/24h, top 100 by turnover, age >= 30d),
conservative taker costs (fee 5.5 + slip 5 bps), no vol targeting, gross 1.0, per-position cap 2%."""
import json, time, numpy as np, pandas as pd
from bt import Data, run, quantile_ls
from strategies import core_signal, mom_signal, funding_signal, reversal_signal, lowvol_signal, volume_shock_signal, vol_h

START, END = '2021-06-01', '2026-08-31 23:00'
d = Data(start=START, end=END, min_turn_ever=5e6)
UNI = dict(min_age_h=720, min_turn=1e7, top_n=100)
rows = []


def test(label, sig_fn, reb_h=24, top=0.2, inv_vol=None, **kw):
    def fn(dd, i, m):
        s = sig_fn(dd, i, m)
        iv = vol_h(dd, i, inv_vol) if inv_vol else None
        return quantile_ls(s, top, inv_vol=iv)
    t0 = time.time()
    m, eq = run(d, fn, reb_h=reb_h, band=0.0, fee_bps=5.5, slip_bps=5.0, vol_target=None, gross=1.0,
                pos_cap=0.02, uni_kwargs=UNI, label=label, verbose=False, **kw)
    by = m['by_year']
    row = dict(label=label, sharpe=m['sharpe'], cagr=m['cagr'], dvol=m['dvol'], mdd=m['mdd'], calmar=m['calmar'],
               worst=m['worst_day'], turnover=m['turnover_x'], fees=m['fees'], slip=m['slip'], funding=m['funding'],
               **{f'sh{y}': v[1] for y, v in by.items()}, **{f'r{y}': v[0] for y, v in by.items()})
    rows.append(row)
    print(f'{label:<38} Sh {m["sharpe"]:5.2f} CAGR {m["cagr"]*100:6.1f}% dvol {m["dvol"]*100:.2f}% MDD {m["mdd"]*100:6.1f}% '
          f'wd {m["worst_day"]*100:5.1f}% turn {m["turnover_x"]:4.0f}x cost {(m["fees"]+m["slip"])*100:.1f}%/y fund {-m["funding"]*100:+.1f}%/y | '
          + ' '.join(f'{y}:{v[1]:.1f}' for y, v in by.items()) + f'  ({time.time()-t0:.0f}s)', flush=True)
    return m


# trend quality (bot core) at several horizons
for lb in (168, 336, 720):
    test(f'core mean/std {lb}h', lambda dd, i, m, lb=lb: core_signal(dd, i, m, lb=lb))
# momentum (skip 24h) at several horizons, EW and inv-vol
for lb in (168, 336, 720, 1440, 2160):
    test(f'mom {lb}h skip24 EW', lambda dd, i, m, lb=lb: mom_signal(dd, i, m, lb=lb, skip=24))
test('mom 720h skip24 invvol', lambda dd, i, m: mom_signal(dd, i, m, lb=720, skip=24), inv_vol=720)
# short-term reversal
for lb in (24, 72, 168):
    test(f'reversal {lb}h', lambda dd, i, m, lb=lb: reversal_signal(dd, i, m, lb=lb))
# funding carry (short high funding)
for lb in (24, 72, 168, 336):
    test(f'funding carry {lb}h', lambda dd, i, m, lb=lb: funding_signal(dd, i, m, lb_h=lb))
# low vol
test('low vol 720h', lambda dd, i, m: lowvol_signal(dd, i, m, lb=720))
# volume shock
test('volume shock 24h/30d', lambda dd, i, m: volume_shock_signal(dd, i, m))
# rebalance frequency sensitivity for the two most plausible
test('mom 720h skip24 EW reb 168h', lambda dd, i, m: mom_signal(dd, i, m, lb=720, skip=24), reb_h=168)
test('funding carry 72h reb 8h', lambda dd, i, m: funding_signal(dd, i, m, lb_h=72), reb_h=8)
pd.DataFrame(rows).to_csv('out_exp2_factors.csv', index=False)
print('DONE')
