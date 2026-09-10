#!/usr/bin/env python3
"""Experiment 2b: remaining factor tests + funding-carry variants (same setup as exp2)."""
import json, time, numpy as np, pandas as pd
from bt import Data, run, quantile_ls, rank_weights
from strategies import mom_signal, funding_signal, volume_shock_signal, vol_h, reversal_signal, core_signal

START, END = '2021-06-01', '2026-08-31 23:00'
d = Data(start=START, end=END, min_turn_ever=5e6)
UNI = dict(min_age_h=720, min_turn=1e7, top_n=100)
rows = []


def test(label, sig_fn, reb_h=24, top=0.2, inv_vol=None, rank=False, **kw):
    def fn(dd, i, m):
        s = sig_fn(dd, i, m)
        if rank:
            return rank_weights(s, m)
        iv = vol_h(dd, i, inv_vol) if inv_vol else None
        return quantile_ls(s, top, inv_vol=iv)
    t0 = time.time()
    args = dict(reb_h=reb_h, band=0.0, fee_bps=5.5, slip_bps=5.0, vol_target=None, gross=1.0,
                pos_cap=0.02, uni_kwargs=UNI, label=label, verbose=False)
    args.update(kw)
    m, eq = run(d, fn, **args)
    by = m['by_year']
    rows.append(dict(label=label, sharpe=m['sharpe'], cagr=m['cagr'], dvol=m['dvol'], mdd=m['mdd'], calmar=m['calmar'],
                     worst=m['worst_day'], turnover=m['turnover_x'], fees=m['fees'], slip=m['slip'], funding=m['funding'],
                     **{f'sh{y}': v[1] for y, v in by.items()}, **{f'r{y}': v[0] for y, v in by.items()}))
    print(f'{label:<40} Sh {m["sharpe"]:5.2f} CAGR {m["cagr"]*100:6.1f}% dvol {m["dvol"]*100:.2f}% MDD {m["mdd"]*100:6.1f}% '
          f'wd {m["worst_day"]*100:5.1f}% turn {m["turnover_x"]:4.0f}x cost {(m["fees"]+m["slip"])*100:.1f}%/y fund {-m["funding"]*100:+.1f}%/y | '
          + ' '.join(f'{y}:{v[1]:.1f}' for y, v in by.items()) + f'  ({time.time()-t0:.0f}s)', flush=True)
    return m


test('volume shock 24h/30d', lambda dd, i, m: volume_shock_signal(dd, i, m))
test('mom 720h skip24 EW reb 168h', lambda dd, i, m: mom_signal(dd, i, m, lb=720, skip=24), reb_h=168)
test('funding carry 72h reb 8h', lambda dd, i, m: funding_signal(dd, i, m, lb_h=72), reb_h=8)
# funding carry variants
test('funding carry 72h rank-weights', lambda dd, i, m: funding_signal(dd, i, m, lb_h=72), rank=True)
test('funding carry 168h top30%', lambda dd, i, m: funding_signal(dd, i, m, lb_h=168), top=0.3)
test('funding carry 168h top10%', lambda dd, i, m: funding_signal(dd, i, m, lb_h=168), top=0.1)
test('funding carry 168h invvol', lambda dd, i, m: funding_signal(dd, i, m, lb_h=168), inv_vol=720)
test('funding carry 720h', lambda dd, i, m: funding_signal(dd, i, m, lb_h=720))
test('funding carry 168h reb 168h', lambda dd, i, m: funding_signal(dd, i, m, lb_h=168), reb_h=168)
# universe sensitivity for carry
test('funding carry 168h uni top50', lambda dd, i, m: funding_signal(dd, i, m, lb_h=168), uni_kwargs=dict(min_age_h=720, min_turn=3e7, top_n=50))
test('funding carry 168h uni top150 5M', lambda dd, i, m: funding_signal(dd, i, m, lb_h=168), uni_kwargs=dict(min_age_h=720, min_turn=5e6, top_n=150))
# reversal variants
test('reversal 72h reb 8h', lambda dd, i, m: reversal_signal(dd, i, m, lb=72), reb_h=8)
test('reversal 168h invvol', lambda dd, i, m: reversal_signal(dd, i, m, lb=168), inv_vol=720)
pd.DataFrame(rows).to_csv('out_exp2b_factors.csv', index=False)
print('DONE')
