#!/usr/bin/env python3
"""Experiment 1: replicate MUBITE-LADDER strategy on clean Binance data (no zombie positions),
then Opus's requested stresses: rebalance band, rebalance frequency, slippage."""
import sys, json, time, numpy as np, pandas as pd
from bt import Data, run, quantile_ls
from strategies import bot_baseline, core_signal, mom_signal, vol_h

START, END = '2021-06-01', '2026-08-31 23:00'
d = Data(start=START, end=END, min_turn_ever=3e5)
uni_bot = dict(min_age_h=720, min_turn=3e5, top_n=120)
res = {}


def go(label, **kw):
    t0 = time.time()
    args = dict(reb_h=12, band=0.05, fee_bps=4.7, slip_bps=2.0, vol_target=0.0075, max_lev=6.0,
                pos_cap=0.05, uni_kwargs=uni_bot, label=label, attrib=True)
    args.update(kw)
    m, eq = run(d, args.pop('fn', bot_baseline), **args)
    res[label] = {k: v for k, v in m.items() if k not in ('attrib',)}
    res[label]['by_year'] = {str(k): v for k, v in m['by_year'].items()}
    print(f'   ({time.time()-t0:.0f}s)\n', flush=True)
    return m, eq


# --- A. replica with bot-like params (Opus-comparable: fee 4.7 blend, slip 2) ---
m0, eq0 = go('A1 replica 12h band5 slip2 VT0.75')
eq0.to_csv('out_exp1_eq_replica.csv')
a = m0['attrib'].copy()
a['total'] = a.price + a.funding - a.cost
a['t24_med'] = np.nanmedian(np.where(d.t24 > 0, d.t24, np.nan), axis=0)
a.sort_values('total').to_csv('out_exp1_attrib_replica.csv')
print('  top-10 contributors:\n', a.sort_values('total', ascending=False).head(10)[['price', 'funding', 'cost', 'total', 't24_med']].round(0).to_string())
print('  bottom-10:\n', a.sort_values('total').head(10)[['price', 'funding', 'cost', 'total', 't24_med']].round(0).to_string())
print(f'  total price {a.price.sum():,.0f} funding {a.funding.sum():,.0f} cost {a.cost.sum():,.0f}')
print(f'  long-side pnl {a.long_pnl.sum():,.0f}  short-side pnl {a.short_pnl.sum():,.0f}')
liq = pd.cut(a.t24_med, [0, 3e6, 1e7, 3e7, 1e8, 1e12], labels=['<3M', '3-10M', '10-30M', '30-100M', '>100M'])
print('  by liquidity tier (median 24h turnover):\n', a.groupby(liq, observed=True)[['price', 'funding', 'cost', 'total']].sum().round(0).to_string(), '\n')

# --- B. slippage stress ---
for s in (5.0, 10.0):
    go(f'A2 replica slip{int(s)}', slip_bps=s)
# --- C. turnover reduction ---
go('B1 band20', band=0.20)
go('B2 band40', band=0.40)
go('B3 reb24h', reb_h=24)
go('B4 reb24h band20', reb_h=24, band=0.20)
go('B5 reb48h band20', reb_h=48, band=0.20)
# --- D. no vol-target (raw signal quality), gross 1 ---
go('C1 no VT gross1', vol_target=None, gross=1.0)
# --- E. core only / mom only ---
go('D1 core only', fn=lambda dd, i, m: quantile_ls(core_signal(dd, i, m), 0.2), vol_target=None, gross=1.0)
go('D2 mom only invvol', fn=lambda dd, i, m: quantile_ls(mom_signal(dd, i, m), 0.2, inv_vol=vol_h(dd, i, 720)), vol_target=None, gross=1.0, reb_h=168)
# --- F. liquid universe (>= $10M, top 100), tighter cap 2% ---
go('E1 liquid uni 10M top100 cap2', uni_kwargs=dict(min_age_h=720, min_turn=1e7, top_n=100), pos_cap=0.02)
go('E2 liquid uni + reb24 band20 slip5', uni_kwargs=dict(min_age_h=720, min_turn=1e7, top_n=100), pos_cap=0.02, reb_h=24, band=0.2, slip_bps=5.0)
json.dump(res, open('out_exp1_results.json', 'w'), indent=1, default=float)
print('DONE')
