#!/usr/bin/env python3
"""Audit E: compare logs/E_indep_<run>.pkl (independent) with logs/E_proj_<run>.pkl (bt.run). Prints a markdown table."""
import os, sys, pickle, numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); L = os.path.join(ROOT, 'logs')

def daily(eq): return eq.groupby(eq.index.floor('D')).last().pct_change().dropna()
def sh(r): return r.mean() / r.std() * np.sqrt(365)
def yearly(r): return {y: (1 + g).prod() - 1 for y, g in r.groupby(r.index.year)}

runs = sys.argv[1:] or ['core_raw', 'fcarry_raw', 'core_full', 'fcarry_full', 'combo_full']
rows = []
for n in runs:
    A = pickle.load(open(os.path.join(L, f'E_indep_{n}.pkl'), 'rb')); B = pickle.load(open(os.path.join(L, f'E_proj_{n}.pkl'), 'rb'))
    ea, eb = A['eq'], B['eq']; ra, rb = daily(ea), daily(eb); j = ra.index.intersection(rb.index); ra, rb = ra[j], rb[j]
    yrs = (ea.index[-1] - ea.index[0]).days / 365.25
    c = A['comp']; eqa = ea.shift(1).bfill()                          # equity before each hour (fractions of equity)
    frac = lambda s: float((s / eqa).sum() / yrs)
    ya, yb = yearly(ra), yearly(rb); m = B['m']
    # hourly price pnl correlation on project's hp
    rows.append(dict(run=n, corr_d=ra.corr(rb), sh_E=sh(ra), sh_P=sh(rb), end_E=ea.iloc[-1], end_P=eb.iloc[-1],
                     turn_E=frac(c['traded']), turn_P=m['turnover_x'], fee_E=-frac(c['fee']), fee_P=m['fees'], slip_E=-frac(c['slip']), slip_P=m['slip'],
                     fundinc_E=frac(c['fund']), fundinc_P=-m['funding'], price_E=frac(c['price']), shock_E=-frac(c['shock']), shock_P=m['delist'],
                     **{f'y{y}_E': ya[y] for y in ya}, **{f'y{y}_P': yb.get(y, np.nan) for y in ya}))
df = pd.DataFrame(rows).set_index('run')
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 60)
print(df.T.to_string(float_format=lambda x: f'{x:.4f}'))
