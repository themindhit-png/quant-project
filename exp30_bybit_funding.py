#!/usr/bin/env python3
"""Experiment 30 — the carry book on BYBIT funding (Opus round 4, P1): replace the Binance funding panel by Bybit settlement
rates (fetched by fetch_bybit_funding.py -> data/bybit_funding.csv) on the overlap window and re-run F2 / F / D2 with signals
AND P&L on Bybit rates. Also: per-symbol correlation of daily funding sums Bybit vs Binance, mean level difference, share of
hours with a settlement (interval mix). Acceptance (PREREG 6a): Sharpe(F2 | Bybit) >= 0.8 x Sharpe(F2 | Binance) on the same
window and corr(daily funding sums) >= 0.7; otherwise the production book is F."""
import sys, math, numpy as np, pandas as pd
from exp_common import d, F2, F, go, BASE8, BASE24, combo, Held, sl_listing, sl_core, ML3n, ML7n, fund_change, fund_level, CF, FUND0
import exp_common as X
path = sys.argv[1] if len(sys.argv) > 1 else 'data/bybit_funding.csv'
bf = pd.read_csv(path); bf['ts'] = pd.to_datetime(bf.ts_ms, unit='ms', utc=True).dt.floor('h')
piv = bf.pivot_table(index='ts', columns='symbol', values='rate', aggfunc='sum')
piv = piv.reindex(index=d.idx, columns=list(d.cols))           # research grid (hours) x research symbols; NaN where Bybit has nothing
have = piv.notna().any(axis=0); print(f'Bybit funding: {int(have.sum())} of {len(d.cols)} research symbols; window {piv.dropna(how="all").index.min()} .. {piv.dropna(how="all").index.max()}')
BY = np.where(np.isfinite(piv.values), piv.values, 0.0).astype(np.float64)
BN = FUND0
i0 = int(d.idx.searchsorted(piv.dropna(how='all').index.min())); i1 = int(d.idx.searchsorted(piv.dropna(how='all').index.max()))
# ---- diagnostics on the overlap
day = d.idx.floor('D')
dby = pd.DataFrame(BY[i0:i1], index=day[i0:i1], columns=d.cols).groupby(level=0).sum(); dbn = pd.DataFrame(BN[i0:i1], index=day[i0:i1], columns=d.cols).groupby(level=0).sum()
cols = [c for c in d.cols if have[c] and np.isfinite(dbn[c]).all() and dbn[c].abs().sum() > 0]
corr = pd.Series({c: dby[c].corr(dbn[c]) for c in cols}).dropna()
print(f'daily funding sums, {len(corr)} symbols: corr median {corr.median():.2f} p25 {corr.quantile(.25):.2f} | mean level Bybit {dby[cols].values.mean()*1e4:+.2f} bps/day vs Binance {dbn[cols].values.mean()*1e4:+.2f} bps/day | mean |diff| {np.abs(dby[cols].values - dbn[cols].values).mean()*1e4:.2f} bps/day')
sett = (BY[i0:i1] != 0).mean(axis=0); print(f'settlement hours share (Bybit): median {np.median(sett[sett>0])*100:.1f}% of hours (8h => 12.5%, 4h => 25%, 1h => 100%)')
# ---- re-run books on the overlap: Binance (baseline) vs Bybit (signals + P&L)
X.OOS = str(d.idx[i0].date()); d.start_i = i0
def run_set(tag):
    go(f'F2 [{tag}]', F2(), end_i=i1); go(f'F  [{tag}]', F(), end_i=i1)
    go(f'D2 daily [{tag}]', combo([sl_listing, sl_core, ML3n, ML7n, fund_change, fund_level], [1, 1, 1, 1, 1, 1]), BASE24, end_i=i1)
print('\n=== baseline: Binance funding (signals + P&L) ===', flush=True); run_set('Binance')
print('\n=== Bybit funding (signals + P&L) ===', flush=True)
X.CF[:] = np.cumsum(BY, axis=0, dtype=np.float64); d.fund = BY.astype(np.float32); run_set('Bybit')
print('\n=== mixed: signals on Binance, P&L on Bybit (what a Binance-trained signal earns on Bybit) ===', flush=True)
X.CF[:] = np.cumsum(BN, axis=0, dtype=np.float64); d.fund = BY.astype(np.float32); run_set('sig Binance / pnl Bybit')
print('DONE')
