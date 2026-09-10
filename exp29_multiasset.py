#!/usr/bin/env python3
"""Experiment 29 — multi-asset diversification check for the personal account: time-series momentum (TSMOM) on gold,
silver, S&P 500, Nasdaq-100, 20y Treasuries, oil, DXY (daily, Yahoo), vol-targeted, with costs; standalone stats, the
overlap with the F2 crypto book (2022-01..2026-08), correlations and the Sharpe of capital splits (F2 alone vs 50/50 vs
optimal). Also a gold+silver-only trend book and a passive 60/40 reference. Purpose: honest answer to 'add gold/silver/
stocks for diversification?' — does it raise the portfolio Sharpe, or only lower the crypto-specific tail risk?"""
import numpy as np, pandas as pd, math
A = {k: pd.read_csv(f'data/macro/{k}.csv', index_col=0, parse_dates=True)['close'].astype(float) for k in ('gold_fut', 'silver_fut', 'spx', 'ndx', 'tlt', 'oil_fut', 'dxy')}
px = pd.DataFrame(A).dropna(how='all').ffill().dropna(); px = px[px.index >= '2004-01-01']
ret = px.pct_change().fillna(0.0)
def sh(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]; return float(x.mean() / x.std(ddof=1) * math.sqrt(252)) if len(x) > 50 and x.std(ddof=1) > 0 else float('nan')
def mdd(r): e = (1 + pd.Series(r)).cumprod(); return float((e / e.cummax() - 1).min())
def tsmom(ret, look=(63, 126, 252), vol_win=60, tgt_vol=0.10, cost_bps=5.0, long_only=False):
    """Average sign of trailing returns over several look-backs, scaled to a per-asset vol target; costs on turnover."""
    sig = sum(np.sign((1 + ret).rolling(L).apply(np.prod, raw=True) - 1) for L in look) / len(look)
    if long_only: sig = sig.clip(lower=0)
    vol = ret.rolling(vol_win).std() * math.sqrt(252)
    w = (sig * (tgt_vol / vol)).clip(-3, 3).shift(1).fillna(0.0)        # weights decided on yesterday's close
    gross = w.abs().sum(axis=1).replace(0, np.nan)
    turn = w.diff().abs().sum(axis=1).fillna(0.0)
    pnl = (w * ret).sum(axis=1) - turn * cost_bps / 1e4
    return pnl, w
print('=== TSMOM per asset (2005-2026), vol target 10%/asset, 5 bps costs ===')
for c in px.columns:
    p, _ = tsmom(ret[[c]]); print(f'  {c:<11} Sharpe {sh(p):5.2f} | CAGR {((1+p).prod()**(252/len(p))-1)*100:5.1f}% | MDD {mdd(p)*100:6.1f}% | 2022-26 Sharpe {sh(p["2022-01-01":]):5.2f}')
P_all, W = tsmom(ret); P_all = P_all / (P_all.std() * math.sqrt(252)) * 0.10          # normalise the book to 10% vol
P_pm, _ = tsmom(ret[['gold_fut', 'silver_fut']]); P_pm = P_pm / (P_pm.std() * math.sqrt(252)) * 0.10
P_eq, _ = tsmom(ret[['spx', 'ndx']]); P_eq = P_eq / (P_eq.std() * math.sqrt(252)) * 0.10
bench = (0.6 * ret['spx'] + 0.4 * ret['tlt'])
print('\n=== books, full period ===')
for name, p in (('TSMOM 7 assets', P_all), ('TSMOM gold+silver', P_pm), ('TSMOM SPX+NDX', P_eq), ('60/40 passive', bench)):
    print(f'  {name:<18} Sharpe {sh(p):5.2f} | MDD {mdd(p)*100:6.1f}% | 2022-26: Sharpe {sh(p["2022-01-01":]):5.2f} MDD {mdd(p["2022-01-01":])*100:6.1f}%')
# ---- overlap with the F2 crypto book
F = pd.read_csv('out_exp26_daily_returns.csv', index_col=0, parse_dates=True)['F2'].dropna(); F.index = F.index.tz_localize(None).normalize()
F = F.groupby(F.index).sum()
J = pd.concat([F.rename('F2'), P_all.rename('TSMOM'), P_pm.rename('PM'), P_eq.rename('EQ'), bench.rename('B6040')], axis=1)
J = J.loc['2022-01-01':'2026-08-31']; J['F2'] = J['F2'].fillna(0.0); J = J.dropna()      # trading days only (crypto weekends fall out)
print(f'\n=== overlap {J.index[0].date()}..{J.index[-1].date()} ({len(J)} trading days; crypto weekend returns dropped from F2 -> its Sharpe is slightly understated) ===')
print('  Sharpe: ' + ' | '.join(f'{c} {sh(J[c]):.2f}' for c in J.columns))
print('  correlations with F2: ' + ' | '.join(f'{c} {J["F2"].corr(J[c]):+.2f}' for c in J.columns if c != 'F2'))
vF = J['F2'].std() * math.sqrt(252); print(f'  F2 annual vol (weekdays) {vF*100:.1f}% ; TSMOM 10%')
print('\n=== capital splits (returns of each book at its own vol; w = share of capital in TSMOM, F2 at research vol target) ===')
for other in ('TSMOM', 'PM', 'EQ'):
    line = []
    for wq in (0.0, 0.25, 0.5, 0.75):
        p = (1 - wq) * J['F2'] + wq * J[other]; line.append(f'w={wq:.2f}: Sh {sh(p):.2f} MDD {mdd(p)*100:.1f}%')
    # optimal (unconstrained) mean-variance weight on the other book
    a, b = J['F2'], J[other]; c = np.cov(a, b); m = np.array([a.mean(), b.mean()]); wopt = np.linalg.solve(c, m); wopt = wopt / wopt.sum()
    print(f'  F2 + {other:<5}: ' + ' | '.join(line) + f' | MV-optimal capital share of {other}: {wopt[1]*100:.0f}%')
print('\n=== risk-parity split (both books scaled to the same vol, 50/50 risk) ===')
for other in ('TSMOM', 'PM'):
    b = J[other] / (J[other].std()) * J['F2'].std(); p = 0.5 * J['F2'] + 0.5 * b
    print(f'  F2 + {other:<5} 50/50 risk: Sharpe {sh(p):.2f} (F2 alone {sh(J["F2"]):.2f}) | MDD {mdd(p)*100:.1f}% (F2 alone {mdd(J["F2"])*100:.1f}%) | worst month {((1+p).groupby(p.index.to_period("M")).prod()-1).min()*100:.1f}% (F2 {((1+J["F2"]).groupby(J["F2"].index.to_period("M")).prod()-1).min()*100:.1f}%)')
print('DONE')
