#!/usr/bin/env python3
"""Reg. 4: old vs new F2 (research engine, R2 lag-1 configuration) from research/phase4_out/*.csv daily returns.
Run after research/exp_r2.py (OUT_PREFIX=R4_*) and research/r4_sleeves.py: python3 research/r4_compare.py"""
import math, os, pandas as pd, numpy as np
P = 'research/phase4_out/'
DEV = ('2022-01-01', '2026-02-28'); HOLD = ('2026-03-01', '2026-10-26')
def sh(x): x = x.dropna(); return x.mean() / x.std(ddof=1) * math.sqrt(365) if len(x) > 30 else float('nan')
def mdd(x): e = (1 + x.dropna()).cumprod(); return (e / e.cummax() - 1).min()
def cagr(x): x = x.dropna(); return (1 + x).prod() ** (365.25 / len(x)) - 1
RUNS = [('R2_V2', 'old: panels reg.3, preds _oct, argsort ties (= R2_V2)'), ('R4_B_V2', '+ holes filled (1)'), ('R4_C_V2', '+ frozen prices NaN (2)'),
        ('R4_D_V2', '+ funding tie-break (3)'), ('R4_E_V2', '+ walk-forward preds regenerated = NEW'), ('R2_V0', 'V0 (F2 without V2) old'), ('R4_E_V0', 'V0 new')]
rows = []
for f, lab in RUNS:
    if not os.path.exists(P + f + '.csv'):
        continue
    df = pd.read_csv(P + f + '.csv', index_col=0, parse_dates=True); a, b = df['lag1_x1'], df['lag1_x1.5']
    yr = {f'{y}': (1 + a[str(y)]).prod() - 1 for y in range(2022, 2027)}
    rows.append(dict(run=f, what=lab, Sh=sh(a), Sh_x15=sh(b), CAGR=cagr(a), MDD=mdd(a), DEV=sh(a[DEV[0]:DEV[1]]), HOLD=sh(a[HOLD[0]:HOLD[1]]),
                     wd=a.min(), **yr))
T = pd.DataFrame(rows).set_index('run')
pd.set_option('display.width', 250)
fmt = T.copy()
for c in ['CAGR', 'MDD', 'wd', '2022', '2023', '2024', '2025', '2026']:
    fmt[c] = (T[c] * 100).round(1)
for c in ['Sh', 'Sh_x15', 'DEV', 'HOLD']:
    fmt[c] = T[c].round(2)
print(fmt.to_string()); print()
for f in ('R2_V2', 'R4_E_V2'):
    if os.path.exists(P + f + '.csv'):
        a = pd.read_csv(P + f + '.csv', index_col=0, parse_dates=True)['lag1_x1']
        w = a.nsmallest(6); print(f'{f} worst days: ' + ', '.join(f'{d.date()} {v*100:+.2f}%' for d, v in w.items()))
print()
for f in ('R4_old_sleeves', 'R4_Dpred_sleeves', 'R4_new_sleeves'):
    if os.path.exists(P + f + '.csv'):
        s = pd.read_csv(P + f + '.csv', index_col=0, parse_dates=True)
        print(f'{f}: ' + '  '.join(f'{c} {sh(s[c]):.2f} (2025-26 {sh(s[c]["2025-01-01":]):.2f})' for c in s.columns))
