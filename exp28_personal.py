#!/usr/bin/env python3
"""Experiment 28 — personal account (no firm rules): the F2 book at higher vol targets (0.6 / 0.8 / 1.0 / 1.5 %/day), max
leverage 5, position caps as in research; plus the same at taker-only execution. Reports Sharpe, CAGR, MDD, worst day,
turnover, costs — for the PLAN 'personal account' section (research engine, funding accrued)."""
from exp_common import d, F2, go, BASE8, slip_model
print('\n=== exp28: F2 for a personal account, vol targets ===', flush=True)
for vt in (0.006, 0.008, 0.010, 0.015):
    go(f'F2 personal vt {vt*100:.1f}%/d maxlev 5', F2(), vol_target=vt, max_lev=5.0, max_gross_x=3.0, pos_cap=0.03, cap_short=0.02)
print('\n=== taker-only (all orders at market) ===', flush=True)
for vt in (0.006, 0.010):
    go(f'F2 personal vt {vt*100:.1f}%/d TAKER', F2(), vol_target=vt, max_lev=5.0, max_gross_x=3.0, pos_cap=0.03, cap_short=0.02, maker_share=0.0, slip_fn=slip_model(scale=1.0))
print('DONE')
