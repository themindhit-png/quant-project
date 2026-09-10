#!/usr/bin/env python3
"""Summarize rolling funded logs (rows printed by harness_v4 --starts in funded mode): survival, payouts %/yr, min equity."""
import sys, ast, glob, numpy as np
for f in sys.argv[1:] or sorted(glob.glob('logs/roll5_funded*.log')):
    rows = [ast.literal_eval(l) for l in open(f) if l.strip().startswith('{')]
    if not rows:
        print(f'{f}: no rows'); continue
    halts = [r['start'] for r in rows if r['p1'] == 'halt']
    paid = np.array([r.get('paid_pct_yr', 0.0) for r in rows], dtype=float); mn = np.array([r['p1_min_eq'] for r in rows], dtype=float)
    print(f'{f}: n {len(rows)} alive {len(rows) - len(halts)}/{len(rows)} | payouts %/yr median {np.median(paid):.1f} mean {paid.mean():.1f} '
          f'min {paid.min():.1f} max {paid.max():.1f} | min_eq median {np.median(mn):.0f} worst {mn.min():.0f}' + (f' | halts: {", ".join(halts)}' if halts else ''))
