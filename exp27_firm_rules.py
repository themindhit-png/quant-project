#!/usr/bin/env python3
"""Experiment 27 — firm-rule sensitivities on the F2 book: HyroTrader's 'low-cap altcoins <= 5% of initial balance' rule
mostly hits the listing sleeve (young small caps). Variants: listing sleeve scale 1 (base) / 0.5 / 0.25 / 0 and a
listing-sleeve gross cap in weight space (sleeve gross <= 5% of the book's gross weight)."""
import time, math, numpy as np, pandas as pd
from exp26_opus3 import d, F2, Held, sl_listing, sl_core, ML3n, ML7n, ML8, fund_change, fund_level, combo, BASE8, go
def capped_listing(max_gross):
    def f(dd, i, mask):
        w = sl_listing(dd, i, mask); g = float(np.abs(w).sum())
        return w * (max_gross / g) if g > max_gross else w
    return f
print('\n=== exp27: listing sleeve scale (HyroTrader low-cap rule proxy) ===', flush=True)
go('F2 base (listing scale 1)', F2())
for s in (0.5, 0.25):
    go(f'F2 listing scale {s}', combo([Held(sl_listing), Held(sl_core), Held(ML3n), Held(ML7n), ML8, Held(fund_change), fund_level], [s, 1, 1, 1, 2, 1, 1]))
go('F2 without listing sleeve', combo([Held(sl_core), Held(ML3n), Held(ML7n), ML8, Held(fund_change), fund_level], [1, 1, 1, 2, 1, 1]))
go('F2 listing gross capped at 0.4 (sleeve weight space; ~5% of equity at lev 1)', combo([Held(capped_listing(0.4)), Held(sl_core), Held(ML3n), Held(ML7n), ML8, Held(fund_change), fund_level], [1, 1, 1, 1, 2, 1, 1]))
print('DONE')
