#!/usr/bin/env python3
"""Experiment 7: deep-dive on the new-listing drift (short young perps vs long majors).
 (a) event study: mean/median log return by age (days since first bar) for listings with turnover >= X
 (b) strategy variants: age windows, liquidity thresholds, n_max, hedge (majors / BTC / none),
     stop-loss, weekly vs daily rebalance; by-year table.
 (c) sub-period stability (2021-23 vs 2024-26)."""
import time, numpy as np, pandas as pd
from bt import Data, run, slip_model

START, END = '2021-01-01', '2026-08-31 23:00'
d = Data(start=START, end=END, min_turn_ever=5e6, load_hl=True)
MAJORS = [s for s in ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT'] if s in d.cols]
jm = np.array([int(np.flatnonzero(d.cols == s)[0]) for s in MAJORS])
jbtc = int(np.flatnonzero(d.cols == 'BTCUSDT')[0])
T, N = d.ret.shape

# ---------------- (a) event study ----------------
print('\n=== event study: cumulative log return from day 3 close by listing age, listings with day-3 turnover >= thr ===')
for thr in (5e6, 2e7, 5e7):
    curves = []
    for j in range(N):
        a = d.age[:, j]
        starts = np.flatnonzero(a == 0)
        for s0 in starts:
            i3 = s0 + 72
            if i3 + 24 * 90 >= T or not d.valid[i3, j] or d.t24[i3, j] < thr:
                continue
            if d.idx[s0] < pd.Timestamp('2021-01-01', tz='UTC'):
                continue
            p = d.cff[i3:i3 + 24 * 90 + 1:24, j]
            if not np.all(np.isfinite(p)) or p[0] <= 0:
                continue
            curves.append(np.log(p / p[0]) - np.log(d.cff[i3:i3 + 24 * 90 + 1:24, jbtc] / d.cff[i3, jbtc]))
    C = np.array(curves)
    if len(C) == 0:
        continue
    days = [1, 3, 7, 14, 30, 45, 60, 90]
    print(f'thr ${thr/1e6:.0f}M: n listings {len(C)} | BTC-relative log return from day3: ' +
          '  '.join(f'd{k}: mean {C[:, k].mean()*100:+.1f}% med {np.median(C[:, k])*100:+.1f}%' for k in days))
    # by cohort year
    yrs = np.array([d.idx[s0].year for j in range(N) for s0 in np.flatnonzero(d.age[:, j] == 0)
                    if (s0 + 72 + 24 * 90 < T and d.valid[s0 + 72, j] and d.t24[s0 + 72, j] >= thr and d.idx[s0] >= pd.Timestamp('2021-01-01', tz='UTC')
                        and np.all(np.isfinite(d.cff[s0 + 72:s0 + 72 + 24 * 90 + 1:24, j])) and d.cff[s0 + 72, j] > 0)])
    if len(yrs) == len(C):
        for y in sorted(set(yrs)):
            sel = yrs == y
            print(f'    cohort {y}: n {sel.sum():3d}  d30 mean {C[sel, 30].mean()*100:+.1f}%  d60 mean {C[sel, 60].mean()*100:+.1f}%  hit-rate(d30<0) {(C[sel, 30] < 0).mean()*100:.0f}%')

# ---------------- (b) strategy variants ----------------
rows = []


def report(label, m, t0):
    by = m['by_year']
    rows.append(dict(label=label, sharpe=m['sharpe'], cagr=m['cagr'], dvol=m['dvol'], mdd=m['mdd'], calmar=m['calmar'],
                     worst=m['worst_day'], turnover=m['turnover_x'], fees=m['fees'], slip=m['slip'], funding=m['funding'],
                     npos=m['avg_npos'], **{f'sh{y}': v[1] for y, v in by.items()}, **{f'r{y}': v[0] for y, v in by.items()}))
    print(f'{label:<58} Sh {m["sharpe"]:5.2f} CAGR {m["cagr"]*100:6.1f}% dvol {m["dvol"]*100:.2f}% MDD {m["mdd"]*100:6.1f}% '
          f'wd {m["worst_day"]*100:5.1f}% turn {m["turnover_x"]:4.0f}x cost {(m["fees"]+m["slip"])*100:.1f}% fund {-m["funding"]*100:+.1f}% npos {m["avg_npos"]:.0f} | '
          + ' '.join(f'{y}:{v[1]:.1f}' for y, v in by.items()) + f'  ({time.time()-t0:.0f}s)', flush=True)


def listing_short(lo=72, hi=24 * 60, thr=2e7, n_max=15, hedge='majors', exclude_pump=None):
    def fn(dd, i, mask):
        age = dd.age[i]; t = dd.t24[i]
        cand = (age >= lo) & (age <= hi) & dd.valid[i] & np.isfinite(t) & (t >= thr)
        cand[jm] = False
        if exclude_pump:                       # don't short names that pumped > x in last 24h (squeeze risk)
            r24 = dd.cff[i] / dd.cff[i - 24] - 1
            cand &= ~(r24 > exclude_pump)
        idx = np.flatnonzero(cand)
        w = np.zeros(N)
        if len(idx) == 0:
            return w
        idx = idx[np.argsort(-t[idx])][:n_max]
        w[idx] = -1.0 / len(idx)
        if hedge == 'majors':
            w[jm] += 1.0 / len(jm)
        elif hedge == 'btc':
            w[jbtc] += 1.0
        return w
    return fn


def go(label, fn, **kw):
    t0 = time.time()
    args = dict(reb_h=24, band=0.0, fee_bps=5.5, slip_fn=slip_model(), gross=1.0, pos_cap=0.5, cap_short=0.1,
                uni_kwargs=dict(min_age_h=0, min_turn=0, top_n=0), verbose=False, label=label)
    args.update(kw)
    m, eq = run(d, fn, **args)
    report(label, m, t0)
    return m, eq


d.start_i = int(d.idx.searchsorted(pd.Timestamp('2021-06-01', tz='UTC')))
go('L0 age3-60d thr20M n15 hedge majors daily', listing_short())
go('L1 hedge BTC only', listing_short(hedge='btc'))
go('L2 no hedge (net short)', listing_short(hedge='none'))
go('L3 thr 10M', listing_short(thr=1e7))
go('L4 thr 50M', listing_short(thr=5e7))
go('L5 n_max 30', listing_short(n_max=30))
go('L6 age 1-30d', listing_short(lo=24, hi=24 * 30))
go('L7 age 3-90d', listing_short(hi=24 * 90))
go('L8 age 7-60d', listing_short(lo=24 * 7))
go('L9 weekly rebalance', listing_short(), reb_h=168)
go('L10 stop 30% + cooldown', listing_short(), stop_pct=0.30)
go('L11 exclude 24h pump >30%', listing_short(exclude_pump=0.30))
go('L12 exclude pump + stop 40%', listing_short(exclude_pump=0.30), stop_pct=0.40)
go('L13 L0 with cap_short 5% (max 5% per name)', listing_short(), cap_short=0.05)
pd.DataFrame(rows).to_csv('out_exp7_listing.csv', index=False)
print('DONE')
