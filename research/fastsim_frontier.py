#!/usr/bin/env python3
"""Phase T: risk-budget frontier for the consensus challenge settings (from research/fastsim_grid.py marginals: CPPI 1.0, daily throttle 0.5 %,
floor guard 3 %, drawdown mode at 30 % buffer, K_MAX 1.5x base, V2 book). Scans VT x BASE on the four Mubite Two-Step Pro variants:
fail share, worst margin to the firm floor over all starts (% of initial), median / p75 days to pass both phases.
Run: python3 research/fastsim_frontier.py"""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fastsim as fs

ts, R, G, V, hour, day, RH = fs.load_books()
st = fs.starts_idx(ts, every_days=7)
CONS = dict(KMAXR=1.5, CPPI=1.0, KMIN=0.0, DD_BUF=0.3, THROTTLE=0.5, HALT_FRAC=0.5, FG=0.03, BOOK=0, RAT=0.35, MAXG=2.0, P2R=1.0)


def run(vt, base, t1, dd, cons=CONS):
    c = dict(cons)
    p = fs.pvec(VT=vt, BASE=base, BASE_P2=base * c['P2R'], KMAX=base * c['KMAXR'] if c['KMAXR'] > 1 else 0.0, CPPI=c['CPPI'], KMIN=c['KMIN'],
                DD_BUF=c['DD_BUF'], THROTTLE=c['THROTTLE'], HALT_FRAC=c['HALT_FRAC'], FG=c['FG'], BOOK=c['BOOK'], RAT=c['RAT'], MAXG=c['MAXG'])
    f = fs.fvec(T1=t1, T2=5, TOTAL_DD=dd, DAILY_DD=5, MIN_D1=10, MIN_D2=3, PROF_D1=3, PROF_D2=2, NOTIONAL_X=100)
    fail = 0; days = []; margin = []; T = R.shape[2]
    for i0 in st:
        o1, ie1, mn1, *_ = fs.sim(R, G, V, RH, hour, day, i0, p, f, 0, 1, 0, 10000.0)
        margin.append((mn1 - (10000 - dd * 100)) / 100)
        if o1 in (2, 3):
            fail += 1; continue
        if o1 != 1 or ie1 + 48 >= T - 48:
            continue
        o2, ie2, mn2, *_ = fs.sim(R, G, V, RH, hour, day, ie1 + 48, p, f, 0, 2, 0, 10000.0)
        margin.append((mn2 - (10000 - dd * 100)) / 100)
        if o2 in (2, 3):
            fail += 1
        elif o2 == 1:
            days.append((ie2 - i0) / 24)
    return dict(fail=fail, n=len(st), worst_margin_pct=round(min(margin), 2), p5_margin=round(np.percentile(margin, 5), 2),
                days_med=round(np.median(days)), days_p75=round(np.percentile(days, 75)), n_pass=len(days))


if __name__ == '__main__':
    rows = []
    for t1, dd in ((10, 8), (10, 10), (8, 8), (8, 10)):
        for vt in (0.004, 0.005, 0.006, 0.007, 0.008):
            for base in (1.5, 2.0, 2.5):
                rows.append(dict(acct=f'T{t1}_DD{dd}', VT=vt, BASE=base, risk=round(vt * base * 100, 2), **run(vt, base, t1, dd)))
    df = pd.DataFrame(rows); df.to_csv(os.path.join(fs.ROOT, 'research', 'phase4_out', 'T_frontier_challenge.csv'), index=False)
    pd.set_option('display.width', 200)
    print(df.sort_values(['acct', 'risk']).to_string(index=False))
