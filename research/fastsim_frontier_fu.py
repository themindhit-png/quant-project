#!/usr/bin/env python3
"""Phase T: funded frontier (Mubite Two-Step Pro funded: static DD 8/10 %, daily 5 %, notional <= 3x initial, payouts every 14 days with caps
2.5 -> 5 %, Funded Score < 40 %, min $50) for the consensus settings (CPPI 0.75, throttle 0.5 %, K_MAX 1.5x, floor guard 3 %, keep 2 % buffer,
V2 book). Weekly starts 2022-01..2025-10, 365 days. Per VT x BASE: deaths, worst margin to floor, payouts mean / p10 (% of account, gross of
the 80 % split), share of 90-day windows with >= 10 % cumulative profit (scaling plan: +25 % of the account).
Run: python3 research/fastsim_frontier_fu.py"""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fastsim as fs

ts, R, G, V, hour, day, RH = fs.load_books()
st = fs.starts_idx(ts, last='2025-10-20', every_days=7)
CONS = dict(KMAXR=1.5, CPPI=0.75, KMIN=0.0, DD_BUF=0.3, THROTTLE=0.5, HALT_FRAC=0.5, FG=0.03, BOOK=0, KEEP_BUF=0.02, MAXG=2.0)


def run(vt, base, dd, cons=CONS, days=365):
    c = dict(cons)
    p = fs.pvec(VT=vt, BASE=base, KMAX=base * c['KMAXR'] if c['KMAXR'] > 1 else 0.0, CPPI=c['CPPI'], KMIN=c['KMIN'], DD_BUF=c['DD_BUF'],
                THROTTLE=c['THROTTLE'], HALT_FRAC=c['HALT_FRAC'], FG=c['FG'], BOOK=c['BOOK'], KEEP_BUF=c['KEEP_BUF'], MAXG=c['MAXG'])
    f = fs.fvec(T1=1e9, T2=1e9, TOTAL_DD=dd, DAILY_DD=5, NOTIONAL_X=3, PAY_EVERY=14, FSCORE=40, MIN_PAY=50)
    dead = 0; paid = []; margin = []; scal = []; T = R.shape[2]
    for i0 in st:
        w = np.zeros(T)
        o, ie, mn, ee, pd_, npay, fsb = fs.sim(R, G, V, RH, hour, day, i0, p, f, 1, 1, days * 24, 10000.0, np.zeros(1), w)
        dead += o in (2, 3); paid.append(pd_ / 100); margin.append((mn - (10000 - dd * 100)) / 100)
        ww = pd.Series(w[i0 + 1:ie + 1]).iloc[::24].values / 10000.0          # daily wealth / initial
        if len(ww) > 90:
            scal.append(np.mean(ww[90:] - ww[:-90] >= 0.10))
    return dict(dead=dead, n=len(st), worst_margin=round(min(margin), 2), paid_pct=round(np.mean(paid), 1), paid_p10=round(np.percentile(paid, 10), 1),
                paid_min=round(min(paid), 1), scale90=round(np.mean(scal), 3))


if __name__ == '__main__':
    rows = []
    for dd in (8, 10):
        for vt in (0.004, 0.005, 0.006, 0.007, 0.008, 0.010):
            for base in (1.0, 1.5, 2.0):
                rows.append(dict(dd=dd, VT=vt, BASE=base, risk=round(vt * base * 100, 2), **run(vt, base, dd)))
    df = pd.DataFrame(rows); df.to_csv(os.path.join(fs.ROOT, 'research', 'phase4_out', 'T_frontier_funded.csv'), index=False)
    pd.set_option('display.width', 200)
    print(df.sort_values(['dd', 'risk']).to_string(index=False))
