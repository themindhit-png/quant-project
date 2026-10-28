#!/usr/bin/env python3
"""Phase T: validate research/fastsim.py against the full harness runs of R3-bis (logs/r3b_ch_<P>_<VT>.csv, logs/r3b_fu_<P>_<VT>.csv):
same firm (mubite_8_dd10), same policies, same 123 starts; funded 90 days from the harness pass dates, payouts every 30 days uncapped.
Prints per variant: P(both phases), halts, median days, funded paid — harness vs fastsim — and the per-start agreement of outcomes.
Run: python3 research/fastsim_check.py"""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fastsim as fs

books = fs.load_books()
ts, R, G, V, hour, day, RH = books
F_CH = fs.fvec(T1=8, T2=5, TOTAL_DD=10, DAILY_DD=5, MIN_D1=10, MIN_D2=3, NOTIONAL_X=3)
F_FU = fs.fvec(T1=1e9, T2=1e9, TOTAL_DD=10, DAILY_DD=5, NOTIONAL_X=3, PAY_EVERY=30, MIN_PAY=50, NOCAP=1)
POL = dict(P0=(dict(CPPI=0.25, ABANDON=0.30), dict(CPPI=0.5)), P1=(dict(CPPI=0.25), dict(CPPI=0.25)),
           P2=(dict(CPPI=0.5, KMIN=0.3), dict(CPPI=0.5, KMIN=0.3)),
           P3=(dict(CPPI=0.5, KMIN=0.3, DD_BUF=0.5), dict(CPPI=0.5, KMIN=0.3, DD_BUF=0.5)))
OUT = ('end', 'pass', 'halt', 'breach')
rows = []
for P, (pc, pf) in POL.items():
    for vt in ('0.004', '0.006', '0.008'):
        tag = f'{P}_{vt}'
        if not os.path.exists(f'logs/r3b_ch_{tag}.csv'):
            continue
        h = pd.read_csv(f'logs/r3b_ch_{tag}.csv'); hf = pd.read_csv(f'logs/r3b_fu_{tag}.csv')
        p_ch = fs.pvec(VT=float(vt), BASE=2.0, **pc); p_fu = fs.pvec(VT=float(vt), BASE=1.25, **pf)
        res = []
        for s in h.start:
            i0 = max(int(ts.searchsorted(pd.Timestamp(s, tz=ts.tz))), 0)
            o1, ie1, *_ = fs.sim(R, G, V, RH, hour, day, i0, p_ch, F_CH, 0, 1, 0, 10000.0)
            r = dict(p1=OUT[o1], d1=(ie1 - i0) / 24)
            if o1 == 1:
                o2, ie2, *_ = fs.sim(R, G, V, RH, hour, day, ie1 + 1, p_ch, F_CH, 0, 2, 0, 10000.0)
                r.update(p2=OUT[o2], d=(ie2 - i0) / 24)
            res.append(r)
        f = pd.DataFrame(res)
        hp = (h.p2 == 'pass'); fp = (f.get('p2') == 'pass')
        hh = (h.p1 == 'halt') | (h.p2 == 'halt'); fh = f.p1.isin(['halt', 'breach']) | f.get('p2', pd.Series(index=f.index)).isin(['halt', 'breach'])
        paid_f = []
        hpass = h[hp]; pdates = pd.to_datetime(hpass.start) + pd.to_timedelta(hpass.total_days.round(), unit='D')
        for dte in pdates[pdates <= '2026-07-26']:
            i0 = int(ts.searchsorted(pd.Timestamp(dte, tz=ts.tz)))
            o, ie, mn, ee, paid, npay, _ = fs.sim(R, G, V, RH, hour, day, i0, p_fu, F_FU, 1, 1, 90 * 24, 10000.0)
            paid_f.append(paid)
        rows.append(dict(var=tag, P_h=round(hp.mean(), 3), P_f=round(fp.mean(), 3), fail_h=int(hh.sum()), fail_f=int(fh.sum()),
                         agree=round((hp.values == fp.values).mean(), 3), days_h=h.total_days.median(), days_f=round(f.d.median(), 0),
                         paid_h=round(hf.paid.mean(), 0), paid_f=round(np.mean(paid_f), 0) if paid_f else np.nan))
df = pd.DataFrame(rows)
print(df.to_string(index=False))
print(f"\ncorr across variants: P(pass) {np.corrcoef(df.P_h, df.P_f)[0, 1]:.2f}, fails {np.corrcoef(df.fail_h, df.fail_f)[0, 1]:.2f}, "
      f"paid {np.corrcoef(df.paid_h, df.paid_f)[0, 1]:.2f}")
