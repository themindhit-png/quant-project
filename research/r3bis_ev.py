#!/usr/bin/env python3
"""R3-bis: EV per challenge purchase for drawdown policies P0-P3 x VT on the V2 book (same formula as research/r3_ev.py):
EV = P(pass both phases) x E[payouts of the first 3 months funded] x 0.8 - 170 $. Inputs: logs/r3b_ch_<P>_<VT>.csv, logs/r3b_fu_<P>_<VT>.csv
(research/r3bis_run.sh). Pass dates after 2026-07-26 get the variant's mean funded payout (imputed). Unfinished starts count as failed.
Run: python3 research/r3bis_ev.py  -> table + research/phase4_out/R3bis_summary.csv"""
import os, pandas as pd, numpy as np
FEE, SHARE, CUT = 170.0, 0.8, '2026-07-26'
POL, VTS = ('P0', 'P1', 'P2', 'P3'), ('0.004', '0.006', '0.008')


def load(tag):
    ch = pd.read_csv(f'logs/r3b_ch_{tag}.csv'); fu = pd.read_csv(f'logs/r3b_fu_{tag}.csv')
    ch['passed'] = ch.p2 == 'pass'; ch['pdate'] = pd.to_datetime(ch.start) + pd.to_timedelta(ch.total_days.round(), unit='D')
    ok = ch.passed & (ch.pdate <= CUT); assert ok.sum() == len(fu), (tag, ok.sum(), len(fu))
    ch['paid'] = np.nan; ch.loc[ok, 'paid'] = fu.paid.values
    ch['fu_halt'] = np.nan; ch.loc[ok, 'fu_halt'] = (fu.p1 == 'halt').astype(float).values
    ch['imputed'] = ch.passed & ~ok; ch.loc[ch.imputed, 'paid'] = fu.paid.mean(); ch.loc[~ch.passed, 'paid'] = 0.0
    ch['ev'] = ch.paid * SHARE - FEE
    return ch


def summ(ch):
    p = ch.passed
    return dict(n=len(ch), P_pass=round(p.mean(), 3), halt_p1=round((ch.p1 == 'halt').mean(), 3), halt_p2=round((ch.p2 == 'halt').mean(), 3),
                days_med=round(ch.loc[p, 'total_days'].median(), 0), pay3m=round(ch.loc[p, 'paid'].mean(), 0),
                fu_halt=round(ch.fu_halt.mean(), 3), EV=round(ch.ev.mean(), 0), EV_se=round(ch.ev.std(ddof=1) / np.sqrt(len(ch)), 0))


rows = []
for P in POL:
    for vt in VTS:
        tag = f'{P}_{vt}'
        if not (os.path.exists(f'logs/r3b_ch_{tag}.csv') and os.path.exists(f'logs/r3b_fu_{tag}.csv')):
            continue
        ch = load(tag)
        for lab, sel in (('all', ch.start >= '2022'), ('2022-24', ch.start < '2025'), ('2025-26', ch.start >= '2025'), ('2026', ch.start >= '2026')):
            rows.append(dict(policy=P, vt=vt, set=lab, **summ(ch[sel])))
df = pd.DataFrame(rows)
df.to_csv('research/phase4_out/R3bis_summary.csv', index=False)
pd.set_option('display.width', 200)
print(df.to_string(index=False))
print('\nEV $ (all / 2026):')
print(df.pivot_table(index='policy', columns=['set', 'vt'], values='EV').loc[:, ['all', '2026']].to_string())
