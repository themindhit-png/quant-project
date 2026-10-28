#!/usr/bin/env python3
"""R3: EV per challenge purchase = P(pass both phases) x E[payouts of first 3 months funded x 0.8] - fee. Inputs: logs/r3_challenge_<m>.csv, logs/r3_funded_<tag>.csv
(harness rolling runs; funded starts = challenge pass dates <= 2026-07-26, same order). Pass dates later than the cutoff get the variant's mean funded payout (imputed, flagged).
Run: python3 research/r3_ev.py"""
import pandas as pd, numpy as np
FEE, SHARE, ACC, CUT = 170.0, 0.8, 10000.0, '2026-07-26'
def load(m, tag):
    ch = pd.read_csv(f'logs/r3_challenge_{m}.csv'); fu = pd.read_csv(f'logs/r3_funded_{tag}.csv')
    ch['passed'] = ch.p2 == 'pass'; ch['pdate'] = pd.to_datetime(ch.start) + pd.to_timedelta(ch.total_days.round(), unit='D')
    ok = ch.passed & (ch.pdate <= CUT); assert ok.sum() == len(fu), (ok.sum(), len(fu))
    ch['paid'] = np.nan; ch.loc[ok, 'paid'] = fu.paid.values; ch['alive'] = np.nan; ch.loc[ok, 'alive'] = (fu.p1 != 'halt').values
    ch['imputed'] = ch.passed & ~ok; ch.loc[ch.imputed, 'paid'] = fu.paid.mean(); ch.loc[~ch.passed, 'paid'] = 0.0
    ch['ev'] = ch.paid * SHARE - FEE
    return ch
def summ(ch, label):
    n = len(ch); pp = ch.passed.mean(); pay = ch.loc[ch.passed, 'paid'].mean()
    return dict(set=label, n=n, P_pass=round(pp, 3), halt_p1=round((ch.p1 == 'halt').mean(), 3), pay3m_usd=round(pay, 0), EV_usd=round(ch.ev.mean(), 0), imputed=int(ch.imputed.sum()))
rows = []
for name, m, tag in (('base', '1.0', 'base'), ('V2', '0.5', 'v2')):
    ch = load(m, tag); ch.to_csv(f'research/phase4_out/R3_{name}.csv', index=False)
    for lab, sel in (('all 2022-26', ch.start >= '2022'), ('2022-24', ch.start < '2025'), ('2025-26', ch.start >= '2025'), ('2026', ch.start >= '2026'), ('2026 no-impute', (ch.start >= '2026') & ~ch.imputed)):
        rows.append(dict(book=name, **summ(ch[sel], lab)))
print(pd.DataFrame(rows).to_string(index=False))
