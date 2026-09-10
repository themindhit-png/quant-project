#!/usr/bin/env python3
"""From rolling real-path outcomes (logs/roll*.log) estimate time-to-funded WITH restarts: each attempt's outcome
(pass in T days / fail at F days) is drawn from the empirical set of start dates; a failed attempt costs the fee
and 3 days; repeat until funded. Reports median/p75/p90 days, mean attempts, mean fees, P(funded <= 90/180/365)."""
import sys, glob, os, ast, numpy as np, pandas as pd
FEE = float(sys.argv[1]) if len(sys.argv) > 1 else 154.0
rng = np.random.default_rng(1)
rows = []
for f in sorted(glob.glob('logs/roll*.log')):
    recs = [ast.literal_eval(l.strip()) for l in open(f) if l.strip().startswith('{')]
    if not recs or 'paid' in recs[0] or 'p1_days' not in recs[0]:
        continue
    outcomes = []
    for r in recs:
        if r['p1'] != 'pass':
            outcomes.append(('fail', r['p1_days'])); continue
        if r.get('p2') == 'pass':
            outcomes.append(('pass', r['total_days']))
        elif r.get('p2') in ('halt', 'timeout'):
            outcomes.append(('fail', r['p1_days'] + r['p2_days']))
        # 'end' (data ran out) -> ignore
    if len(outcomes) < 10:
        continue
    N = 20000; days = np.zeros(N); att = np.zeros(N); fees = np.zeros(N)
    idx = rng.integers(0, len(outcomes), size=(N, 30))
    for n in range(N):
        t = 0.0; a = 0
        for k in range(30):
            o, dur = outcomes[idx[n, k]]; a += 1; t += dur
            if o == 'pass':
                break
            t += 3
        days[n] = t; att[n] = a; fees[n] = a * FEE
    label = os.path.basename(f).replace('.log', '')
    p_pass = np.mean([o == 'pass' for o, _ in outcomes])
    rows.append(dict(config=label, attempts_obs=len(outcomes), p_pass_attempt=round(p_pass, 2), med=np.median(days), p75=np.quantile(days, .75),
                     p90=np.quantile(days, .9), P90d=np.mean(days <= 90), P180d=np.mean(days <= 180), P365d=np.mean(days <= 365),
                     attempts=att.mean(), fees=fees.mean()))
pd.set_option('display.width', 220)
df = pd.DataFrame(rows)
for c in ('P90d', 'P180d', 'P365d'):
    df[c] = (df[c] * 100).round(0)
print(df.round(1).to_string(index=False))
