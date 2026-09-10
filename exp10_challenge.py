#!/usr/bin/env python3
"""Experiment 10: prop-challenge Monte-Carlo on the portfolio's OOS hourly returns (exp9 output).
For each firm preset and risk scale: pass probability, failure modes, median/p90 days to pass,
funded-account 1y survival and payout. Block bootstrap (5-day blocks) preserves clustering.
Scale s multiplies hourly returns (leverage knob); base series is the VT 0.6%/day portfolio."""
import sys, numpy as np, pandas as pd
from challenge_sim import run_challenge, run_funded, RULES

import os
src = sys.argv[1] if len(sys.argv) > 1 else ('out_exp9b_eq.csv' if os.path.exists('out_exp9b_eq.csv') else 'out_exp9_eq.csv')
print('source:', src)
eq = pd.read_csv(src, index_col=0, parse_dates=True).iloc[:, 0].dropna()
hr = eq.pct_change().dropna().values
print(f'series: {len(hr)} hours, dvol {hr.reshape(-1,24).sum(1).std()*100 if len(hr)%24==0 else np.std(hr)*np.sqrt(24)*100:.2f}%/day, '
      f'ann ret {hr.mean()*24*365*100:.1f}%')
rows = []
for firm in ['hyro_2step', 'hyro_2step_swing', 'hyro_1step', 'mubite_2step', 'mubite_2step_10']:
    for s in (0.5, 0.75, 1.0, 1.5, 2.0, 3.0):
        summ, df = run_challenge(hr, firm, scale=s, n_paths=1500, max_days=400, block_days=5, seed=7)
        fs, fdf = run_funded(hr, firm, scale=s, n_paths=800, days=365, block_days=5, seed=11)
        row = dict(firm=firm, scale=s, pass_=summ.get('pass', 0), fail_daily=summ.get('fail_daily', 0),
                   fail_total=summ.get('fail_total', 0), timeout=summ.get('timeout', 0),
                   med_days=summ.get('median_days_pass'), p90_days=summ.get('p90_days_pass'),
                   funded_surv_1y=fs['survival'], funded_paid_mean=fs['mean_paid'], funded_paid_med=fs['median_paid'])
        rows.append(row)
        print(f'{firm:<18} scale {s:3.2f} | pass {row["pass_"]*100:5.1f}%  fail_daily {row["fail_daily"]*100:4.1f}%  fail_total {row["fail_total"]*100:4.1f}%  '
              f'timeout {row["timeout"]*100:4.1f}%  med days {row["med_days"]:6.0f}  p90 {row["p90_days"]:6.0f} | funded 1y surv {fs["survival"]*100:5.1f}%  '
              f'paid mean {fs["mean_paid"]*100:5.1f}% med {fs["median_paid"]*100:5.1f}%', flush=True)
pd.DataFrame(rows).to_csv('out_exp10_challenge.csv', index=False)
print('DONE')
