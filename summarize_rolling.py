#!/usr/bin/env python3
"""Summarise rolling-start bot simulations from logs/roll*.log (each line a dict per start date)."""
import glob, os, ast, pandas as pd, numpy as np
rows = []
for f in sorted(glob.glob('logs/roll*.log')):
    recs = []
    for line in open(f):
        line = line.strip()
        if line.startswith('{') and line.endswith('}'):
            try:
                recs.append(ast.literal_eval(line))
            except Exception:
                pass
    if not recs:
        continue
    df = pd.DataFrame(recs); label = os.path.basename(f).replace('.log', '')
    if 'paid' in df:
        rows.append(dict(config=label, n=len(df), alive=int((df.p1 != 'halt').sum()), paid_pct_yr_med=df.paid_pct_yr.median(),
                         paid_pct_yr_mean=df.paid_pct_yr.mean(), min_eq=df.p1_min_eq.min(), worst_start=df.loc[df.p1_min_eq.idxmin(), 'start']))
    else:
        ok = df[df.p1 == 'pass']
        both = df.dropna(subset=['total_days']) if 'total_days' in df else pd.DataFrame()
        p2 = df.p2.value_counts().to_dict() if 'p2' in df else {}
        rows.append(dict(config=label, n=len(df), p1_pass=len(ok), p1_halt=int((df.p1 == 'halt').sum()), p1_med=ok.p1_days.median(), p1_p75=ok.p1_days.quantile(.75),
                         p1_max=ok.p1_days.max(), p2_pass=p2.get('pass', 0), p2_halt=p2.get('halt', 0), p2_end=p2.get('end', 0),
                         total_med=both.total_days.median() if len(both) else np.nan, total_p75=both.total_days.quantile(.75) if len(both) else np.nan,
                         total_max=both.total_days.max() if len(both) else np.nan, min_eq=df.p1_min_eq.min(),
                         halts_at=','.join(df.loc[(df.p1 == 'halt') | (df.get('p2', pd.Series(index=df.index, dtype=object)) == 'halt'), 'start'].astype(str).tolist())))
pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 60)
print(pd.DataFrame(rows).round(0).to_string(index=False))
