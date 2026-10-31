"""D_ audit: listing-sleeve candidates whose TRUE Binance age (first bar in the full 2020-01 panel / meta.json) is > 90 days
('fake new listings' created by segment resets), over the OOS period. Also P&L of those names inside the listing sleeve."""
import os, sys, json, numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, ROOT); os.chdir(ROOT)
import exp_common as X
from exp_common import d, N, sl_listing, launch_h, jm
meta = json.load(open('data/panels/meta.json'))
first_h = np.array([pd.Timestamp(meta[s]['first']).value / 3.6e12 if s in meta else np.nan for s in d.cols])
i0 = int(d.idx.searchsorted(pd.Timestamp(X.OOS, tz='UTC')))
rows = []; nreal = 0
for i in range(i0, len(d.idx) - 1):
    if d.hh[i] % 24: continue
    w = sl_listing(d, i, None); sh = np.flatnonzero(w < 0)
    true_age_d = (d.hh[i] - first_h[sh]) / 24
    for j, a in zip(sh, true_age_d):
        if a > 91: rows.append((str(d.idx[i].date()), d.cols[j], round(a), int(d.age[i, j] / 24), round((d.hh[i] - launch_h[j]) / 24) if np.isfinite(launch_h[j]) else None, -w[j]))
        else: nreal += 1
df = pd.DataFrame(rows, columns=['date', 'sym', 'true_binance_age_d', 'segment_age_d', 'bybit_age_d', 'w'])
print('real candidate-days:', nreal, ' fake candidate-days (true age > 91d):', len(df))
if len(df):
    g = df.groupby('sym').agg(first=('date', 'min'), last=('date', 'max'), days=('date', 'count'), true_age=('true_binance_age_d', 'min'), bybit_age=('bybit_age_d', 'min'))
    # price move while held as fake candidate (short): return from first to last+1 day
    rr = []
    for s, r in g.iterrows():
        j = int(np.flatnonzero(d.cols == s)[0]); a = d.idx.searchsorted(pd.Timestamp(r['first'], tz='UTC')); b = min(d.idx.searchsorted(pd.Timestamp(r['last'], tz='UTC')) + 24, len(d.idx) - 1)
        rr.append(d.cff[b, j] / d.cff[a, j] - 1)
    g['px_ret_first_to_last'] = np.round(rr, 3)
    print(g.sort_values('first').to_string())
    df.to_csv('logs/D_listing_fake.csv', index=False)
