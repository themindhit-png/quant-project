"""D_ audit: F2 book variants. Env: GAP_H (bt segment-break threshold, default 24), VARIANTS (comma list).
Run from repo root: GAP_H=100 python3 research/audit_f2/D_variants.py"""
import os, sys, numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, ROOT); os.chdir(ROOT)
import bt
bt.GAP_H = int(os.environ.get('GAP_H', '24'))
if os.environ.get('PAN'): bt.PAN = os.environ['PAN']
import exp_common as X
from exp_common import d, N, F2, BASE8, combo, Held, sl_listing, sl_core, ML3n, ML7n, ML8, fund_change, fund_level
from bt import run, slip_model
TAG = os.environ.get('TAG', f'gap{bt.GAP_H}')
def lag1(f0):
    def fn(dd, i, mask): return f0(dd, i - 1, mask)
    return fn
def go(label, fn, lag=0, **kw):
    args = dict(BASE8); args.update(kw); args['label'] = label
    if lag: args['reb_offset'] = 1; fn = lag1(fn)
    d.start_i = int(d.idx.searchsorted(pd.Timestamp(X.OOS, tz='UTC')))
    m, eq, hp = run(d, fn, ret_diag=True, **args)
    de = eq.groupby(eq.index.floor('D')).last(); r = de.pct_change().dropna()
    print(f'{TAG:<8} {label:<34} Sh {m["sharpe"]:5.2f} SEL {X.sh(r[:"2024-12-31"]):5.2f} CONF {X.sh(r["2025-01-01":]):5.2f} CAGR {m["cagr"]*100:5.1f}% MDD {m["mdd"]*100:5.1f}% wd {m["worst_day"]*100:5.1f}% '
          f'fund {m["funding"]*100:+5.1f}% delist$ {m["delist_usd"]:7.0f} | ' + ' '.join(f'{y}:{v[1]:.2f}' for y, v in m['by_year'].items()), flush=True)
    return m, r
V = os.environ.get('VARIANTS', 'F2,F2_lag1,F2_noshock,listing,listing_attr').split(',')
out = {}
for v in V:
    if v == 'F2': m, r = go('F2 lag0', F2())
    elif v == 'F2_lag1': m, r = go('F2 lag1', F2(), lag=1)
    elif v == 'F2_noshock': m, r = go('F2 lag0 delist_shock=0', F2(), delist_shock=0.0)
    elif v == 'listing': m, r = go('listing sleeve alone', Held(sl_listing))
    elif v == 'no_listing': m, r = go('F2 minus listing', combo([Held(sl_core), Held(ML3n), Held(ML7n), ML8, Held(fund_change), fund_level], [1, 1, 2, 1, 1][:0] or [1, 1, 1, 2, 1, 1]))
    elif v == 'F2_lag1_mk33': m, r = go('F2 lag1 maker33', F2(), lag=1, maker_share=0.33)
    elif v == 'F2_lag1_mk33_adv': m, r = go('F2 lag1 maker33 adv1bp', F2(), lag=1, maker_share=0.33, maker_adverse_bps=1.0)
    elif v == 'listing_attr':
        m, r = go('F2 attrib', F2(), attrib=True, attrib_from='2022-01-01')
        A = m['attrib']; A['tot'] = A.price + A.funding - A.cost
        A.sort_values('tot').to_csv(f'logs/D_attrib_{TAG}.csv')
        print('top 12 symbols by total P&L ($):'); print(A.sort_values('tot', ascending=False).head(12)[['price', 'funding', 'cost', 'tot', 'hours_held']].round(0).to_string())
        print('bottom 8:'); print(A.sort_values('tot').head(8)[['price', 'funding', 'cost', 'tot', 'hours_held']].round(0).to_string())
        continue
    out[v] = r
pd.DataFrame(out).to_csv(f'logs/D_variants_{TAG}.csv')
