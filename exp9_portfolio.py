#!/usr/bin/env python3
"""Experiment 9: multi-sleeve portfolio (single netted book), OOS window 2022-01 -> 2026-08.
Sleeves (raw weights, each side ~1):
  A listing-drift short (age 3-60d, >=$20M, <=15 names) vs long majors
  B TS trend ensemble on top-10 majors (14/30/60/90d sign, inv-vol)
  C cross-sectional core336 (quantile 20%, liquid top-100)
  D ML ranker prediction panel (exp6, h=3) if available
Risk scalars equalise sleeve vol (from standalone dvol at gross 1); then portfolio VT, caps, stops.
Costs: fee 5.5 taker / 2.0 maker, maker share 70%, slip_model*0.3."""
import os, sys, time, numpy as np, pandas as pd
from bt import Data, run, quantile_ls, rank_weights, slip_model
from strategies import core_signal, mom_signal

START, END = '2021-01-01', '2026-08-31 23:00'
OOS = '2022-01-01'
d = Data(start=START, end=END, min_turn_ever=5e6, load_hl=True, load_premium=True)
T, N = d.ret.shape
MAJ5 = [s for s in ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT'] if s in d.cols]
jm = np.array([int(np.flatnonzero(d.cols == s)[0]) for s in MAJ5])
P = np.load('out_exp6_pred_h3.npy') if os.path.exists('out_exp6_pred_h3.npy') else None
if P is not None and P.shape != (T, N):
    print('WARN: ML panel shape mismatch', P.shape, (T, N)); P = None
rows = []


def report(label, m, t0):
    by = m['by_year']
    rows.append(dict(label=label, sharpe=m['sharpe'], cagr=m['cagr'], dvol=m['dvol'], mdd=m['mdd'], calmar=m['calmar'],
                     worst=m['worst_day'], p1=m['p1'], turnover=m['turnover_x'], fees=m['fees'], slip=m['slip'], funding=m['funding'],
                     npos=m['avg_npos'], gross=m['avg_gross'], **{f'sh{y}': v[1] for y, v in by.items()}, **{f'r{y}': v[0] for y, v in by.items()}))
    print(f'{label:<50} Sh {m["sharpe"]:5.2f} CAGR {m["cagr"]*100:6.1f}% dvol {m["dvol"]*100:.2f}% MDD {m["mdd"]*100:6.1f}% '
          f'wd {m["worst_day"]*100:5.1f}% p1 {m["p1"]*100:5.2f}% turn {m["turnover_x"]:4.0f}x cost {(m["fees"]+m["slip"])*100:.1f}% fund {-m["funding"]*100:+.1f}% '
          f'npos {m["avg_npos"]:.0f} gross {m["avg_gross"]:.2f} | ' + ' '.join(f'{y}:{v[1]:.1f}' for y, v in by.items()) + f'  ({time.time()-t0:.0f}s)', flush=True)


# ---- sleeves
def sl_listing(dd, i, mask, lo=72, hi=24 * 60, thr=2e7, n_max=15):
    age = dd.age[i]; t = dd.t24[i]
    cand = (age >= lo) & (age <= hi) & dd.valid[i] & np.isfinite(t) & (t >= thr)
    cand[jm] = False
    r24 = dd.cff[i] / dd.cff[i - 24] - 1
    cand &= ~(r24 > 0.30)
    idx = np.flatnonzero(cand)
    w = np.zeros(N)
    if len(idx) == 0:
        return w
    idx = idx[np.argsort(-t[idx])][:n_max]
    w[idx] = -1.0 / len(idx)
    w[jm] += 1.0 / len(jm)
    return w


def sl_ts(dd, i, mask, lbs=(336, 720, 1440, 2160), n_top=10):
    t = dd.t24[i]
    ok = dd.valid[i] & np.isfinite(t) & (dd.age[i] >= 24 * 120)
    idx = np.flatnonzero(ok); idx = idx[np.argsort(-t[idx])][:n_top]
    w = np.zeros(N)
    if len(idx) == 0:
        return w
    sig = np.zeros(len(idx))
    for lb in lbs:
        sig += np.sign(np.log(dd.cff[i, idx] / dd.cff[i - lb, idx]))
    sig /= len(lbs)
    vol = np.nanstd(dd.ret[i - 720:i + 1][:, idx], 0)
    w[idx] = sig / np.maximum(vol, 1e-6)
    s = np.abs(w).sum()
    return w / s if s > 0 else w


def sl_core(dd, i, mask):
    m = dd.universe(i, min_age_h=720, min_turn=1e7, top_n=100)
    return quantile_ls(core_signal(dd, i, m, 336), 0.2)


def sl_ml(dd, i, mask):
    m = dd.universe(i, min_age_h=720, min_turn=1e7, top_n=150)
    s = P[i]
    s = np.where(m & np.isfinite(s), s, np.nan)
    return rank_weights(s, m)


SCALE = {'A': 2.6, 'B': 0.45, 'C': 0.85, 'D': 0.85}      # ~equal-vol scalars (standalone dvol 0.38/2.3/1.2/1.2)


def combo(parts):
    def fn(dd, i, mask):
        w = np.zeros(N)
        for k in parts:
            if k == 'A': w += SCALE['A'] * sl_listing(dd, i, mask)
            if k == 'B': w += SCALE['B'] * sl_ts(dd, i, mask)
            if k == 'C': w += SCALE['C'] * sl_core(dd, i, mask)
            if k == 'D' and P is not None: w += SCALE['D'] * sl_ml(dd, i, mask)
        return w / len(parts)
    return fn


def go(label, fn, **kw):
    t0 = time.time()
    d.start_i = int(d.idx.searchsorted(pd.Timestamp(OOS, tz='UTC')))
    args = dict(reb_h=24, band=0.3, smooth=0.5, fee_bps=5.5, maker_share=0.7, slip_fn=slip_model(scale=0.3),
                gross=1.0, pos_cap=0.02, cap_short=0.015, cap_exempt=MAJ5, cap_exempt_val=0.30,
                uni_kwargs=dict(min_age_h=0, min_turn=0, top_n=0), verbose=False, label=label)
    args.update(kw)
    m, eq = run(d, fn, **args)
    report(label, m, t0)
    return m, eq


# standalone sleeves (same OOS window, same execution assumptions)
go('A listing', combo('A'))
go('B TS ens', combo('B'))
go('C core336', combo('C'))
if P is not None:
    go('D ML h3', combo('D'))
# combinations
go('AB', combo('AB'))
go('ABC', combo('ABC'))
if P is not None:
    go('ABD', combo('ABD'))
    go('ABCD', combo('ABCD'))
    best = 'ABCD'
else:
    best = 'ABC'
# risk overlays on the best combo
m, eq = go(f'{best} VT0.6 maxlev3', combo(best), vol_target=0.006, max_lev=3.0)
go(f'{best} VT0.6 maxlev3 stop30', combo(best), vol_target=0.006, max_lev=3.0, stop_pct=0.30)
go(f'{best} VT0.6 maxlev3 stop30 dailystop2%', combo(best), vol_target=0.006, max_lev=3.0, stop_pct=0.30, daily_stop=0.02)
m, eq, hp = run(d, combo(best), reb_h=24, band=0.3, smooth=0.5, fee_bps=5.5, maker_share=0.7, slip_fn=slip_model(scale=0.3),
                gross=1.0, pos_cap=0.02, cap_short=0.015, cap_exempt=MAJ5, cap_exempt_val=0.30,
                uni_kwargs=dict(min_age_h=0, min_turn=0, top_n=0), vol_target=0.006, max_lev=3.0, stop_pct=0.30,
                label=f'{best} final', ret_diag=True, attrib=True)
eq.to_csv('out_exp9_eq.csv'); hp.to_csv('out_exp9_hourly_pnl.csv')
m['attrib'].to_csv('out_exp9_attrib.csv')
pd.DataFrame(rows).to_csv('out_exp9_portfolio.csv', index=False)
print('DONE')
