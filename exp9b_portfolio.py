#!/usr/bin/env python3
"""Experiment 9b: portfolio v2 (single netted book), OOS 2022-01 -> 2026-08.
Sleeves: A listing-drift short (risk-sized, stops, pump filter) vs long majors;
         C xsec core336 quantile 20% (liquid top-100);
         D ML ranker, funding-aware target, h=3 and h=7 (rank^2 weights, top/bottom 30%).
Execution: fee 5.5 taker / 2.0 maker, maker share 70%, slip_model*0.3, band 30%, smooth 0.5.
Risk: caps 2%/1.5% (majors 30%), VT 0.6%/day, max lev 3, stop 30%, daily stop 2%."""
import os, time, numpy as np, pandas as pd
from bt import Data, run, quantile_ls, rank_weights, slip_model
from strategies import core_signal

START, END, OOS = '2021-01-01', '2026-08-31 23:00', '2022-01-01'
d = Data(start=START, end=END, min_turn_ever=5e6, load_hl=True, load_premium=True)
T, N = d.ret.shape
MAJ5 = [s for s in ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT'] if s in d.cols]
jm = np.array([int(np.flatnonzero(d.cols == s)[0]) for s in MAJ5])
P3 = np.load('out_exp6_pred_h3.npy') if os.path.exists('out_exp6_pred_h3.npy') else None
P7 = np.load('out_exp6_pred_h7.npy') if os.path.exists('out_exp6_pred_h7.npy') else None
for nm, P in (('P3', P3), ('P7', P7)):
    if P is not None and P.shape != (T, N):
        print(f'WARN {nm} shape mismatch');
rows = []


def report(label, m, t0):
    by = m['by_year']
    rows.append(dict(label=label, sharpe=m['sharpe'], cagr=m['cagr'], dvol=m['dvol'], mdd=m['mdd'], calmar=m['calmar'],
                     worst=m['worst_day'], p1=m['p1'], turnover=m['turnover_x'], fees=m['fees'], slip=m['slip'], funding=m['funding'],
                     npos=m['avg_npos'], gross=m['avg_gross'], stops=m.get('n_stops', 0),
                     **{f'sh{y}': v[1] for y, v in by.items()}, **{f'r{y}': v[0] for y, v in by.items()}))
    print(f'{label:<46} Sh {m["sharpe"]:5.2f} CAGR {m["cagr"]*100:6.1f}% dvol {m["dvol"]*100:.2f}% MDD {m["mdd"]*100:6.1f}% '
          f'wd {m["worst_day"]*100:5.1f}% p1 {m["p1"]*100:5.2f}% turn {m["turnover_x"]:4.0f}x cost {(m["fees"]+m["slip"])*100:.1f}% fund {-m["funding"]*100:+.1f}% '
          f'npos {m["avg_npos"]:.0f} gross {m["avg_gross"]:.2f} stops {m.get("n_stops",0)} | ' + ' '.join(f'{y}:{v[1]:.1f}' for y, v in by.items()) + f'  ({time.time()-t0:.0f}s)', flush=True)


def sl_listing(dd, i, mask, lo=72, hi=24 * 60, thr=2e7, n_max=30, w_name=0.05):
    """Short young liquid perps, equal weight capped at w_name of the sleeve gross, long majors."""
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
    ws = min(1.0 / len(idx), w_name)            # sleeve gross shrinks when few candidates
    w[idx] = -ws
    w[jm] += ws * len(idx) / len(jm)
    return w


def sl_core(dd, i, mask):
    m = dd.universe(i, min_age_h=720, min_turn=1e7, top_n=100)
    return quantile_ls(core_signal(dd, i, m, 336), 0.2)


def make_ml(P, top=0.3, power=2.0):
    def f(dd, i, mask):
        m = dd.universe(i, min_age_h=720, min_turn=1e7, top_n=150)
        s = np.where(m & np.isfinite(P[i]), P[i], np.nan)
        if top:
            return quantile_ls(s, top)
        return rank_weights(s, m, power=power)
    return f


def combo(parts, scale):
    def fn(dd, i, mask):
        w = np.zeros(N)
        for k, fnk in parts.items():
            w += scale[k] * fnk(dd, i, mask)
        return w / sum(scale[k] for k in parts)
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


S = {'A': sl_listing, 'C': sl_core}
if P3 is not None:
    S['D3q'] = make_ml(P3, top=0.3); S['D3r'] = make_ml(P3, top=None, power=2.0)
if P7 is not None:
    S['D7q'] = make_ml(P7, top=0.3); S['D7r'] = make_ml(P7, top=None, power=2.0)
# standalone
for k in S:
    go(f'{k} standalone', S[k])
go('A standalone + stop30', S['A'], stop_pct=0.30)
# combos (equal-vol-ish scalars: A dvol ~1.2 at gross1 (w_name 5%), C ~1.2, D ~1.2)
sc = {k: 1.0 for k in S}
if 'D3q' in S:
    go('A+C+D3q', combo({k: S[k] for k in ('A', 'C', 'D3q')}, sc))
    go('A+D3q', combo({k: S[k] for k in ('A', 'D3q')}, sc))
if 'D7q' in S:
    go('A+C+D7q', combo({k: S[k] for k in ('A', 'C', 'D7q')}, sc))
    go('A+D7q', combo({k: S[k] for k in ('A', 'D7q')}, sc))
if 'D3q' in S and 'D7q' in S:
    go('A+C+D3q+D7q', combo({k: S[k] for k in ('A', 'C', 'D3q', 'D7q')}, sc))
go('A+C', combo({k: S[k] for k in ('A', 'C')}, sc))
# choose best by Sharpe among combos and apply risk overlays
best_label = max([r for r in rows if '+' in r['label']], key=lambda r: r['sharpe'])['label']
parts = best_label.split('+')
best_fn = combo({k: S[k] for k in parts}, sc)
print(f'\n>>> best combo: {best_label}')
go(f'{best_label} VT0.6 lev3', best_fn, vol_target=0.006, max_lev=3.0)
go(f'{best_label} VT0.6 lev3 stop30', best_fn, vol_target=0.006, max_lev=3.0, stop_pct=0.30)
go(f'{best_label} VT0.6 lev3 stop30 dstop2', best_fn, vol_target=0.006, max_lev=3.0, stop_pct=0.30, daily_stop=0.02)
go(f'{best_label} VT0.6 lev3 stop30 maker50', best_fn, vol_target=0.006, max_lev=3.0, stop_pct=0.30, maker_share=0.5, slip_fn=slip_model(scale=0.5))
go(f'{best_label} VT0.6 lev3 stop30 taker-only', best_fn, vol_target=0.006, max_lev=3.0, stop_pct=0.30, maker_share=0.0, slip_fn=slip_model(scale=1.0))
go(f'{best_label} VT0.6 lev3 stop30 slip x2 taker', best_fn, vol_target=0.006, max_lev=3.0, stop_pct=0.30, maker_share=0.0, slip_fn=slip_model(scale=2.0))
d.start_i = int(d.idx.searchsorted(pd.Timestamp(OOS, tz='UTC')))
m, eq, hp = run(d, best_fn, reb_h=24, band=0.3, smooth=0.5, fee_bps=5.5, maker_share=0.7, slip_fn=slip_model(scale=0.3),
                gross=1.0, pos_cap=0.02, cap_short=0.015, cap_exempt=MAJ5, cap_exempt_val=0.30,
                uni_kwargs=dict(min_age_h=0, min_turn=0, top_n=0), vol_target=0.006, max_lev=3.0, stop_pct=0.30,
                label=f'{best_label} FINAL', ret_diag=True, attrib=True, verbose=True)
eq.to_csv('out_exp9b_eq.csv'); hp.to_csv('out_exp9b_hourly_pnl.csv'); m['attrib'].to_csv('out_exp9b_attrib.csv')
pd.DataFrame(rows).to_csv('out_exp9b_portfolio.csv', index=False)
open('out_exp9b_best.txt', 'w').write(best_label)
print('DONE')
