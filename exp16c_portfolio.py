#!/usr/bin/env python3
"""Experiment 16c: portfolio impact of the ML v3 prediction panels (saved by exp16) vs v2, incl. Bybit-feasible
feature set (suffix _bybit) when available. Same production structure as exp13b P0 (VT0.6, lev3, gross<=2)."""
import os, time, json, numpy as np, pandas as pd
from bt import Data, run, quantile_ls, slip_model
from strategies import core_signal

START, END, OOS = '2021-01-01', '2026-08-31 23:00', '2022-01-01'
d = Data(start=START, end=END, min_turn_ever=5e6, load_hl=True, load_premium=True)
T, N = d.ret.shape
instr = json.load(open('instruments.json')); launch_h = np.full(N, np.nan)
for j, s in enumerate(d.cols):
    mm = instr.get(str(s))
    if mm and mm.get('launch'):
        launch_h[j] = mm['launch'] / 1000 / 3600
MAJ5 = [s for s in ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT'] if s in d.cols]; jm = np.array([int(np.flatnonzero(d.cols == s)[0]) for s in MAJ5])


def sl_listing(dd, i, mask):
    age = np.maximum(dd.age[i].astype(float), np.where(np.isfinite(launch_h), dd.hh[i] - launch_h, -np.inf)); t = dd.t24[i]
    cand = (age >= 24 * 7) & (age <= 24 * 90) & dd.valid[i] & np.isfinite(t) & (t >= 2e7) & (dd.age[i] >= 72); cand[jm] = False
    r24 = dd.cff[i] / dd.cff[i - 24] - 1; cand &= ~(r24 > 0.30)
    idx = np.flatnonzero(cand); w = np.zeros(N)
    if len(idx) == 0:
        return w
    idx = idx[np.argsort(-t[idx])][:30]; ws = min(1.0 / len(idx), 0.05); w[idx] = -ws; w[jm] += ws * len(idx) / len(jm)
    return w


def sl_core(dd, i, mask):
    m = dd.universe(i, min_age_h=720, min_turn=1e7, top_n=100); return quantile_ls(core_signal(dd, i, m, 336), 0.2)


def make_ml(P, top=0.3):
    def f(dd, i, mask):
        m = dd.universe(i, min_age_h=720, min_turn=1e7, top_n=150)
        return quantile_ls(np.where(m & np.isfinite(P[i]), P[i], np.nan), top)
    return f


def combo(parts, scales=None):
    scales = scales or [1.0] * len(parts)
    def fn(dd, i, mask):
        w = np.zeros(N)
        for f, s in zip(parts, scales):
            w += s * f(dd, i, mask)
        return w / sum(scales)
    return fn


rows = []
def go(label, fn, keep=None, **kw):
    t1 = time.time(); d.start_i = int(d.idx.searchsorted(pd.Timestamp(OOS, tz='UTC')))
    args = dict(reb_h=24, band=0.3, smooth=0.5, fee_bps=5.5, maker_share=0.7, slip_fn=slip_model(scale=0.3), gross=1.0, pos_cap=0.02, cap_short=0.015,
                cap_exempt=MAJ5, cap_exempt_val=0.30, uni_kwargs=dict(min_age_h=0, min_turn=0, top_n=0), verbose=False, label=label,
                vol_target=0.006, max_lev=3.0, max_gross_x=2.0)
    args.update(kw)
    if keep:
        m, eq, hp = run(d, fn, ret_diag=True, **args); eq.to_csv(f'out_{keep}_eq.csv')
    else:
        m, eq = run(d, fn, **args)
    by = m['by_year']
    rows.append(dict(label=label, sharpe=m['sharpe'], cagr=m['cagr'], dvol=m['dvol'], mdd=m['mdd'], worst=m['worst_day'], turnover=m['turnover_x'],
                     **{f'sh{y}': v[1] for y, v in by.items()}))
    print(f'{label:<50} Sh {m["sharpe"]:5.2f} CAGR {m["cagr"]*100:6.1f}% dvol {m["dvol"]*100:.2f}% MDD {m["mdd"]*100:6.1f}% wd {m["worst_day"]*100:5.1f}% turn {m["turnover_x"]:4.0f}x cost {(m["fees"]+m["slip"])*100:.1f}% | '
          + ' '.join(f'{y}:{v[1]:.1f}' for y, v in by.items()) + f' ({time.time()-t1:.0f}s)', flush=True)
    return m


P = {k: np.load(f'out_exp16_pred_{k}.npy') for k in ('h3_r', 'h7_r', 'h3_k', 'h7_k')}
P3v2, P7v2 = np.load('out_exp6_pred_h3.npy'), np.load('out_exp6_pred_h7.npy')
go('v2 (ref P0): A+C+D3+D7', combo([sl_listing, sl_core, make_ml(P3v2), make_ml(P7v2)]))
go('v3r: A+C+D3r+D7r', combo([sl_listing, sl_core, make_ml(P['h3_r']), make_ml(P['h7_r'])]))
go('v3k: A+C+D3k+D7k', combo([sl_listing, sl_core, make_ml(P['h3_k']), make_ml(P['h7_k'])]), keep='final_v3k')
go('v3k: A+C+D3k+D7k, ML q20', combo([sl_listing, sl_core, make_ml(P['h3_k'], 0.2), make_ml(P['h7_k'], 0.2)]))
go('v3k ML x1.5 weight', combo([sl_listing, sl_core, make_ml(P['h3_k']), make_ml(P['h7_k'])], [1, 1, 1.5, 1.5]))
go('v3k ML x2 weight', combo([sl_listing, sl_core, make_ml(P['h3_k']), make_ml(P['h7_k'])], [1, 1, 2, 2]))
go('v3k ML only (D3k+D7k)', combo([make_ml(P['h3_k']), make_ml(P['h7_k'])]))
go('v3 r+k: A+C+4 ML', combo([sl_listing, sl_core, make_ml(P['h3_r']), make_ml(P['h7_r']), make_ml(P['h3_k']), make_ml(P['h7_k'])]))
go('v2 ML only (D3+D7)', combo([make_ml(P3v2), make_ml(P7v2)]))
for suf in ('_bybit',):
    f3, f7 = f'out_exp16_pred_h3_k{suf}.npy', f'out_exp16_pred_h7_k{suf}.npy'
    if os.path.exists(f3) and os.path.exists(f7):
        Pb3, Pb7 = np.load(f3), np.load(f7)
        go(f'v3k{suf} (no order-flow): A+C+D3k+D7k', combo([sl_listing, sl_core, make_ml(Pb3), make_ml(Pb7)]), keep=f'final_v3k{suf}')
        go(f'v3k{suf} ML x1.5', combo([sl_listing, sl_core, make_ml(Pb3), make_ml(Pb7)], [1, 1, 1.5, 1.5]))
pd.DataFrame(rows).to_csv('out_exp16c_portfolio.csv', index=False)
print('DONE')
