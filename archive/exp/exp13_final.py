#!/usr/bin/env python3
"""Experiment 13: final candidate portfolios (OOS 2022-01 -> 2026-08) with short-only stops,
BTC beta hedge option, and execution sensitivity; saves the chosen equity curve for the MC."""
import time, json, numpy as np, pandas as pd
from bt import Data, run, quantile_ls, slip_model
from strategies import core_signal

START, END, OOS = '2021-01-01', '2026-08-31 23:00', '2022-01-01'
d = Data(start=START, end=END, min_turn_ever=5e6, load_hl=True, load_premium=True)
T, N = d.ret.shape
MAJ5 = [s for s in ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT'] if s in d.cols]
jm = np.array([int(np.flatnonzero(d.cols == s)[0]) for s in MAJ5])
jbtc = int(np.flatnonzero(d.cols == 'BTCUSDT')[0])
P3 = np.load('out_exp6_pred_h3.npy'); P7 = np.load('out_exp6_pred_h7.npy')
instr = json.load(open('instruments.json'))
launch_h = np.full(N, np.nan)
for j, s in enumerate(d.cols):
    m = instr.get(str(s))
    if m and m.get('launch'):
        launch_h[j] = m['launch'] / 1000 / 3600
rows = []


def report(label, m, t0):
    by = m['by_year']
    rows.append(dict(label=label, sharpe=m['sharpe'], cagr=m['cagr'], dvol=m['dvol'], mdd=m['mdd'], calmar=m['calmar'],
                     worst=m['worst_day'], p1=m['p1'], turnover=m['turnover_x'], fees=m['fees'], slip=m['slip'], funding=m['funding'],
                     npos=m['avg_npos'], gross=m['avg_gross'], stops=m.get('n_stops', 0),
                     **{f'sh{y}': v[1] for y, v in by.items()}, **{f'r{y}': v[0] for y, v in by.items()}))
    print(f'{label:<54} Sh {m["sharpe"]:5.2f} CAGR {m["cagr"]*100:6.1f}% dvol {m["dvol"]*100:.2f}% MDD {m["mdd"]*100:6.1f}% '
          f'wd {m["worst_day"]*100:5.1f}% p1 {m["p1"]*100:5.2f}% turn {m["turnover_x"]:4.0f}x cost {(m["fees"]+m["slip"])*100:.1f}% '
          f'npos {m["avg_npos"]:.0f} gross {m["avg_gross"]:.2f} stops {m.get("n_stops",0)} | ' + ' '.join(f'{y}:{v[1]:.1f}' for y, v in by.items()) + f'  ({time.time()-t0:.0f}s)', flush=True)


def sl_listing(lo=24 * 7, hi=24 * 90, thr=2e7, n_max=30, w_name=0.05, two_clock=True):
    def fn(dd, i, mask):
        age = dd.age[i].astype(float)
        if two_clock:
            ab = np.where(np.isfinite(launch_h), dd.hh[i] - launch_h, -np.inf)
            age = np.maximum(age, ab)
        t = dd.t24[i]
        cand = (age >= lo) & (age <= hi) & dd.valid[i] & np.isfinite(t) & (t >= thr) & (dd.age[i] >= 72)
        cand[jm] = False
        r24 = dd.cff[i] / dd.cff[i - 24] - 1
        cand &= ~(r24 > 0.30)
        idx = np.flatnonzero(cand)
        w = np.zeros(N)
        if len(idx) == 0:
            return w
        idx = idx[np.argsort(-t[idx])][:n_max]
        ws = min(1.0 / len(idx), w_name)
        w[idx] = -ws; w[jm] += ws * len(idx) / len(jm)
        return w
    return fn


def sl_core(dd, i, mask):
    m = dd.universe(i, min_age_h=720, min_turn=1e7, top_n=100)
    return quantile_ls(core_signal(dd, i, m, 336), 0.2)


def make_ml(P, top=0.3, top_n=150):
    def f(dd, i, mask):
        m = dd.universe(i, min_age_h=720, min_turn=1e7, top_n=top_n)
        s = np.where(m & np.isfinite(P[i]), P[i], np.nan)
        return quantile_ls(s, top)
    return f


def beta_hedge(fn, lb=720, cap=0.5):
    def f(dd, i, mask):
        w = fn(dd, i, mask)
        R = dd.ret[i - lb + 1:i + 1]; rb = R[:, jbtc]; ok = np.isfinite(rb)
        Rm = np.where(np.isfinite(R[ok]), R[ok], 0.0); rbb = rb[ok]; vb = rbb.var()
        if vb <= 0:
            return w
        beta = ((Rm - Rm.mean(0)) * (rbb - rbb.mean())[:, None]).mean(0) / vb
        beta = np.where(np.isfinite(beta), np.clip(beta, -3, 3), 1.0)
        w = w.copy(); w[jbtc] -= np.clip(float(np.nansum(w * beta)), -cap, cap)
        return w
    return f


def combo(parts, scales=None):
    scales = scales or [1.0] * len(parts)
    def fn(dd, i, mask):
        w = np.zeros(N)
        for f, s in zip(parts, scales):
            w += s * f(dd, i, mask)
        return w / sum(scales)
    return fn


def go(label, fn, keep=False, **kw):
    t0 = time.time()
    d.start_i = int(d.idx.searchsorted(pd.Timestamp(OOS, tz='UTC')))
    args = dict(reb_h=24, band=0.3, smooth=0.5, fee_bps=5.5, maker_share=0.7, slip_fn=slip_model(scale=0.3),
                gross=1.0, pos_cap=0.02, cap_short=0.015, cap_exempt=MAJ5, cap_exempt_val=0.30,
                uni_kwargs=dict(min_age_h=0, min_turn=0, top_n=0), verbose=False, label=label)
    args.update(kw)
    if keep:
        m, eq, hp = run(d, fn, ret_diag=True, attrib=True, **args)
        eq.to_csv(f'out_exp13_eq_{keep}.csv'); hp.to_csv(f'out_exp13_hp_{keep}.csv')
        print('   worst days: ' + ' | '.join(f'{dte} {p:+.1f}% ({top[0][0]} {top[0][1]:+.1f})' for dte, p, top in m['bad_days'][:5]))
    else:
        m, eq = run(d, fn, **args)
    report(label, m, t0)
    return m


A = sl_listing(); A0 = sl_listing(lo=72, hi=24 * 60, two_clock=False); C = sl_core; D3 = make_ml(P3); D7 = make_ml(P7)
base = combo([A, C, D3, D7]); base0 = combo([A0, C, D3, D7])
go('ref(A0 binance-clock 3-60d)+C+D3+D7', base0)
go('A(two-clock 7-90d)+C+D3+D7', base)
go('A+C+D3+D7 + BTC beta hedge', beta_hedge(base))
R = dict(vol_target=0.006, max_lev=3.0, max_gross_x=2.0)
go('VT0.6 lev3 g<=2', base, **R)
go('VT0.6 lev3 g<=2 stopS30', base, stop_pct=0.30, stop_side='short', **R)
go('VT0.6 lev3 g<=2 stopS40', base, stop_pct=0.40, stop_side='short', **R)
go('VT0.6 lev3 g<=2 stopS30 + beta hedge', beta_hedge(base), stop_pct=0.30, stop_side='short', **R)
go('VT0.6 lev3 g<=2 stopS30 dstop2.5', base, stop_pct=0.30, stop_side='short', daily_stop=0.025, **R)
go('VT0.6 lev3 g<=2 stopS30 smooth0.7 band0.4', base, stop_pct=0.30, stop_side='short', smooth=0.7, band=0.4, **R)
go('VT0.55 lev3 g<=2 stopS30', base, stop_pct=0.30, stop_side='short', vol_target=0.0055, max_lev=3.0, max_gross_x=2.0)
# execution sensitivity for the main candidate
main = dict(stop_pct=0.30, stop_side='short', **R)
go('MAIN maker70 slip0.3', base, keep='main', **main)
go('MAIN maker50 slip0.5', base, maker_share=0.5, slip_fn=slip_model(scale=0.5), **main)
go('MAIN taker slip1.0', base, maker_share=0.0, slip_fn=slip_model(scale=1.0), **main)
go('MAIN taker slip2.0', base, maker_share=0.0, slip_fn=slip_model(scale=2.0), **main)
go('MAIN + beta hedge', beta_hedge(base), keep='hedged', **main)
pd.DataFrame(rows).to_csv('out_exp13_final.csv', index=False)
print('DONE')
