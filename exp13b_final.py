#!/usr/bin/env python3
"""Experiment 13b: production configuration candidates: wide insurance stops (50/60%), BTC beta hedge,
and the FINAL config -> out_final_eq.csv for the challenge MC. OOS 2022-01 -> 2026-08."""
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
    print(f'{label:<50} Sh {m["sharpe"]:5.2f} CAGR {m["cagr"]*100:6.1f}% dvol {m["dvol"]*100:.2f}% MDD {m["mdd"]*100:6.1f}% '
          f'wd {m["worst_day"]*100:5.1f}% p1 {m["p1"]*100:5.2f}% turn {m["turnover_x"]:4.0f}x cost {(m["fees"]+m["slip"])*100:.1f}% '
          f'npos {m["avg_npos"]:.0f} gross {m["avg_gross"]:.2f} stops {m.get("n_stops",0)} | ' + ' '.join(f'{y}:{v[1]:.1f}' for y, v in by.items()) + f'  ({time.time()-t0:.0f}s)', flush=True)


def sl_listing(dd, i, mask, lo=24 * 7, hi=24 * 90, thr=2e7, n_max=30, w_name=0.05):
    age = dd.age[i].astype(float)
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


def combo(parts):
    def fn(dd, i, mask):
        w = np.zeros(N)
        for f in parts:
            w += f(dd, i, mask)
        return w / len(parts)
    return fn


def go(label, fn, keep=None, **kw):
    t0 = time.time()
    d.start_i = int(d.idx.searchsorted(pd.Timestamp(OOS, tz='UTC')))
    args = dict(reb_h=24, band=0.3, smooth=0.5, fee_bps=5.5, maker_share=0.7, slip_fn=slip_model(scale=0.3),
                gross=1.0, pos_cap=0.02, cap_short=0.015, cap_exempt=MAJ5, cap_exempt_val=0.30,
                uni_kwargs=dict(min_age_h=0, min_turn=0, top_n=0), verbose=False, label=label,
                vol_target=0.006, max_lev=3.0, max_gross_x=2.0)
    args.update(kw)
    if keep:
        m, eq, hp = run(d, fn, ret_diag=True, attrib=True, **args)
        eq.to_csv(f'out_{keep}_eq.csv'); hp.to_csv(f'out_{keep}_hp.csv'); m['attrib'].to_csv(f'out_{keep}_attrib.csv')
        print('   worst days: ' + ' | '.join(f'{dte} {p:+.1f}% ({top[0][0]} {top[0][1]:+.1f})' for dte, p, top in m['bad_days'][:6]))
    else:
        m, eq = run(d, fn, **args)
    report(label, m, t0)
    return m


base = combo([sl_listing, sl_core, make_ml(P3), make_ml(P7)])
go('P0 VT0.6 lev3 g<=2 (no stops)', base, keep='final')
go('P1 + stopS50', base, stop_pct=0.50, stop_side='short')
go('P2 + stopS60', base, stop_pct=0.60, stop_side='short')
go('P3 + stopS100', base, stop_pct=1.00, stop_side='short')
go('P4 + beta hedge', beta_hedge(base), keep='final_hedged')
go('P5 + beta hedge + stopS50', beta_hedge(base), stop_pct=0.50, stop_side='short')
go('P6 maker50 (no stops)', base, maker_share=0.5, slip_fn=slip_model(scale=0.5))
go('P7 taker only (no stops)', base, maker_share=0.0, slip_fn=slip_model(scale=1.0))
go('P8 VT0.6 lev2.5', base, max_lev=2.5)
go('P9 band0.2 smooth0.3', base, band=0.2, smooth=0.3)
# robustness: drop years / cost shocks
go('R1 fee+slip x1.5', base, fee_bps=8.25, slip_fn=slip_model(scale=0.45))
go('R2 exclude majors hedge (BTC only)', combo([lambda dd, i, m: (lambda w: (np.where(w < 0, w, 0.0) + np.where(dd.cols == 'BTCUSDT', -np.where(w < 0, w, 0.0).sum(), 0.0)))(sl_listing(dd, i, m)), sl_core, make_ml(P3), make_ml(P7)]))
pd.DataFrame(rows).to_csv('out_exp13b_final.csv', index=False)
print('DONE')
