#!/usr/bin/env python3
"""Experiment 9c: portfolio refinements + sleeve correlation matrix.
 - listing hedge variants: majors (ref) / equal-weight liquid alt basket (top-50) / BTC-only
 - portfolio-level BTC beta hedge (rolling 720h beta, capped)
 - ML quantile 20% vs 30%; add predicted-funding carry sleeve (maker) as 5th sleeve
 - VT 0.5% vs 0.6%; report daily-return correlations between standalone sleeves."""
import os, time, numpy as np, pandas as pd
from bt import Data, run, quantile_ls, rank_weights, slip_model
from strategies import core_signal

START, END, OOS = '2021-01-01', '2026-08-31 23:00', '2022-01-01'
d = Data(start=START, end=END, min_turn_ever=5e6, load_hl=True, load_premium=True)
T, N = d.ret.shape
MAJ5 = [s for s in ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT'] if s in d.cols]
jm = np.array([int(np.flatnonzero(d.cols == s)[0]) for s in MAJ5])
jbtc = int(np.flatnonzero(d.cols == 'BTCUSDT')[0])
P3 = np.load('out_exp6_pred_h3.npy'); P7 = np.load('out_exp6_pred_h7.npy')
rows = []; series = {}


def report(label, m, t0):
    by = m['by_year']
    rows.append(dict(label=label, sharpe=m['sharpe'], cagr=m['cagr'], dvol=m['dvol'], mdd=m['mdd'], calmar=m['calmar'],
                     worst=m['worst_day'], p1=m['p1'], turnover=m['turnover_x'], fees=m['fees'], slip=m['slip'], funding=m['funding'],
                     npos=m['avg_npos'], gross=m['avg_gross'], stops=m.get('n_stops', 0),
                     **{f'sh{y}': v[1] for y, v in by.items()}, **{f'r{y}': v[0] for y, v in by.items()}))
    print(f'{label:<50} Sh {m["sharpe"]:5.2f} CAGR {m["cagr"]*100:6.1f}% dvol {m["dvol"]*100:.2f}% MDD {m["mdd"]*100:6.1f}% '
          f'wd {m["worst_day"]*100:5.1f}% p1 {m["p1"]*100:5.2f}% turn {m["turnover_x"]:4.0f}x cost {(m["fees"]+m["slip"])*100:.1f}% fund {-m["funding"]*100:+.1f}% '
          f'npos {m["avg_npos"]:.0f} gross {m["avg_gross"]:.2f} | ' + ' '.join(f'{y}:{v[1]:.1f}' for y, v in by.items()) + f'  ({time.time()-t0:.0f}s)', flush=True)


def liquid_basket(dd, i, n=50):
    m = dd.universe(i, min_age_h=720 * 3, min_turn=3e7, top_n=n)
    m[jm] = False
    return np.flatnonzero(m)


def sl_listing(hedge='majors', lo=72, hi=24 * 60, thr=2e7, n_max=30, w_name=0.05):
    def fn(dd, i, mask):
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
        ws = min(1.0 / len(idx), w_name)
        w[idx] = -ws
        tot = ws * len(idx)
        if hedge == 'majors':
            w[jm] += tot / len(jm)
        elif hedge == 'btc':
            w[jbtc] += tot
        elif hedge == 'basket':
            b = liquid_basket(dd, i)
            b = b[~np.isin(b, idx)]
            if len(b):
                w[b] += tot / len(b)
        return w
    return fn


def sl_core(dd, i, mask):
    m = dd.universe(i, min_age_h=720, min_turn=1e7, top_n=100)
    return quantile_ls(core_signal(dd, i, m, 336), 0.2)


def make_ml(P, top=0.3):
    def f(dd, i, mask):
        m = dd.universe(i, min_age_h=720, min_turn=1e7, top_n=150)
        s = np.where(m & np.isfinite(P[i]), P[i], np.nan)
        return quantile_ls(s, top)
    return f


def sl_carry(dd, i, mask):
    m = dd.universe(i, min_age_h=720, min_turn=1e7, top_n=100)
    pav = np.nanmean(dd.prem[i - 7:i + 1].astype(np.float64), 0)
    f = pav + np.clip(0.0001 - pav, -0.0005, 0.0005)
    return quantile_ls(np.where(m & np.isfinite(f), -f, np.nan), 0.1)


def beta_hedge(fn, lb=720, cap=0.5):
    def f(dd, i, mask):
        w = fn(dd, i, mask)
        R = dd.ret[i - lb + 1:i + 1]
        rb = R[:, jbtc]; ok = np.isfinite(rb)
        Rm = np.where(np.isfinite(R[ok]), R[ok], 0.0); rbb = rb[ok]
        vb = rbb.var()
        if vb <= 0:
            return w
        beta = ((Rm - Rm.mean(0)) * (rbb - rbb.mean())[:, None]).mean(0) / vb
        beta = np.where(np.isfinite(beta), np.clip(beta, -3, 3), 1.0)
        bb = float(np.nansum(w * beta))
        w = w.copy(); w[jbtc] -= np.clip(bb, -cap, cap)
        return w
    return f


def combo(parts):
    def fn(dd, i, mask):
        w = np.zeros(N)
        for f in parts:
            w += f(dd, i, mask)
        return w / len(parts)
    return fn


def go(label, fn, keep=False, **kw):
    t0 = time.time()
    d.start_i = int(d.idx.searchsorted(pd.Timestamp(OOS, tz='UTC')))
    args = dict(reb_h=24, band=0.3, smooth=0.5, fee_bps=5.5, maker_share=0.7, slip_fn=slip_model(scale=0.3),
                gross=1.0, pos_cap=0.02, cap_short=0.015, cap_exempt=MAJ5, cap_exempt_val=0.30,
                uni_kwargs=dict(min_age_h=0, min_turn=0, top_n=0), verbose=False, label=label)
    args.update(kw)
    if keep:
        m, eq, hp = run(d, fn, ret_diag=True, **args)
        series[label] = (hp / eq.shift(1)).dropna()
    else:
        m, eq = run(d, fn, **args)
    report(label, m, t0)
    return m, eq


A_maj, A_bsk, A_btc = sl_listing('majors'), sl_listing('basket'), sl_listing('btc')
C = sl_core; D3 = make_ml(P3, 0.3); D7 = make_ml(P7, 0.3); D3q2 = make_ml(P3, 0.2); D7q2 = make_ml(P7, 0.2)
# standalone (for correlations)
go('A majors-hedge', A_maj, keep=True)
go('A basket-hedge', A_bsk, keep=True)
go('A btc-hedge', A_btc)
go('C core336', C, keep=True)
go('D3 q30', D3, keep=True)
go('D7 q30', D7, keep=True)
go('D3 q20', D3q2)
go('D7 q20', D7q2)
go('carry pred q10 (maker)', sl_carry, keep=True, reb_h=8, band=0.5, smooth=0.7)
# correlations of daily returns
if series:
    df = pd.DataFrame({k: (1 + v).groupby(v.index.floor('D')).prod() - 1 for k, v in series.items()})
    print('\ndaily return correlations:\n', df.corr().round(2).to_string(), '\n', flush=True)
# combos
go('A_maj+C+D3+D7 (ref)', combo([A_maj, C, D3, D7]))
go('A_bsk+C+D3+D7', combo([A_bsk, C, D3, D7]))
go('A_bsk+C+D3q20+D7q20', combo([A_bsk, C, D3q2, D7q2]))
go('A_bsk+C+D3+D7 + beta hedge', beta_hedge(combo([A_bsk, C, D3, D7])))
go('A_bsk+C+D3+D7+carry', combo([A_bsk, C, D3, D7, sl_carry]))
# risk overlays on basket version
base = combo([A_bsk, C, D3, D7])
go('A_bsk+C+D3+D7 VT0.6 lev3', base, vol_target=0.006, max_lev=3.0)
go('A_bsk+C+D3+D7 VT0.5 lev3', base, vol_target=0.005, max_lev=3.0)
go('A_bsk+C+D3+D7 VT0.6 lev3 stop30 dstop2', base, vol_target=0.006, max_lev=3.0, stop_pct=0.30, daily_stop=0.02)
go('A_bsk+C+D3+D7 VT0.6 lev3 capS1% stop30', base, vol_target=0.006, max_lev=3.0, stop_pct=0.30, cap_short=0.01)
go('A_bsk+C+D3+D7 VT0.6 lev3 gross<=2', base, vol_target=0.006, max_lev=3.0, max_gross_x=2.0)
d.start_i = int(d.idx.searchsorted(pd.Timestamp(OOS, tz='UTC')))
m, eq, hp = run(d, base, reb_h=24, band=0.3, smooth=0.5, fee_bps=5.5, maker_share=0.7, slip_fn=slip_model(scale=0.3),
                gross=1.0, pos_cap=0.02, cap_short=0.015, cap_exempt=MAJ5, cap_exempt_val=0.30,
                uni_kwargs=dict(min_age_h=0, min_turn=0, top_n=0), vol_target=0.006, max_lev=3.0, max_gross_x=2.0,
                label='A_bsk+C+D3+D7 VT0.6 lev3 gross<=2 FINAL', ret_diag=True, attrib=True, verbose=True)
eq.to_csv('out_exp9c_eq.csv'); hp.to_csv('out_exp9c_hourly_pnl.csv'); m['attrib'].to_csv('out_exp9c_attrib.csv')
pd.DataFrame(rows).to_csv('out_exp9c_portfolio.csv', index=False)
print('DONE')
