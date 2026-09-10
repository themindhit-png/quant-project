#!/usr/bin/env python3
"""Experiment 5: cost engineering on the two most promising cheap signals (weekly momentum,
core 336h) and a 3-signal blend: maker execution assumption, bands, smoothing, partial trading,
universe hysteresis, beta hedge. Also the same for the blend.
maker_share m: blended one-way cost = m*2bps + (1-m)*(5.5 + slip_model). We approximate by passing
fee_bps=5.5, maker_share=m (fee part) and scaling slip by (1-m)."""
import time, numpy as np, pandas as pd
from bt import Data, run, quantile_ls, rank_weights, slip_model
from strategies import mom_signal, core_signal, funding_signal, vol_h, zscore

START, END = '2021-06-01', '2026-08-31 23:00'
d = Data(start=START, end=END, min_turn_ever=5e6)
UNI = dict(min_age_h=720, min_turn=1e7, top_n=100)
UNI_H = dict(min_age_h=720, min_turn=1e7, top_n=100, exit_n=140)
jbtc = int(np.flatnonzero(d.cols == 'BTCUSDT')[0])
rows = []


def report(label, m, t0):
    by = m['by_year']
    rows.append(dict(label=label, sharpe=m['sharpe'], cagr=m['cagr'], dvol=m['dvol'], mdd=m['mdd'], calmar=m['calmar'],
                     worst=m['worst_day'], turnover=m['turnover_x'], fees=m['fees'], slip=m['slip'], funding=m['funding'],
                     npos=m['avg_npos'], **{f'sh{y}': v[1] for y, v in by.items()}, **{f'r{y}': v[0] for y, v in by.items()}))
    print(f'{label:<52} Sh {m["sharpe"]:5.2f} CAGR {m["cagr"]*100:6.1f}% dvol {m["dvol"]*100:.2f}% MDD {m["mdd"]*100:6.1f}% '
          f'wd {m["worst_day"]*100:5.1f}% turn {m["turnover_x"]:4.0f}x cost {(m["fees"]+m["slip"])*100:.1f}% fund {-m["funding"]*100:+.1f}% | '
          + ' '.join(f'{y}:{v[1]:.1f}' for y, v in by.items()) + f'  ({time.time()-t0:.0f}s)', flush=True)


def go(label, fn, maker=0.0, **kw):
    t0 = time.time()
    args = dict(reb_h=24, band=0.0, fee_bps=5.5, slip_fn=slip_model(scale=1.0 - maker), maker_share=maker,
                maker_fee_bps=2.0, gross=1.0, pos_cap=0.02, cap_short=0.015, uni_kwargs=UNI, verbose=False, label=label)
    args.update(kw)
    m, eq = run(d, fn, **args)
    report(label, m, t0)
    return m, eq


mom_w = lambda dd, i, m: quantile_ls(mom_signal(dd, i, m, 720, 24), 0.2)
core_w = lambda dd, i, m: quantile_ls(core_signal(dd, i, m, 336), 0.2)


def blend(dd, i, m, wts=(1.0, 1.0, 0.5)):
    z1 = np.nan_to_num(zscore(np.where(m, mom_signal(dd, i, m, 720, 24), np.nan)))
    z2 = np.nan_to_num(zscore(np.where(m, core_signal(dd, i, m, 336), np.nan)))
    z3 = np.nan_to_num(zscore(np.where(m, funding_signal(dd, i, m, 168), np.nan)))
    s = wts[0] * z1 + wts[1] * z2 + wts[2] * z3
    return rank_weights(np.where(m, s, np.nan), m)


def beta_hedged(fn, lb=720):
    """Wrap a weight fn: add BTC position to neutralise the book's rolling beta to BTC."""
    def f(dd, i, m):
        w = fn(dd, i, m)
        R = dd.ret[i - lb + 1:i + 1]
        rb = R[:, jbtc]
        ok = np.isfinite(rb)
        rb = rb[ok]; Rm = np.where(np.isfinite(R[ok]), R[ok], 0.0)
        vb = rb.var()
        if vb <= 0:
            return w
        beta = ((Rm - Rm.mean(0)) * (rb - rb.mean())[:, None]).mean(0) / vb
        book_beta = float(np.nansum(w * np.where(np.isfinite(beta), beta, 1.0)))
        w = w.copy(); w[jbtc] -= book_beta
        return w
    return f


# --- weekly momentum: execution & turnover variants ---
go('M0 mom720 weekly EW taker', mom_w, reb_h=168)
go('M1 mom720 weekly maker70%', mom_w, reb_h=168, maker=0.7)
go('M2 mom720 daily band30 smooth0.5 taker', mom_w, reb_h=24, band=0.3, smooth=0.5)
go('M3 mom720 daily band30 smooth0.5 maker70%', mom_w, reb_h=24, band=0.3, smooth=0.5, maker=0.7)
go('M4 mom720 weekly maker70% uni-hysteresis', mom_w, reb_h=168, maker=0.7, uni_kwargs=UNI_H)
go('M5 mom720 weekly maker70% beta-hedged', beta_hedged(mom_w), reb_h=168, maker=0.7, pos_cap=0.5, cap_short=0.5)
# --- core 336 ---
go('C0 core336 daily taker', core_w, reb_h=24)
go('C1 core336 daily band30 smooth0.5 maker70%', core_w, reb_h=24, band=0.3, smooth=0.5, maker=0.7)
go('C2 core336 reb 48h band30 maker70%', core_w, reb_h=48, band=0.3, maker=0.7)
# --- blend ---
go('B0 blend(mom,core,0.5carry) rank daily taker', blend, reb_h=24)
go('B1 blend rank daily band30 smooth0.5 maker70%', blend, reb_h=24, band=0.3, smooth=0.5, maker=0.7)
go('B2 blend rank daily band30 smooth0.7 maker70% hyst', blend, reb_h=24, band=0.3, smooth=0.7, maker=0.7, uni_kwargs=UNI_H)
go('B3 blend B2 + beta hedge', beta_hedged(blend), reb_h=24, band=0.3, smooth=0.7, maker=0.7, uni_kwargs=UNI_H, pos_cap=0.5, cap_short=0.5)
go('B4 blend B2 + VT 0.6% maxlev 2', blend, reb_h=24, band=0.3, smooth=0.7, maker=0.7, uni_kwargs=UNI_H, vol_target=0.006, max_lev=2.0)
pd.DataFrame(rows).to_csv('out_exp5_costs.csv', index=False)
print('DONE')
