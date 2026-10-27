#!/usr/bin/env python3
"""Experiment 3: structural / event-driven ideas with intrinsically low turnover.
 I1 new-listing drift: short perps aged 3..45 days (liquid), long BTC/ETH hedge (dollar neutral).
 I2 crash reversal: long names with 24h return < -X% and volume spike, hold ~48h (long-only + BTC short hedge).
 I3 extreme-funding carry: short only names with trailing funding > threshold, long low-funding majors; caps.
 I4 carry x reversal alignment: z(funding) + z(24h reversal).
 I5 time-series trend on majors (BTC/ETH/SOL/BNB/XRP): sign of 720h return, inv-vol, weekly.
Costs: fee 5.5 + liquidity slip model. Per-position cap 2%, shorts 1.5%."""
import time, numpy as np, pandas as pd
from bt import Data, run, quantile_ls, rank_weights, slip_model
from strategies import funding_signal, reversal_signal, mom_signal, vol_h, zscore

START, END = '2021-06-01', '2026-08-31 23:00'
d = Data(start=START, end=END, min_turn_ever=5e6, load_hl=True)
MAJORS = [s for s in ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT'] if s in d.cols]
jm = np.array([int(np.flatnonzero(d.cols == s)[0]) for s in MAJORS])
jbtc = int(np.flatnonzero(d.cols == 'BTCUSDT')[0])
rows = []


def report(label, m, t0):
    by = m['by_year']
    rows.append(dict(label=label, sharpe=m['sharpe'], cagr=m['cagr'], dvol=m['dvol'], mdd=m['mdd'], calmar=m['calmar'],
                     worst=m['worst_day'], turnover=m['turnover_x'], fees=m['fees'], slip=m['slip'], funding=m['funding'],
                     npos=m['avg_npos'], **{f'sh{y}': v[1] for y, v in by.items()}, **{f'r{y}': v[0] for y, v in by.items()}))
    print(f'{label:<44} Sh {m["sharpe"]:5.2f} CAGR {m["cagr"]*100:6.1f}% dvol {m["dvol"]*100:.2f}% MDD {m["mdd"]*100:6.1f}% '
          f'wd {m["worst_day"]*100:5.1f}% turn {m["turnover_x"]:4.0f}x cost {(m["fees"]+m["slip"])*100:.1f}% fund {-m["funding"]*100:+.1f}% npos {m["avg_npos"]:.0f} | '
          + ' '.join(f'{y}:{v[1]:.1f}' for y, v in by.items()) + f'  ({time.time()-t0:.0f}s)', flush=True)


def go(label, fn, **kw):
    t0 = time.time()
    args = dict(reb_h=24, band=0.0, fee_bps=5.5, slip_fn=slip_model(), gross=1.0, pos_cap=0.02, cap_short=0.015,
                verbose=False, label=label)
    args.update(kw)
    m, eq = run(d, fn, **args)
    report(label, m, t0)
    return m, eq


def hedge_with(w_alts, hedge_idx, side):
    """Put the offsetting dollar amount into hedge symbols equally (side=+1 long hedge, -1 short)."""
    w = w_alts.copy()
    tot = np.abs(w).sum()
    if tot > 0:
        w[hedge_idx] += side * tot / len(hedge_idx)
    return w


# ---------------- I1: new listing drift ----------------
def new_listing_short(dd, i, mask, lo=72, hi=24 * 45, n_max=15):
    age = dd.age[i]
    t = dd.t24[i]
    cand = (age >= lo) & (age <= hi) & dd.valid[i] & np.isfinite(t) & (t >= 2e7) & (dd.cols != 'BTCUSDT')
    idx = np.flatnonzero(cand)
    w = np.zeros(len(mask))
    if len(idx) == 0:
        return w
    idx = idx[np.argsort(-t[idx])][:n_max]
    w[idx] = -1.0 / len(idx)
    return hedge_with(w, jm, +1)


for lo, hi in ((72, 24 * 30), (72, 24 * 60), (24 * 7, 24 * 90)):
    go(f'I1 short new listings age {lo//24}-{hi//24}d, long majors', lambda dd, i, m, lo=lo, hi=hi: new_listing_short(dd, i, m, lo, hi),
       uni_kwargs=dict(min_age_h=0, min_turn=0, top_n=0), reb_h=24, stop_pct=0.30)


# ---------------- I2: crash reversal (long crashed liquid names, short BTC hedge) ----------------
def crash_long(dd, i, mask, thr=-0.15, vol_mult=2.0, n_max=10):
    r24 = dd.cff[i] / dd.cff[i - 24] - 1.0
    t = dd.t24[i]
    tavg = np.nanmean(dd.t24[i - 24 * 30:i:24], 0)
    cand = mask & (r24 <= thr) & (t >= vol_mult * tavg)
    idx = np.flatnonzero(cand)
    w = np.zeros(len(mask))
    if len(idx) == 0:
        return w
    idx = idx[np.argsort(r24[idx])][:n_max]
    w[idx] = 1.0 / len(idx)
    return hedge_with(w, np.array([jbtc]), -1)


for thr in (-0.12, -0.20, -0.30):
    go(f'I2 crash reversal thr {thr} hold 24h (daily)', lambda dd, i, m, thr=thr: crash_long(dd, i, m, thr),
       uni_kwargs=dict(min_age_h=720, min_turn=1e7, top_n=150), reb_h=24)
go('I2 crash reversal thr -0.2 reb 8h', lambda dd, i, m: crash_long(dd, i, m, -0.2), uni_kwargs=dict(min_age_h=720, min_turn=1e7, top_n=150), reb_h=8)


# ---------------- I3: extreme funding carry ----------------
def extreme_carry(dd, i, mask, lb_h=72, thr_8h=0.0005, n_max=15):
    f = dd.fund[i - lb_h + 1:i + 1]
    ev = (f != 0).sum(0)
    avg8 = np.where(ev > 0, np.nansum(f, 0) / np.maximum(ev, 1), np.nan)   # avg rate per funding event
    cand = mask & np.isfinite(avg8) & (avg8 >= thr_8h)
    idx = np.flatnonzero(cand)
    w = np.zeros(len(mask))
    if len(idx) == 0:
        return w
    idx = idx[np.argsort(-avg8[idx])][:n_max]
    w[idx] = -1.0 / len(idx)
    # long side: lowest-funding liquid names (carry both sides)
    lowc = mask & np.isfinite(avg8) & ~cand
    li = np.flatnonzero(lowc)
    if len(li):
        li = li[np.argsort(avg8[li])][:n_max]
        w[li] = 1.0 / len(li)
    return w


for thr in (0.0003, 0.0005, 0.001):
    go(f'I3 extreme carry avg8h>={thr*100:.2f}% (72h) both sides', lambda dd, i, m, thr=thr: extreme_carry(dd, i, m, 72, thr),
       uni_kwargs=dict(min_age_h=720, min_turn=1e7, top_n=150), reb_h=8, stop_pct=0.30)
go('I3 extreme carry 0.05% stop 20%', lambda dd, i, m: extreme_carry(dd, i, m, 72, 0.0005),
   uni_kwargs=dict(min_age_h=720, min_turn=1e7, top_n=150), reb_h=8, stop_pct=0.20)


# ---------------- I4: carry x reversal ----------------
def carry_rev(dd, i, mask, wf=0.5, wr=0.5):
    zf = zscore(np.where(mask, funding_signal(dd, i, mask, 168), np.nan))
    zr = zscore(np.where(mask, reversal_signal(dd, i, mask, 72), np.nan))
    s = wf * np.nan_to_num(zf) + wr * np.nan_to_num(zr)
    return quantile_ls(np.where(mask, s, np.nan), 0.2)


go('I4 carry168 x reversal72 quantile20', carry_rev, uni_kwargs=dict(min_age_h=720, min_turn=1e7, top_n=100), reb_h=24)
go('I4 carry168 x reversal72 rank', lambda dd, i, m: rank_weights(0.5 * np.nan_to_num(zscore(np.where(m, funding_signal(dd, i, m, 168), np.nan))) + 0.5 * np.nan_to_num(zscore(np.where(m, reversal_signal(dd, i, m, 72), np.nan))), m),
   uni_kwargs=dict(min_age_h=720, min_turn=1e7, top_n=100), reb_h=24)


# ---------------- I5: TS trend on majors ----------------
def ts_trend(dd, i, mask, lb=720, skip=0):
    w = np.zeros(len(mask))
    for j in jm:
        a, b = dd.cff[i - skip, j], dd.cff[i - lb, j]
        if np.isfinite(a) and np.isfinite(b) and b > 0:
            v = np.nanstd(dd.ret[i - 720:i + 1, j])
            w[j] = np.sign(a / b - 1) / max(v, 1e-6)
    s = np.abs(w).sum()
    return w / s if s > 0 else w


for lb in (336, 720, 1440):
    go(f'I5 TS trend majors {lb//24}d weekly', lambda dd, i, m, lb=lb: ts_trend(dd, i, m, lb), uni_kwargs=dict(min_age_h=0, min_turn=0, top_n=0), reb_h=168, pos_cap=0.5, cap_short=0.5)
pd.DataFrame(rows).to_csv('out_exp3_ideas.csv', index=False)
print('DONE')
