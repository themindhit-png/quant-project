#!/usr/bin/env python3
"""Experiment 4: premium-index (basis) based signals.
 P1 predicted-funding carry: mean premium over last 8h (+ last realised funding) -> short high / long low.
 P2 basis spike mean reversion: z-score of current premium vs its 7d distribution; short spikes, long dips.
 P3 combined carry: realised funding 168h + predicted funding (premium 8h), both z-scored.
Setup: liquid universe (>= $10M, top 100), fee 5.5 + liquidity slip, caps 2%/1.5%, rebalance 8h at funding times."""
import time, numpy as np, pandas as pd
from bt import Data, run, quantile_ls, rank_weights, slip_model
from strategies import funding_signal, zscore, mom_signal, vol_h

START, END = '2021-06-01', '2026-08-31 23:00'
d = Data(start=START, end=END, min_turn_ever=5e6, load_premium=True)
UNI = dict(min_age_h=720, min_turn=1e7, top_n=100)
rows = []


def report(label, m, t0):
    by = m['by_year']
    rows.append(dict(label=label, sharpe=m['sharpe'], cagr=m['cagr'], dvol=m['dvol'], mdd=m['mdd'], calmar=m['calmar'],
                     worst=m['worst_day'], turnover=m['turnover_x'], fees=m['fees'], slip=m['slip'], funding=m['funding'],
                     npos=m['avg_npos'], **{f'sh{y}': v[1] for y, v in by.items()}, **{f'r{y}': v[0] for y, v in by.items()}))
    print(f'{label:<46} Sh {m["sharpe"]:5.2f} CAGR {m["cagr"]*100:6.1f}% dvol {m["dvol"]*100:.2f}% MDD {m["mdd"]*100:6.1f}% '
          f'wd {m["worst_day"]*100:5.1f}% turn {m["turnover_x"]:4.0f}x cost {(m["fees"]+m["slip"])*100:.1f}% fund {-m["funding"]*100:+.1f}% | '
          + ' '.join(f'{y}:{v[1]:.1f}' for y, v in by.items()) + f'  ({time.time()-t0:.0f}s)', flush=True)


def go(label, fn, **kw):
    t0 = time.time()
    args = dict(reb_h=8, band=0.0, fee_bps=5.5, slip_fn=slip_model(), gross=1.0, pos_cap=0.02, cap_short=0.015,
                uni_kwargs=UNI, verbose=False, label=label)
    args.update(kw)
    m, eq = run(d, fn, **args)
    report(label, m, t0)
    return m


def prem_mean(dd, i, h):
    p = dd.prem[i - h + 1:i + 1].astype(np.float64)
    return np.nanmean(p, 0)


def pred_funding(dd, i, mask, h=8):
    """Binance-style predicted funding: P_avg + clamp(0.0001 - P_avg, -0.0005, 0.0005)."""
    pav = prem_mean(dd, i, h)
    f = pav + np.clip(0.0001 - pav, -0.0005, 0.0005)
    return np.where(mask & np.isfinite(f), -f, np.nan)      # high predicted funding -> short


go('P1 predicted funding (prem 8h) q20', lambda dd, i, m: quantile_ls(pred_funding(dd, i, m), 0.2))
go('P1 predicted funding (prem 8h) q10', lambda dd, i, m: quantile_ls(pred_funding(dd, i, m), 0.1))
go('P1 predicted funding (prem 24h) q20', lambda dd, i, m: quantile_ls(pred_funding(dd, i, m, 24), 0.2))
go('P1 predicted funding (prem 8h) rank', lambda dd, i, m: rank_weights(pred_funding(dd, i, m), m))


def basis_z(dd, i, mask, h=8, win=24 * 7):
    cur = prem_mean(dd, i, h)
    hist = dd.prem[i - win + 1:i + 1].astype(np.float64)
    mu, sd = np.nanmean(hist, 0), np.nanstd(hist, 0)
    z = (cur - mu) / np.where(sd > 0, sd, np.nan)
    return np.where(mask & np.isfinite(z), -z, np.nan)


go('P2 basis z (8h vs 7d) q20', lambda dd, i, m: quantile_ls(basis_z(dd, i, m), 0.2))
go('P2 basis z (8h vs 7d) q10', lambda dd, i, m: quantile_ls(basis_z(dd, i, m), 0.1))
go('P2 basis z (1h vs 7d) q10 reb1h', lambda dd, i, m: quantile_ls(basis_z(dd, i, m, 1), 0.1), reb_h=1)
go('P2 basis z (8h vs 30d) q20', lambda dd, i, m: quantile_ls(basis_z(dd, i, m, 8, 24 * 30), 0.2))


def combo_carry(dd, i, mask):
    z1 = zscore(np.where(mask, funding_signal(dd, i, mask, 168), np.nan))
    z2 = zscore(np.where(mask, pred_funding(dd, i, mask), np.nan))
    s = np.nan_to_num(z1) + np.nan_to_num(z2)
    return quantile_ls(np.where(mask, s, np.nan), 0.2)


go('P3 realised168 + predicted carry q20', combo_carry)
go('P3 realised168 + predicted carry q20 reb24', combo_carry, reb_h=24)
# momentum filtered by carry: long winners only if not paying heavy funding
def mom_carry(dd, i, mask):
    zm = zscore(np.where(mask, mom_signal(dd, i, mask, 720, 24), np.nan))
    zc = zscore(np.where(mask, funding_signal(dd, i, mask, 168), np.nan))
    s = np.nan_to_num(zm) + 0.5 * np.nan_to_num(zc)
    return quantile_ls(np.where(mask, s, np.nan), 0.2, inv_vol=vol_h(dd, i, 720))


go('P4 mom720 + 0.5 carry168 invvol weekly', mom_carry, reb_h=168)
pd.DataFrame(rows).to_csv('out_exp4_premium.csv', index=False)
print('DONE')
