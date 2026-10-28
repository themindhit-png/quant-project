#!/usr/bin/env python3
"""Fast screen of new cross-sectional signal ideas (Opus, 2026-10-28). Ideas fixed BEFORE looking at results.
Each idea: signal at hour h (daily 00:00, or every 8h for fast ones), universe = top-150 by 24h turnover, age >= 30d.
Paper quintile long/short (each side 1.0), entry 1 bar after the signal (lag 1h), held to next decision; cost = 10 bps per
unit traded (taker-ish). Sign of each idea is chosen on DEV (2022-01..2026-02) only; 2026-03..10 = check.
Prints IC (Spearman vs forward return), net Sharpe DEV / HOLD, corr with the F2 paper book (legs_daily.parquet).
Run: END="2026-10-26 23:00" python3 research/screen_ideas.py"""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bt import Data, PAN
from scipy.stats import spearmanr

d = Data(start='2021-01-01', end=os.environ.get('END', '2026-10-26 23:00'), min_turn_ever=5e6, load_hl=True, load_premium=True)
T, N = d.ret.shape; cols = list(d.cols); s0 = d.idx[0]; e = d.idx[-1]
ld = lambda f: pd.read_parquet(os.path.join(PAN, f + '.parquet'), columns=cols).loc[s0:e].reindex(d.idx).values.astype(np.float64)
QV, TB, NT = ld('qvol'), ld('tbq'), ld('ntrades')
R = np.where(np.isfinite(d.ret), d.ret, 0.0); CR = np.vstack([np.zeros((1, N)), np.cumsum(R, 0)])
FU = np.where(np.isfinite(d.fund), d.fund, 0.0); CFu = np.vstack([np.zeros((1, N)), np.cumsum(FU, 0)])
JB = cols.index('BTCUSDT')
def csum(X):
    X = np.where(np.isfinite(X), X, 0.0); return np.vstack([np.zeros((1, X.shape[1])), np.cumsum(X, 0)])
CQV, CTB, CNT = csum(QV), csum(TB), csum(NT)
R2 = R * R; CR2 = np.vstack([np.zeros((1, N)), np.cumsum(R2, 0)]); CR3 = np.vstack([np.zeros((1, N)), np.cumsum(R2 * R, 0)])
rb = R[:, JB:JB + 1]; CRB = np.vstack([np.zeros((1, 1)), np.cumsum(rb, 0)]); CRRB = np.vstack([np.zeros((1, N)), np.cumsum(R * rb, 0)])
CRB2 = np.vstack([np.zeros((1, 1)), np.cumsum(rb * rb, 0)])
def win(C, i, h): return C[i + 1] - C[i + 1 - h]            # sum over bars i-h+1..i
def ret(i, h): return win(CR, i, h)
def beta(i, h):
    n = h; mx, my = win(CRB, i, h)[0] / n, win(CR, i, h) / n
    cov = win(CRRB, i, h) / n - mx * my; var = win(CRB2, i, h)[0] / n - mx * mx
    return cov / var if var > 0 else np.full(N, np.nan)
def maxret(i, h): return np.nanmax(R[i + 1 - h:i + 1], 0)
def daily_max(i, days):
    x = R[i + 1 - 24 * days:i + 1].reshape(days, 24, N).sum(1); return x.max(0)

IDEAS = {   # name: (fn(i) -> signal, step hours, hypothesis text)
    'ofi24':      (lambda i: win(CTB, i, 24) / np.maximum(win(CQV, i, 24), 1) - 0.5, 24, 'taker-buy share 24h (order flow)'),
    'ofi_shock':  (lambda i: win(CTB, i, 24) / np.maximum(win(CQV, i, 24), 1) - win(CTB, i, 720) / np.maximum(win(CQV, i, 720), 1), 24, 'order flow 24h vs 30d'),
    'ofi8':       (lambda i: win(CTB, i, 8) / np.maximum(win(CQV, i, 8), 1) - 0.5, 8, 'taker-buy share 8h'),
    'tradesize':  (lambda i: np.log(np.maximum(win(CQV, i, 24), 1) / np.maximum(win(CNT, i, 24), 1)) - np.log(np.maximum(win(CQV, i, 720), 1) / np.maximum(win(CNT, i, 720), 1)), 24, 'avg trade size 24h vs 30d (whales)'),
    'volshock':   (lambda i: np.log(np.maximum(win(CQV, i, 24), 1) / np.maximum(win(CQV, i, 720) / 30, 1)), 24, 'volume 24h vs 30d avg (attention)'),
    'quietvol':   (lambda i: np.log(np.maximum(win(CQV, i, 24), 1) / np.maximum(win(CQV, i, 720) / 30, 1)) - 20 * np.abs(ret(i, 24)), 24, 'volume shock without price move'),
    'btc_catch':  (lambda i: beta(i, 720) * ret(i, 4)[JB] - ret(i, 4), 8, 'alt lags BTC move of last 4h (lead-lag)'),
    'rev4':       (lambda i: -ret(i, 4), 8, 'short-term reversal 4h'),
    'pumpfade':   (lambda i: -maxret(i, 24), 24, 'fade the largest 1h jump of 24h'),
    'max30':      (lambda i: -daily_max(i, 30), 24, 'lottery: short max daily return 30d'),
    'skew30':     (lambda i: -(win(CR3, i, 720) / 720) / np.maximum(win(CR2, i, 720) / 720, 1e-12) ** 1.5, 24, 'short positive skew 30d'),
    'ivol30':     (lambda i: -np.sqrt(np.maximum(win(CR2, i, 720) / 720 - (beta(i, 720) ** 2) * (win(CRB2, i, 720)[0] / 720), 0)), 24, 'low idiosyncratic vol'),
    'resmom30':   (lambda i: (ret(i - 24, 696) - beta(i, 720) * ret(i - 24, 696)[JB]), 24, 'residual momentum 30d skip 1d'),
    'fsqueeze':   (lambda i: -(win(CFu, i, 72)) * np.sign(ret(i, 168)), 24, 'negative funding + uptrend (squeeze)'),
    'premrev':    (lambda i: -np.nanmean(d.prem[i - 7:i + 1], 0), 8, 'perp premium 8h -> reversal'),
    'hlrange':    (lambda i: -np.log(np.nanmax(d.high[i - 23:i + 1], 0) / np.nanmin(d.low[i - 23:i + 1], 0)), 24, 'low 24h range'),
}

legs = pd.read_parquet('research/phase4_out/legs_daily.parquet').reset_index().groupby('day').pnl.sum()
i0 = int(d.idx.searchsorted(pd.Timestamp('2022-01-01', tz='UTC'))); i0 -= i0 % 24
DEV, HOLD = ('2022-01-01', '2026-02-28'), ('2026-03-01', '2026-10-26')
def sh(x): x = x.dropna(); return x.mean() / x.std(ddof=1) * np.sqrt(365) if len(x) > 30 and x.std(ddof=1) > 0 else np.nan
# pass 2 (fixed before running it): WEIGHT=invvol sizes each side by 1/vol30 (shrinks pump tails), COST_BPS=5 (live bot: maker
# share ~90 %, Bybit maker 2 / taker 5.5 bps + slippage), only ideas with DEV t > 5 and same-sign IC on HOLD in pass 1
WEIGHT, COST = os.environ.get('WEIGHT', 'equal'), float(os.environ.get('COST_BPS', '10')) / 1e4
ONLY = [x for x in os.environ.get('ONLY', '').split(',') if x]
VOL30 = None
if WEIGHT == 'invvol':
    VOL30 = lambda i: np.sqrt(np.maximum(win(CR2, i, 720) / 720, 1e-10))
SUF = os.environ.get('SUF', '')
res = []; NET = {}
for name, (fn, step, hyp) in IDEAS.items():
    if ONLY and name not in ONLY: continue
    w_prev = np.zeros(N); gross, cost, ics = {}, {}, []
    for i in range(i0, T - step - 2, step):
        m = d.universe(i, min_age_h=720, min_turn=1e7, top_n=150)
        with np.errstate(all='ignore'): s_ = np.asarray(fn(i), float)
        idx = np.flatnonzero(m & np.isfinite(s_)); w = np.zeros(N)
        fwd = (CR[i + 2 + step] - CR[i + 2]) - (CFu[i + 2 + step] - CFu[i + 2])     # enter 1 bar after the signal (lag 1h)
        if len(idx) >= 30:
            k = max(int(len(idx) * 0.2), 1); o = idx[np.argsort(s_[idx])]
            w[o[-k:]] = 1.0 / k; w[o[:k]] = -1.0 / k
            if VOL30 is not None:
                iv = 1.0 / VOL30(i); w = np.where(w != 0, iv, 0.0) * np.sign(w)
                w[w > 0] /= w[w > 0].sum(); w[w < 0] /= -w[w < 0].sum()
            ics.append((d.idx[i], spearmanr(s_[idx], fwd[idx])[0]))
        day = d.idx[i].floor('D').tz_localize(None)
        gross[day] = gross.get(day, 0.0) + float((w * fwd).sum()); cost[day] = cost.get(day, 0.0) + COST * float(np.abs(w - w_prev).sum()); w_prev = w
    g, c, ic = pd.Series(gross), pd.Series(cost), pd.Series(dict(ics))
    sg = 1.0 if ic[DEV[0]:DEV[1]].mean() >= 0 else -1.0                    # direction chosen on DEV only
    r = sg * g - c; NET[name] = r; icd, ich = sg * ic[DEV[0]:DEV[1]], sg * ic[HOLD[0]:HOLD[1]]
    lj = legs.reindex(r.index)
    res.append(dict(idea=name, sign='+' if sg > 0 else '-', step=step, IC_dev=icd.mean(), t_dev=icd.mean() / icd.std(ddof=1) * np.sqrt(len(icd)),
                    IC_hold=ich.mean(), Sh_dev=sh(r[DEV[0]:DEV[1]]), Shg_dev=sh(sg * g[DEV[0]:DEV[1]]), Shg_hold=sh(sg * g[HOLD[0]:HOLD[1]]), Sh_hold=sh(r[HOLD[0]:HOLD[1]]), ret_dev=r[DEV[0]:DEV[1]].mean() * 36500,
                    cost_yr=c[DEV[0]:DEV[1]].mean() * 36500, corrF2=r[DEV[0]:DEV[1]].corr(lj[DEV[0]:DEV[1]]), hyp=hyp))
    print(res[-1], flush=True)
pd.DataFrame(NET).to_csv(f'research/phase4_out/screen_ideas{SUF}_daily.csv')
t = pd.DataFrame(res); t.to_csv(f'research/phase4_out/screen_ideas{SUF}.csv', index=False)
print(t.drop(columns='hyp').round(2).to_string(index=False))
