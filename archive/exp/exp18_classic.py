#!/usr/bin/env python3
"""Experiment 18: 'trader-style' systematic strategies on liquid majors, hourly bars, realistic costs
(taker 5.5 bps + 1 bp slippage per side), stops/take-profits/trailing, risk 1% of equity per trade via stop
distance, max 1 position per symbol. Strategies:
  DON   Donchian breakout N hours, exit on opposite N/2 channel or ATR trailing stop
  MAX   MA crossover fast/slow with ATR stop
  RSI   RSI(14h) mean reversion: long <30 / short >70, TP 2*ATR, SL 1.5*ATR, max hold 48h
  VBO   volatility breakout: |1h return| > 2 sigma -> follow, hold 24h, stop 1.5 sigma
  ORB   opening range breakout: first 4h of the UTC day range, breakout follow with stop at range, exit at day end
  SESS  session seasonality: long 13:00-21:00 UTC (US), short 21:00-05:00 (and reverse)
Reports per-symbol and equal-weight basket Sharpe/CAGR/MDD 2021-06 -> 2026-08."""
import time, numpy as np, pandas as pd
from bt import Data

START, END = '2021-06-01', '2026-08-31 23:00'
d = Data(start=START, end=END, min_turn_ever=5e6, load_hl=True)
T, N = d.ret.shape
i0 = d.start_i
# top-10 majors by average turnover over the sample
avg_t = np.nanmean(np.where(np.isfinite(d.t24[i0:]), d.t24[i0:], np.nan), 0)
order = np.argsort(-np.nan_to_num(avg_t))
MAJ = [str(d.cols[j]) for j in order[:10]]
print('universe:', MAJ, flush=True)
FEE = 5.5e-4 + 1e-4       # per side (taker + slippage)


def atr(h, l, c, n=24):
    tr = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
    out = np.full(len(c), np.nan); a = pd.Series(tr).rolling(n).mean().values; out[1:] = a
    return out


def simulate(sym, strategy, **kw):
    j = int(np.flatnonzero(d.cols == sym)[0])
    c = d.cff[i0 - 720:, j].astype(float); h = d.high[i0 - 720:, j].astype(float); l = d.low[i0 - 720:, j].astype(float)
    idx = d.idx[i0 - 720:]
    ok = np.isfinite(c) & np.isfinite(h) & np.isfinite(l)
    if ok.mean() < 0.95:
        return None
    h = np.where(np.isfinite(h), h, c); l = np.where(np.isfinite(l), l, c)
    n = len(c); A = atr(h, l, c, 24)
    hr = np.array([t.hour for t in idx])
    eq = 1.0; pos = 0; entry = 0.0; stop = 0.0; tp = 0.0; qty = 0.0; hold = 0; trail = 0.0
    eqs = np.full(n, np.nan); trades = 0
    r1 = np.zeros(n); r1[1:] = c[1:] / c[:-1] - 1
    sig = pd.Series(r1).rolling(168).std().values
    if strategy == 'DON':
        Nn = kw.get('n', 72); hi_ch = pd.Series(h).rolling(Nn).max().shift(1).values; lo_ch = pd.Series(l).rolling(Nn).min().shift(1).values
        hi_x = pd.Series(h).rolling(Nn // 2).max().shift(1).values; lo_x = pd.Series(l).rolling(Nn // 2).min().shift(1).values
    if strategy == 'MAX':
        f, s = kw.get('f', 24), kw.get('s', 168); ma_f = pd.Series(c).rolling(f).mean().values; ma_s = pd.Series(c).rolling(s).mean().values
    if strategy == 'RSI':
        delta = np.diff(c, prepend=c[0]); up = pd.Series(np.clip(delta, 0, None)).rolling(14).mean().values; dn = pd.Series(np.clip(-delta, 0, None)).rolling(14).mean().values
        rsi = 100 - 100 / (1 + up / np.where(dn > 0, dn, 1e-12))
    if strategy == 'ORB':
        day = np.array([t.date() for t in idx]); orb_hi = np.full(n, np.nan); orb_lo = np.full(n, np.nan)
        cur = None; hi_ = -np.inf; lo_ = np.inf
        for k in range(n):
            if day[k] != cur:
                cur = day[k]; hi_ = -np.inf; lo_ = np.inf
            if hr[k] < 4:
                hi_ = max(hi_, h[k]); lo_ = min(lo_, l[k])
            elif hr[k] >= 4:
                orb_hi[k] = hi_; orb_lo[k] = lo_
    for k in range(721, n - 1):
        px = c[k]
        # ---- manage open position on bar k (stops checked on high/low of bar k)
        if pos != 0:
            hold += 1
            exit_px = None
            if pos > 0:
                trail = max(trail, h[k] - kw.get('trail_atr', 3.0) * A[k]) if strategy in ('DON', 'MAX') else trail
                s_ = max(stop, trail) if strategy in ('DON', 'MAX') else stop
                if l[k] <= s_: exit_px = min(s_, c[k]) if s_ < c[k - 1] else c[k]
                elif tp and h[k] >= tp: exit_px = tp
            else:
                trail = min(trail, l[k] + kw.get('trail_atr', 3.0) * A[k]) if strategy in ('DON', 'MAX') else trail
                s_ = min(stop, trail) if strategy in ('DON', 'MAX') else stop
                if h[k] >= s_: exit_px = max(s_, c[k]) if s_ > c[k - 1] else c[k]
                elif tp and l[k] <= tp: exit_px = tp
            if exit_px is None:
                if strategy == 'DON' and ((pos > 0 and l[k] <= lo_x[k]) or (pos < 0 and h[k] >= hi_x[k])): exit_px = px
                if strategy == 'MAX' and ((pos > 0 and ma_f[k] < ma_s[k]) or (pos < 0 and ma_f[k] > ma_s[k])): exit_px = px
                if strategy in ('RSI', 'VBO') and hold >= kw.get('max_hold', 48): exit_px = px
                if strategy == 'ORB' and hr[k] == 23: exit_px = px
                if strategy == 'SESS' and ((pos > 0 and hr[k] == 21) or (pos < 0 and hr[k] == 5)): exit_px = px
            if exit_px is not None:
                eq += qty * (exit_px - entry) * (1 if pos > 0 else -1) - abs(qty) * exit_px * FEE
                pos = 0; qty = 0.0; trades += 1
        # ---- entries at close of bar k
        if pos == 0 and np.isfinite(A[k]) and A[k] > 0 and np.isfinite(sig[k]):
            side = 0; sl_dist = None; tp_px = None
            if strategy == 'DON':
                if c[k] > hi_ch[k]: side = 1
                elif c[k] < lo_ch[k]: side = -1
                sl_dist = kw.get('sl_atr', 3.0) * A[k]
            elif strategy == 'MAX':
                if ma_f[k] > ma_s[k] and ma_f[k - 1] <= ma_s[k - 1]: side = 1
                elif ma_f[k] < ma_s[k] and ma_f[k - 1] >= ma_s[k - 1]: side = -1
                sl_dist = kw.get('sl_atr', 3.0) * A[k]
            elif strategy == 'RSI':
                if rsi[k] < 30: side = 1
                elif rsi[k] > 70: side = -1
                sl_dist = 1.5 * A[k]; tp_d = 2.0 * A[k]
            elif strategy == 'VBO':
                if r1[k] > 2 * sig[k]: side = 1
                elif r1[k] < -2 * sig[k]: side = -1
                sl_dist = 1.5 * sig[k] * px
            elif strategy == 'ORB':
                if hr[k] >= 4 and hr[k] < 16 and np.isfinite(orb_hi[k]):
                    if c[k] > orb_hi[k]: side = 1; sl_dist = max(c[k] - orb_lo[k], 0.5 * A[k])
                    elif c[k] < orb_lo[k]: side = -1; sl_dist = max(orb_hi[k] - c[k], 0.5 * A[k])
            elif strategy == 'SESS':
                if hr[k] == 13: side = 1 * kw.get('sign', 1); sl_dist = 3 * A[k]
                elif hr[k] == 21: side = -1 * kw.get('sign', 1); sl_dist = 3 * A[k]
            if side != 0 and sl_dist and sl_dist > 0:
                risk = kw.get('risk', 0.01) * eq
                q = risk / sl_dist
                q = min(q, 2.0 * eq / px)            # cap notional at 2x equity (firm rule)
                pos = side; entry = px; qty = q * side; hold = 0
                stop = px - sl_dist if side > 0 else px + sl_dist
                trail = stop
                tp = (px + tp_d if side > 0 else px - tp_d) if strategy == 'RSI' else 0.0
                eq -= abs(q) * px * FEE
        eqs[k] = eq
    s = pd.Series(eqs, index=idx).dropna()
    r = s.pct_change().dropna(); dd = (1 + r).groupby(r.index.floor('D')).prod() - 1
    yrs = (s.index[-1] - s.index[0]).days / 365.25
    return dict(sym=sym, sharpe=dd.mean() / (dd.std() + 1e-12) * np.sqrt(365), cagr=(s.iloc[-1] / s.iloc[0]) ** (1 / yrs) - 1,
                mdd=(s / s.cummax() - 1).min(), trades_per_year=trades / yrs, curve=s)


configs = [('DON n72 trail3', 'DON', dict(n=72)), ('DON n168 trail3', 'DON', dict(n=168)), ('DON n24 trail2', 'DON', dict(n=24, trail_atr=2.0, sl_atr=2.0)),
           ('MAX 24/168', 'MAX', dict(f=24, s=168)), ('MAX 48/336', 'MAX', dict(f=48, s=336)),
           ('RSI14 30/70 TP2 SL1.5', 'RSI', dict()), ('VBO 2sig hold24', 'VBO', dict(max_hold=24)), ('ORB 4h', 'ORB', dict()),
           ('SESS long US / short Asia', 'SESS', dict(sign=1)), ('SESS short US / long Asia', 'SESS', dict(sign=-1))]
rows = []
for label, strat, kw in configs:
    t0 = time.time(); curves = []; per = []
    for sym in MAJ:
        r = simulate(sym, strat, **kw)
        if r is None:
            continue
        per.append(f'{sym[:-4]} {r["sharpe"]:+.2f}'); curves.append(r['curve'].pct_change().fillna(0))
    if not curves:
        continue
    basket = pd.concat(curves, axis=1).mean(1); eqb = (1 + basket).cumprod()
    dd = (1 + basket).groupby(basket.index.floor('D')).prod() - 1
    yrs = (eqb.index[-1] - eqb.index[0]).days / 365.25
    sh = dd.mean() / (dd.std() + 1e-12) * np.sqrt(365); cagr = eqb.iloc[-1] ** (1 / yrs) - 1; mdd = (eqb / eqb.cummax() - 1).min()
    by = {y: (g.mean() / (g.std() + 1e-12) * np.sqrt(365)) for y, g in dd.groupby(dd.index.year)}
    rows.append(dict(label=label, sharpe=sh, cagr=cagr, mdd=mdd))
    print(f'{label:<28} basket Sh {sh:5.2f} CAGR {cagr*100:6.1f}% MDD {mdd*100:6.1f}% | by year ' + ' '.join(f'{y}:{v:.1f}' for y, v in by.items())
          + ' | per symbol: ' + ', '.join(per) + f' ({time.time()-t0:.0f}s)', flush=True)
pd.DataFrame(rows).to_csv('out_exp18_classic.csv', index=False)
print('DONE')
