#!/usr/bin/env python3
"""Experiment (owner's question, 2026-10-28): grid trading on liquid USDT perps — a Bybit-style static grid vs a trailing ('shifting window')
grid vs a 'smart' grid (volatility-adaptive step + trend pause). NOTE: grid trading is forbidden at Mubite (T&C 13.1.7) — own account only.

Hourly OHLC (data/panels, Binance), 20 liquid coins, 2022-01..2026-10, equal capital per coin, each coin's grid may hold at most MAXPOS x
its capital (notional). Fills: a level is filled when the bar's range touches it, at the level price (maker 0.02 %); the intrabar path is
open -> low -> high -> close on up bars, open -> high -> low -> close on down bars (a round trip inside one bar is only counted when the path
allows it). Stops: market (taker 0.055 % + 0.05 % slippage). Funding paid/received on the held position at the funding hours.
Run: python3 research/exp_grid.py"""
import os, sys, numpy as np, pandas as pd
from numba import njit

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SYMS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'DOGEUSDT', 'BNBUSDT', 'ADAUSDT', 'AVAXUSDT', 'LINKUSDT', 'LTCUSDT', 'DOTUSDT', 'TRXUSDT',
        'BCHUSDT', 'NEARUSDT', 'ATOMUSDT', 'APTUSDT', 'ARBUSDT', 'OPUSDT', 'FILUSDT', 'UNIUSDT']
MAKER, TAKER, SLIP = 0.0002, 0.00055, 0.0005


@njit(cache=True)
def grid_sim(o, h, l, c, fund, sig, mode, step_k, n_lvl, maxpos, stop_lv, trend_z, range_pct, month_start, bpd=24):
    """One coin, capital 1.0. mode 0 = static monthly range (Bybit-like), 1 = trailing grid, 2 = trailing + vol step + trend pause.
    Returns equity per bar."""
    T = len(c); eq = np.full(T, np.nan); cash = 1.0; units = 0.0
    q = maxpos / n_lvl                                    # notional per level (fraction of capital)
    ref = np.nan; step = 0.01; lo_b = 0.0; hi_b = 0.0; started = False
    for i in range(T):
        if not (c[i] > 0) or not (h[i] > 0) or not (l[i] > 0) or not (o[i] > 0):
            if started:
                eq[i] = eq[i - 1]
            continue
        if not started:
            started = True; ref = o[i]
        e_now = cash + units * o[i]
        if e_now <= 0.05:                                   # grid capital lost: the coin is out
            eq[i] = e_now; cash = e_now; units = 0.0
            continue
        if i % bpd == 0 or month_start[i]:                   # level size follows the CURRENT capital (Bybit grid: fixed investment)
            q = maxpos * e_now / n_lvl
        # ---- (re)configure
        if mode == 0 and month_start[i]:
            if units != 0.0:                               # close the old grid at the open (market)
                cash += units * o[i] * (1 - np.sign(units) * SLIP) - abs(units) * o[i] * TAKER; units = 0.0
                q = maxpos * cash / n_lvl
            lo_b = o[i] * (1 - range_pct); hi_b = o[i] * (1 + range_pct)
            step = (np.log(hi_b) - np.log(lo_b)) / n_lvl; ref = o[i]
        if mode >= 1:
            s_ = sig[i] if sig[i] > 0 else 0.03
            step = step_k * s_ if mode == 2 else step_k * 0.03     # mode 1: fixed 3 % daily-vol-equivalent step x k
        up_bar = c[i] >= o[i]
        # path points: open -> (low, high) in order -> close
        p1 = l[i] if up_bar else h[i]; p2 = h[i] if up_bar else l[i]
        for leg in range(2):
            target = p1 if leg == 0 else p2
            # walk the grid from ref towards target, one level at a time
            for _ in range(200):
                buy_px = ref * np.exp(-step); sell_px = ref * np.exp(step)
                pos_lv = units * ref / q if q > 0 else 0.0
                if target <= buy_px:
                    if mode == 0 and buy_px < lo_b:
                        break
                    if pos_lv >= n_lvl - 1e-9:                     # inventory full
                        if mode >= 1 and target <= ref * np.exp(-step * stop_lv):   # stop: price ran stop_lv steps past a full grid
                            sp = ref * np.exp(-step * stop_lv)
                            cash += units * sp * (1 - SLIP) - abs(units) * sp * TAKER; units = 0.0; ref = sp
                            continue
                        break
                    if mode == 2 and sig[i] > 0 and (np.log(c[i - bpd] / ref) if i >= bpd and c[i - bpd] > 0 else 0.0) > trend_z * sig[i]:
                        break                                      # strong down-trend over 24 h: do not add longs
                    u = q / buy_px; units += u; cash -= q + q * MAKER; ref = buy_px
                elif target >= sell_px:
                    if mode == 0 and sell_px > hi_b:
                        break
                    if pos_lv <= -n_lvl + 1e-9:
                        if mode >= 1 and target >= ref * np.exp(step * stop_lv):
                            sp = ref * np.exp(step * stop_lv)
                            cash += units * sp * (1 + SLIP) - abs(units) * sp * TAKER; units = 0.0; ref = sp
                            continue
                        break
                    if mode == 2 and sig[i] > 0 and (np.log(ref / c[i - bpd]) if i >= bpd and c[i - bpd] > 0 else 0.0) > trend_z * sig[i]:
                        break                                      # strong up-trend: do not add shorts
                    u = q / sell_px; units -= u; cash += q - q * MAKER; ref = sell_px
                else:
                    break
        if fund[i] != 0.0 and np.isfinite(fund[i]):
            cash -= units * c[i] * fund[i]
        eq[i] = cash + units * c[i]
    return eq


def stats(eq):
    d = eq.resample('D').last().dropna(); r = d.pct_change().dropna()
    if len(d) < 30:
        return dict(cagr=np.nan, sharpe=np.nan, mdd=np.nan)
    yrs = len(d) / 365.0
    return dict(cagr=round(((d.iloc[-1] / d.iloc[0]) ** (1 / yrs) - 1) * 100, 1), sharpe=round(r.mean() / r.std() * np.sqrt(365), 2) if r.std() > 0 else 0,
                mdd=round((d / d.cummax() - 1).min() * 100, 1))


def main():
    global SYMS
    P = os.path.join(ROOT, 'data', 'panels')
    M1 = '--m1' in sys.argv; bpd = 1440 if M1 else 24
    if M1:                                               # 1-minute bars (research/dl_m1.py), 2023-01..2026-09, 5 coins
        SYMS = [f[:-8] for f in sorted(os.listdir(os.path.join(ROOT, 'data', 'm1'))) if f.endswith('.parquet')]
        m = {s: pd.read_parquet(os.path.join(ROOT, 'data', 'm1', s + '.parquet')) for s in SYMS}
        O, Hh, Ll, Cc = (pd.DataFrame({s: m[s][k] for s in SYMS}).astype('float64') for k in 'ohlc')
        fh = pd.read_parquet(os.path.join(P, 'funding.parquet'), columns=SYMS).fillna(0.0)
        Fu = fh.reindex(Cc.index).fillna(0.0)                    # funding at the settlement minute (hh:00)
    else:
        rd = lambda n: pd.read_parquet(os.path.join(P, n + '.parquet'), columns=SYMS).loc['2021-12-01':]
        O, Hh, Ll, Cc, Fu = rd('open'), rd('high'), rd('low'), rd('close'), rd('funding').fillna(0.0)
    idx = Cc.index; ms = np.r_[True, idx.month[1:] != idx.month[:-1]]
    dr = np.log(Cc).diff().resample('D').sum(min_count=1)
    sig = dr.rolling(30, min_periods=10).std().shift(1).reindex(idx.floor('D')).set_axis(idx)
    variants = {
        'A static ±15 %, 30 lv, monthly (Bybit-like)': dict(mode=0, step_k=0, n_lvl=30, maxpos=1.0, stop_lv=0, trend_z=0, range_pct=0.15),
        'B trailing, step 1.5 %, 20 lv, stop +5': dict(mode=1, step_k=0.5, n_lvl=20, maxpos=1.0, stop_lv=5, trend_z=0, range_pct=0),
        'C smart: step 0.5σ, 20 lv, stop +5, trend pause 1.5σ': dict(mode=2, step_k=0.5, n_lvl=20, maxpos=1.0, stop_lv=5, trend_z=1.5, range_pct=0),
        'C2 smart: step 0.3σ, 30 lv, stop +6, pause 1σ': dict(mode=2, step_k=0.3, n_lvl=30, maxpos=1.0, stop_lv=6, trend_z=1.0, range_pct=0),
        'C3 smart: step 1σ, 10 lv, stop +4, pause 2σ': dict(mode=2, step_k=1.0, n_lvl=10, maxpos=1.0, stop_lv=4, trend_z=2.0, range_pct=0),
    }
    book = pd.read_csv(os.path.join(ROOT, 'logs', 'fs_base_h0.5_e1.csv'), index_col=0, parse_dates=True).iloc[:, 0].resample('D').last().pct_change()
    rows = []
    for name, v in variants.items():
        eqs = {}
        for s in SYMS:
            e = grid_sim(O[s].values, Hh[s].values, Ll[s].values, Cc[s].values, Fu[s].values, sig[s].fillna(0).values,
                         v['mode'], v['step_k'], v['n_lvl'], v['maxpos'], v['stop_lv'], v['trend_z'], v['range_pct'], ms, bpd)
            eqs[s] = pd.Series(e, index=idx)
        E = pd.DataFrame(eqs).loc['2022-01-01':]
        if M1:
            E = E.resample('h').last()
        port = (E.ffill().pct_change().fillna(0).mean(1) + 1).cumprod()        # equal capital, rebalanced hourly across coins
        res = {lab: stats(port.loc[a:b]) for lab, a, b in (('2022-26', '2022-01-01', None), ('2022', '2022-01-01', '2022-12-31'),
                                                            ('2023-24', '2023-01-01', '2024-12-31'), ('2025', '2025-01-01', '2025-12-31'), ('2026', '2026-01-01', None))}
        corr = port.resample('D').last().pct_change().corr(book)
        coin = {s: stats(E[s].dropna())['cagr'] for s in ('BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'DOGEUSDT') if s in E}
        rows.append(dict(variant=name, **{f'{k}_{m}': x[m] for k, x in res.items() for m in x}, corr_book=round(corr, 2), **{f'cagr_{k[:-4]}': v_ for k, v_ in coin.items()}))
        print(rows[-1], flush=True)
    df = pd.DataFrame(rows); df.to_csv(os.path.join(ROOT, 'research', 'phase4_out', f'grid_exp{"_m1" if M1 else ""}.csv'), index=False)
    pd.set_option('display.width', 250); print(df.to_string(index=False))


if __name__ == '__main__':
    main()
