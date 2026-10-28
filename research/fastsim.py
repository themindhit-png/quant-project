#!/usr/bin/env python3
"""Phase T: fast account simulator. Replays the bot's risk layer (v4/risk.py multiplier + vol target, v4/main.py guards / scale pass /
daily halt / phase target, P3 drawdown mode) on top of the book's hourly return per unit of exposure, recorded once by a full harness
run of the real bot (research/fastsim_base.sh: fees, slippage, funding, 1 h lag, short stops included). Thousands of risk settings x
starts run in minutes instead of ~1 h per setting in v4/harness_v4.py. Validated against the harness on the R3-bis runs (fastsim_check).

Books: 'A' = V2 (HIVOL_MULT 0.5), 'B' = drawdown-mode book (HIVOL 0, used while buffer < DD_BUF x allowance), 'C' = F2 (HIVOL 1).
Approximations: exposure acts linearly on the unit return (position caps are those of the base run at E0), the gross cap is applied
hourly, the vol target uses the 30-day std of the book's daily unit returns (the bot de-levers its own equity returns — same quantity).
"""
import os, numpy as np, pandas as pd
from numba import njit

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ------------------------------------------------------------------ parameter vectors (index constants, numba-friendly)
P_NAMES = ['VT', 'BASE', 'BASE_P2', 'KMAX', 'CPPI', 'KMIN', 'FG', 'ABANDON', 'THROTTLE', 'HALT_FRAC', 'RAT', 'TBUF', 'DD_BUF',
           'VT_MAX', 'MAXG', 'SAFETY', 'BOOK', 'DD_BOOK', 'KEEP_BUF']
P = {n: i for i, n in enumerate(P_NAMES)}
F_NAMES = ['T1', 'T2', 'TOTAL_DD', 'DAILY_DD', 'TRAIL', 'TRAIL_LOCK', 'MIN_D1', 'MIN_D2', 'PROF_D1', 'PROF_D2', 'NOTIONAL_X',
           'PAY_EVERY', 'FSCORE', 'MIN_PAY', 'CONSIST', 'NOCAP', 'PAUSE_H', 'PAY_COST']
F = {n: i for i, n in enumerate(F_NAMES)}
PAY_CAPS = np.array([2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 1e9])   # Mubite: % of initial per payout, uncapped from the 7th

DEFAULT = dict(VT=0.006, BASE=2.0, BASE_P2=0.0, KMAX=0.0, CPPI=0.5, KMIN=0.0, FG=0.05, ABANDON=0.0, THROTTLE=1.0, HALT_FRAC=0.5,
               RAT=0.35, TBUF=0.30, DD_BUF=0.0, VT_MAX=3.0, MAXG=2.0, SAFETY=0.85, BOOK=0, DD_BOOK=1, KEEP_BUF=0.0)


def pvec(**kw):
    d = dict(DEFAULT); d.update(kw); return np.array([float(d[n]) for n in P_NAMES])


def fvec(**kw):
    return np.array([float(kw.get(n, 0.0)) for n in F_NAMES])


# ------------------------------------------------------------------ books
LEVELS = (1, 2, 3, 4)     # exposures (lev x K) of the base runs; returns / gross / rho are interpolated between them (position caps bind more at high E)


def load_books(books=(('A', 0.5), ('B', 0), ('C', 1.0)), levels=LEVELS):
    """Returns ts, R (nb, nl, T) return per unit exposure at each level, G (nb, nl, T) gross/equity per unit exposure, V (nb, T) 30d std of
    daily unit returns at level 1 (completed days), hour (T,), day (T,), RH (nb, nl, T) realised exposure / (lev x K) of the bot (smoothing and
    caps: the vol target de-levers daily returns by the realised exposure, risk.update_lev). A missing level copies the nearest lower one."""
    R, G, V, RH, ts = [], [], [], [], None
    for name, h in books:
        Rb, Gb, RHb = [], [], []
        for e0 in levels:
            f = os.path.join(ROOT, 'logs', f'fs_base_h{h}_e{e0}.csv')
            if not os.path.exists(f):
                Rb.append(Rb[-1]); Gb.append(Gb[-1]); RHb.append(RHb[-1]); continue
            eq = pd.read_csv(f, index_col=0, parse_dates=True).iloc[:, 0]
            gx = pd.read_csv(f.replace('.csv', '_gx.csv'), index_col=0, parse_dates=True).iloc[:, 0]
            if ts is None:
                ts = eq.index
            eq = eq.reindex(ts).ffill(); r = (eq.pct_change().fillna(0.0) / e0).values
            Rb.append(r); Gb.append((gx.reindex(ts).ffill().bfill() / e0).values)
            rho = pd.read_csv(f.replace('.csv', '_rho.csv'), index_col=0, parse_dates=True).iloc[:, 0]
            RHb.append(rho[~rho.index.duplicated()].reindex(ts, method='ffill').bfill().clip(0.2, 1.5).values)
            if len(Rb) == 1:
                dr = pd.Series(r, index=ts).groupby(ts.floor('D')).sum()
                vh = dr.rolling(30, min_periods=10).std().shift(1).reindex(ts.floor('D')).values     # completed days only
                V.append(np.where(np.isfinite(vh), vh, np.nan))
        R.append(Rb); G.append(Gb); RH.append(RHb)
    day = ((ts - ts[0].floor('D')).days).values.astype(np.int64)
    return ts, np.array(R), np.array(G), np.array(V), ts.hour.values.astype(np.int64), day, np.array(RH)


@njit(cache=True)
def _at(X, b, E, i):
    """Piecewise-linear interpolation of E x X(E) over the levels 1..L (X = unit return or gross per unit); below 1 linear in E,
    above L proportional. Returns the per-unit value at exposure E."""
    L = X.shape[1]
    if E <= 1.0:
        return X[b, 0, i]
    if E >= L:
        return X[b, L - 1, i]
    k = int(E) - 1; w = E - int(E)
    return ((1 - w) * (k + 1) * X[b, k, i] + w * (k + 2) * X[b, k + 1, i]) / E


@njit(cache=True)
def _rho(RH, b, E, i):
    L = RH.shape[1]
    if E <= 1.0:
        return RH[b, 0, i]
    if E >= L:
        return RH[b, L - 1, i]
    k = int(E) - 1; w = E - int(E)
    return (1 - w) * RH[b, k, i] + w * RH[b, k + 1, i]


# ------------------------------------------------------------------ core
@njit(cache=True)
def _k(p, f, eq, init, floor, dd_amt, day_ref, halted, mode, phase, tdays, tgt):
    buf = eq - floor
    if buf <= p[6] * dd_amt:
        return 0.0
    base = p[1]
    if mode == 0 and phase >= 2 and p[2] > 0:
        base = p[2]
    kmax = max(p[3], base) if p[3] > 0 else base
    k = base * min(kmax / base, buf / (p[4] * dd_amt)) if p[4] > 0 else base
    if p[5] > 0:
        k = max(k, p[5] * base)
    if halted:
        return 0.0
    day_loss = (eq - day_ref) / init
    if day_loss <= -p[9] * f[3] / 100.0:
        return 0.0
    if day_loss <= -p[8] / 100.0:
        k *= 0.5
    if mode == 0:
        md = f[6] if phase == 1 else f[7]
        if eq >= tgt and tdays < md:
            k *= p[10]
    return k


@njit(cache=True)
def sim(R, G, V, RH, hour, day, i0, p, f, mode, phase, max_h, init, rec=np.zeros(1), rec_w=np.zeros(1), rec_p=np.zeros(1)):
    """mode 0 = challenge (phase 1/2), 1 = funded. Returns (outcome, i_end, min_eq, eq_end, paid, n_pay, fs_blocked_days).
    outcome: 0 end of data / max_h, 1 target passed, 2 bot halt (floor guard / abandon latch), 3 firm breach (total or daily)."""
    T = R.shape[2]
    eq = init; peak = init; min_eq = init
    dd_amt = init * f[2] / 100.0
    tpct = f[0] if phase == 1 else f[1]
    tgt = init * (1 + tpct / 100.0); trig = init * (1 + (tpct + p[11]) / 100.0)
    E = 0.0; kref = 0.0; book = int(p[16])
    cur_day = day[i0]; anchor = init; halted = False
    tdays = 0; pdays = 0
    paid = 0.0; n_pay = 0; last_pay_day = day[i0]; cyc_best = 0.0; cyc_pos = 0.0; fs_block = 0
    i_end = min(T - 1, i0 + max_h) if max_h > 0 else T - 1
    nd_max = (i_end - i0) // 24 + 3
    eqd = np.zeros(nd_max); mld = np.zeros(nd_max); fld = np.zeros(nd_max); nd = 0     # bot's eq_daily / mult_daily / flow_daily
    msum = 0.0; mn_ = 0; flow_today = 0.0; lev = 1.0; kl = 0.0
    pause_until = 0                                       # Mubite payout: all positions closed, no trading while it is reviewed
    for i in range(i0 + 1, i_end + 1):
        # ---- P&L of the hour with the exposure set before it (gross cap: equity multiple and firm notional of INITIAL)
        if E > 0 and not halted and i >= pause_until:
            gw = _at(G, book, E, i)
            e_eff = E
            if gw > 1e-9:
                gcap = min(p[14], f[10] * init * p[15] / eq)
                e_eff = min(E, gcap / gw)
            eq *= 1.0 + e_eff * _at(R, book, e_eff, i)
            if rec.shape[0] > i:
                rec[i] = e_eff * _at(G, book, e_eff, i)
        min_eq = min(min_eq, eq)
        if rec_w.shape[0] > i:
            rec_w[i] = eq + paid                          # wealth: equity + payouts so far (funded scaling rule)
        if rec_p.shape[0] > i:
            rec_p[i] = paid
        # ---- day roll (00:00 UTC): trading / profitable days, payouts in funded
        if day[i] != cur_day:
            pnl = eq - anchor
            if abs(pnl) >= 0.0025 * init:
                tdays += 1
            if pnl >= 0.005 * init:
                pdays += 1
            if mode == 1:
                if pnl > 0:
                    cyc_pos += pnl; cyc_best = max(cyc_best, pnl)
                if day[i] - last_pay_day >= f[11]:
                    avail = eq - init * (1.0 + p[18])
                    cap = PAY_CAPS[min(n_pay, 6)] / 100.0 * init if f[15] <= 0 else 1e18
                    amt = min(avail, cap)
                    fs_ok = f[12] <= 0 or cyc_pos <= 0 or cyc_best < f[12] / 100.0 * cyc_pos
                    if amt >= f[13] and fs_ok:
                        eq -= amt; paid += amt; n_pay += 1; flow_today -= amt; last_pay_day = day[i]; cyc_best = 0.0; cyc_pos = 0.0
                        peak = max(init, peak - amt)
                        if f[16] > 0 and E > 0:               # close everything before the request (taker cost on the gross) and wait
                            eq -= f[17] * E * _at(G, book, E, i) * eq
                            pause_until = i + int(f[16]); E = 0.0; kref = 0.0
                    elif amt >= f[13]:
                        fs_block += 1
            eqd[nd] = eq; fld[nd] = flow_today; mld[nd] = msum / mn_ if mn_ > 0 else max(lev * max(kl, 0.05), 1e-6); nd += 1
            msum = 0.0; mn_ = 0; flow_today = 0.0
            cur_day = day[i]; anchor = eq; halted = False
        peak = max(peak, eq)
        # ---- firm floors (static on initial, or trailing from the peak, optionally locked at the initial balance)
        if f[4] > 0:
            floor = peak - dd_amt
            if f[5] > 0:
                floor = min(floor, init)
        else:
            floor = init - dd_amt
        if eq <= floor or eq <= anchor - f[3] / 100.0 * init:
            return 3, i, min_eq, eq, paid, n_pay, fs_block
        # ---- bot guards: floor guard / abandon latch, daily halt, phase target
        buf = eq - floor
        if buf <= p[6] * dd_amt:
            return 2, i, min_eq, eq, paid, n_pay, fs_block
        if mode == 0 and p[7] > 0 and buf <= p[7] * dd_amt:
            return 2, i, min_eq, eq, paid, n_pay, fs_block
        if (not halted) and (eq - anchor) / init <= -p[9] * f[3] / 100.0:
            halted = True; E = 0.0
            continue
        if mode == 0:
            md = f[6] if phase == 1 else f[7]
            pd_req = f[8] if phase == 1 else f[9]
            if eq >= trig and tdays >= md and pdays >= pd_req:
                return 1, i, min_eq, eq, paid, n_pay, fs_block
        if halted or i < pause_until:
            continue
        # ---- rebalance every 8 h (01/09/17 UTC): book choice (P3), K, vol target; otherwise the hourly de-risk-only scale pass
        k = _k(p, f, eq, init, floor, dd_amt, anchor, halted, mode, phase, tdays, tgt)
        if hour[i] % 8 == 1:
            book = int(p[17]) if (p[12] > 0 and buf < p[12] * dd_amt) else int(p[16])
            if nd >= 11:                                  # risk.update_lev: realised vol of de-levered daily NAV returns
                n0 = max(0, nd - 31); m_ = nd - n0 - 1
                ru = np.empty(m_)
                for j in range(m_):
                    ru[j] = ((eqd[n0 + j + 1] - fld[n0 + j + 1]) / eqd[n0 + j] - 1.0) / max(mld[n0 + j + 1], 1e-6)
                sd = np.std(ru) * np.sqrt(m_ / (m_ - 1.0))
                if sd > 1e-6:
                    lev = min(max(p[0] / sd, 0.05), p[13])
            else:                                         # cold start: ex-ante vol of the raw weights
                v = V[book, i] / RH[book, 0, i]
                if v > 1e-9:
                    lev = min(max(p[0] / v, 0.05), p[13])
            E = lev * k; kref = k; kl = k
            gw = _at(G, book, E, i); e_c = E
            if gw > 1e-9:
                e_c = min(E, min(p[14], f[10] * init * p[15] / eq) / gw)
            msum += e_c * _rho(RH, book, e_c, i); mn_ += 1
        elif kref > 0 and k / kref < 0.85:
            E *= k / kref; kref = k; kl = k
    return 0, i_end, min_eq, eq, paid, n_pay, fs_block


# ------------------------------------------------------------------ helpers
def starts_idx(ts, first='2022-01-03', last='2026-09-07', every_days=14):
    return np.array([int(ts.searchsorted(pd.Timestamp(d, tz=ts.tz))) for d in pd.date_range(first, last, freq=f'{every_days}D')])


def run_path(books, p_ch, p_fu, f_ch, f_fu, i0, init=10000.0, gap_days=3, funded_days=365, phases=2):
    """One purchase: phase 1 -> phase 2 -> funded (gap_days between, fresh balance each stage). Returns a dict."""
    ts, R, G, V, hour, day, RH = books
    out = dict(start=str(ts[i0].date()))
    i = i0; days = 0.0
    for ph in range(1, phases + 1):
        o, ie, mn, ee, *_ = sim(R, G, V, RH, hour, day, i, p_ch, f_ch, 0, ph, 0, init)
        out[f'p{ph}'] = ('end', 'pass', 'halt', 'breach')[o]; out[f'p{ph}_days'] = (ie - i) / 24.0; days += (ie - i) / 24.0
        if o != 1:
            out['funded'] = False; out['ch_days'] = days; return out
        i = ie + 24 * gap_days
        if i >= R.shape[2] - 48:
            out['funded'] = False; out['ch_days'] = days; out[f'p{ph}'] = 'pass'; return out
    out['funded'] = True; out['ch_days'] = days
    o, ie, mn, ee, paid, npay, fsb = sim(R, G, V, RH, hour, day, i, p_fu, f_fu, 1, 1, funded_days * 24, init)
    out.update(fu=('end', 'pass', 'halt', 'breach')[o], fu_days=(ie - i) / 24.0, fu_paid=paid, fu_npay=npay, fu_fsblock=fsb,
               fu_min=mn, fu_end=ee, fu_full=(ie - i) / 24.0 >= funded_days - 1)
    return out
