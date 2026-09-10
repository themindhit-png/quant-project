#!/usr/bin/env python3
"""Vectorised research backtester for cross-sectional crypto-perp portfolios (hourly grid). v2

Conventions (no look-ahead):
  * panel index = bar open_time (UTC). Bar t closes at t+1h. A decision made with data <= t
    (bar t closed) is executed at close[t] and earns ret[t+1] = close[t+1]/close[t]-1.
  * target weights are produced by a signal function at rebalance bars only; between rebalances
    dollar positions drift with prices.
  * costs: fee + slippage (bps) on traded notional; slippage may depend on liquidity (slip_fn).
  * funding: at hour t a position held into t pays pos*rate[t] (long pays when rate>0).
  * data gaps > 24h and delistings: position force-closed at the last valid close before the gap
    (optional extra `delist_shock` loss on notional); returns across gaps are zeroed; a symbol
    becomes eligible again only min_age_h after the new listing segment starts.
  * optional per-position stop-loss using intra-bar high/low with conservative fill at the worse
    of stop level and bar close.
"""
import os, json, time, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
np.seterr(all='ignore')

ROOT = os.path.dirname(os.path.abspath(__file__))
PAN = os.path.join(ROOT, 'data', 'panels')
GAP_H = 24


class Data:
    def __init__(self, start=None, end=None, symbols=None, min_turn_ever=1e6, load_hl=False,
                 load_premium=False):
        t0 = time.time()
        s0 = pd.Timestamp(start, tz='UTC') - pd.Timedelta(days=60) if start else None
        e = pd.Timestamp(end, tz='UTC') if end else None
        qvol = pd.read_parquet(os.path.join(PAN, 'qvol.parquet'), columns=symbols).loc[s0:e]
        t24 = qvol.rolling(24, min_periods=6).sum()
        del qvol
        ever = t24.max() >= min_turn_ever
        cols = [c for c in t24.columns if ever.get(c, False) and str(c).isascii()]
        t24 = t24[cols]
        close = pd.read_parquet(os.path.join(PAN, 'close.parquet'), columns=cols).loc[s0:e]
        fund = pd.read_parquet(os.path.join(PAN, 'funding.parquet'), columns=cols).loc[s0:e]
        self.idx = close.index
        self.cols = np.array(cols)
        close_v = close.values.astype(np.float64)
        self.t24 = t24.values.astype(np.float32)       # memory: f32 is ample for turnover
        self.fund = fund.values.astype(np.float32)      # and for funding rates (~1e-4)
        del close, t24, fund
        T, N = close_v.shape
        self.valid = np.isfinite(close_v) & (close_v > 0)
        # --- listing segments: age since segment start, force-close events before gaps ---
        self.age = np.full((T, N), -1, dtype=np.int32)
        self.close_event = np.zeros((T, N), dtype=bool)
        vi = self.valid.astype(np.int8)
        for j in range(N):
            idx = np.flatnonzero(vi[:, j])
            if len(idx) == 0:
                continue
            gaps = np.diff(idx)
            brk = np.flatnonzero(gaps > GAP_H)
            starts = np.concatenate(([idx[0]], idx[brk + 1]))
            ends = np.concatenate((idx[brk], [idx[-1]]))
            for s_, e_ in zip(starts, ends):
                self.age[s_:e_ + 1, j] = np.arange(e_ - s_ + 1)
                if e_ < T - 1:                      # gap or delisting follows this segment
                    self.close_event[e_, j] = True
        # forward-filled prices and returns; zero returns across gaps
        cff = pd.DataFrame(close_v).ffill().values
        del close_v
        self.cff = cff
        r = np.full_like(cff, np.nan)
        r[1:] = cff[1:] / cff[:-1] - 1.0
        r[~np.isfinite(cff)] = np.nan
        # a bar with age 0 (segment start) must not carry the jump from the previous segment
        r[self.age == 0] = np.nan
        # inside a gap (invalid bars after a close_event) no return accrues
        r[~self.valid] = np.nan
        self.ret = r
        self.hh = (self.idx.asi8 // 10**9 // 3600).astype(np.int64)
        self.start_i = int(self.idx.searchsorted(pd.Timestamp(start, tz='UTC'))) if start else 0
        self.high = self.low = self.prem = None
        if load_hl:
            self.high = pd.read_parquet(os.path.join(PAN, 'high.parquet'), columns=cols).loc[s0:e].values.astype(np.float32)
            self.low = pd.read_parquet(os.path.join(PAN, 'low.parquet'), columns=cols).loc[s0:e].values.astype(np.float32)
        if load_premium and os.path.exists(os.path.join(PAN, 'premium.parquet')):
            self.prem = pd.read_parquet(os.path.join(PAN, 'premium.parquet'), columns=cols).loc[s0:e].values.astype(np.float32)
        self.prev_mask = None
        print(f'Data: {T} h x {N} syms, {self.idx[0]} -> {self.idx[-1]} ({time.time()-t0:.1f}s)')

    def universe(self, i, min_age_h=720, min_turn=1e7, top_n=100, exit_n=None, blacklist=(),
                 exclude_close_soon=True):
        """Point-in-time eligible set at bar i (bar i closed): current listing segment age >=
        min_age_h, valid price at i, 24h turnover >= min_turn, top_n by turnover. With exit_n >
        top_n, names already held (self.prev_mask) stay eligible while ranked <= exit_n."""
        t = self.t24[i]
        m = (self.age[i] >= min_age_h) & self.valid[i] & np.isfinite(t) & (t >= min_turn)
        if len(blacklist):
            m &= ~np.isin(self.cols, list(blacklist))
        if top_n and m.sum() > top_n:
            score = np.where(m, t, -np.inf)
            order = np.argsort(-score)
            rank = np.empty(len(m), dtype=np.int64); rank[order] = np.arange(len(m))
            keep = rank < top_n
            if exit_n and self.prev_mask is not None:
                keep |= (self.prev_mask & (rank < exit_n))
            m &= keep
        self.prev_mask = m.copy()
        return m


def slip_model(scale=1.0, floor=1.0, a=8.0, b=-0.35):
    """Liquidity-dependent taker slippage (bps, beyond exchange fee): a*(t24/1e6)^b floored.
    Bybit tick data (Sep 2026) measured half-spreads: ~2.7bp (<$1M/day), 1.4 ($1-3M), 1.1 ($3-10M),
    0.9 ($10-50M), 0.5 ($50M-1B), 0.02 (>$1B). Defaults give 8.0/3.6/2.0/1.25/0.7 bps at
    $1M/10M/50M/200M/1B — ~3x measured half-spread to cover impact and adverse selection."""
    def f(data, i):
        t = data.t24[i]
        t = np.where(np.isfinite(t) & (t > 0), t, 1e5)
        return np.maximum(floor, a * (t / 1e6) ** b) * scale
    return f


def quantile_ls(sig, top_frac=0.2, inv_vol=None, min_n=10):
    """Long top / short bottom quantile; each side sums to 1.0 (equal or inverse-vol weights)."""
    s = np.where(np.isfinite(sig), sig, np.nan)
    n = int(np.isfinite(s).sum())
    w = np.zeros_like(s)
    if n < min_n:
        return w
    k = max(2, int(round(n * top_frac)))
    order = np.argsort(np.where(np.isfinite(s), s, -np.inf))
    valid_idx = order[-n:]
    shorts, longs = valid_idx[:k], valid_idx[-k:]
    if inv_vol is None:
        w[longs] = 1.0 / k; w[shorts] = -1.0 / k
    else:
        iv = 1.0 / np.where(np.isfinite(inv_vol) & (inv_vol > 0), inv_vol, np.nan)
        wl = np.nan_to_num(iv[longs]); ws = np.nan_to_num(iv[shorts])
        if wl.sum() > 0: w[longs] = wl / wl.sum()
        if ws.sum() > 0: w[shorts] = -ws / ws.sum()
    return w


def rank_weights(sig, mask=None, power=1.0):
    """Centered-rank weights (each side sums to 1). power>1 concentrates in the tails."""
    s = np.where(np.isfinite(sig), sig, np.nan)
    if mask is not None:
        s = np.where(mask, s, np.nan)
    ok = np.isfinite(s)
    w = np.zeros_like(s)
    n = ok.sum()
    if n < 10:
        return w
    r = np.empty(n); r[np.argsort(s[ok])] = np.arange(n)
    c = r - (n - 1) / 2.0
    c = np.sign(c) * np.abs(c) ** power
    pos = c.clip(min=0).sum()
    if pos > 0:
        w[ok] = c / pos
    return w


def metrics(eq, label='', verbose=True):
    eq = pd.Series(eq).dropna()
    r = eq.pct_change().dropna()
    d = (1 + r).groupby(r.index.floor('D')).prod() - 1
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / max(yrs, 1e-6)) - 1 if eq.iloc[-1] > 0 else -1.0
    sd = d.std()
    sh = d.mean() / (sd + 1e-12) * np.sqrt(365)
    dd = eq / eq.cummax() - 1
    m = (1 + d).groupby(d.index.strftime('%Y-%m')).prod() - 1
    out = dict(cagr=cagr, sharpe=sh, dvol=sd, mdd=dd.min(), calmar=cagr / (abs(dd.min()) + 1e-9),
               worst_day=d.min(), p1=d.quantile(.01), pos_months=(m > 0).mean(), years=yrs,
               end=eq.iloc[-1], sortino=d.mean() / (d[d < 0].std() + 1e-12) * np.sqrt(365),
               ann_ret=d.mean() * 365)
    by = {}
    for y in sorted(set(d.index.year)):
        dy = d[d.index.year == y]
        if len(dy) >= 20:
            by[y] = ((1 + dy).prod() - 1, dy.mean() / (dy.std() + 1e-12) * np.sqrt(365), dy.min())
    out['by_year'] = by
    if verbose:
        print(f'== {label} {eq.index[0].date()}->{eq.index[-1].date()} ({yrs:.2f}y)')
        print(f'  CAGR {cagr*100:+.2f}%  Sharpe {sh:.2f}  Sortino {out["sortino"]:.2f}  dvol {sd*100:.2f}%  '
              f'MDD {dd.min()*100:.1f}%  Calmar {out["calmar"]:.2f}  worst day {d.min()*100:.2f}%  '
              f'p1 {d.quantile(.01)*100:.2f}%  +months {out["pos_months"]*100:.0f}%')
        print('  by year: ' + '  '.join(f'{y}: {v*100:+.1f}% (Sh {s:.2f}, wd {w*100:.1f}%)' for y, (v, s, w) in by.items()))
    return out


def run(data, signal_fn, reb_h=24, band=0.0, fee_bps=5.5, slip_bps=None, slip_fn=None,
        gross=1.0, vol_target=None, vol_win_d=30, max_lev=3.0, pos_cap=0.02, cap_short=None,
        delist_shock=0.05, start_eq=10000.0, min_trade=15.0, uni_kwargs=None, smooth=0.0,
        trade_frac=1.0, verbose=True, label='', ret_diag=False, maker_share=0.0, maker_fee_bps=2.0,
        attrib=False, pos_cap_abs=None, stop_pct=None, stop_cooldown_h=48, reb_offset=0,
        max_gross_x=None, daily_stop=None, cap_exempt=None, cap_exempt_val=0.30, stop_side='both', attrib_from=None,
        cap_young=None, young_h=180 * 24, vt_mode='direct', vt_smooth=0.0, vt_delever='realised',
        end_i=None, maker_adverse_bps=0.0, band_fn=None):
    """Generic engine. signal_fn(data, i, mask) -> target weights (N,), each side sums ~1.
    band: skip trades where |target-current| < band*|target| (unless full close).
    smooth: EMA smoothing of target weights across rebalances. trade_frac: partial rebalance.
    vol_target: daily vol target using realised strategy vol over vol_win_d days.
    stop_pct: per-position stop (fraction adverse move from entry, checked on bar high/low).
    daily_stop: if today's pnl <= -daily_stop*eq_at_day_start -> flatten until next UTC day.
    max_gross_x: hard cap on gross notional / equity."""
    uni_kwargs = uni_kwargs or {}
    if slip_fn is None and slip_bps is None:
        slip_fn = slip_model()
    T, N = data.ret.shape
    i0 = max(data.start_i, 24 * 35)
    pos = np.zeros(N); entry = np.zeros(N)
    eq = start_eq
    eq_path = np.full(T, np.nan)
    turn_total = fee_total = slip_total = fund_total = delist_total = stop_total = 0.0
    turn_frac = fee_frac = slip_frac = fund_frac = 0.0
    hourly_pnl = np.zeros(T)
    w_prev_target = None
    lev = 1.0; lev_eff = 1.0; n_clip = 0; n_reb = 0
    pnl_hist = []; pnl_hist_u = []                # levered and unlevered strategy returns per hour
    n_pos_hist, gross_hist, lev_hist = [], [], []
    cooldown_until = np.zeros(N, dtype=np.int64)
    n_stops = 0
    hh = data.hh; ret = data.ret; fund = data.fund; cev = data.close_event
    high = data.high; low = data.low; cff = data.cff
    cost_per_unit = fee_bps * (1 - maker_share) + (maker_fee_bps + maker_adverse_bps) * maker_share   # maker_adverse_bps: adverse selection of maker fills
    cap_short = cap_short or pos_cap
    exempt = np.isin(data.cols, list(cap_exempt)) if cap_exempt else np.zeros(N, dtype=bool)
    A = None
    if attrib:
        A = dict(price=np.zeros(N), funding=np.zeros(N), cost=np.zeros(N), traded=np.zeros(N),
                 hours_held=np.zeros(N), long_pnl=np.zeros(N), short_pnl=np.zeros(N), stops=np.zeros(N))
    day_start_eq = eq; day_code = hh[i0] // 24; halted_day = -1
    af = int(data.idx.searchsorted(pd.Timestamp(attrib_from, tz='UTC'))) if attrib_from is not None else 0
    data.prev_mask = None
    day_sym = np.zeros(N); bad_days = []          # per-symbol P&L of the current UTC day (diagnostics)

    def trade_cost(traded_vec, i):
        if slip_fn is not None:
            slv = traded_vec * slip_fn(data, i) / 1e4
        else:
            slv = traded_vec * slip_bps / 1e4
        return slv, traded_vec * cost_per_unit / 1e4

    i_end = T - 1 if end_i is None else min(int(end_i), T - 1)
    for i in range(i0, i_end):
        # ---- new UTC day bookkeeping
        dc = hh[i] // 24
        if dc != day_code:
            if A is not None and day_start_eq > 0 and day_sym.sum() / day_start_eq <= -0.015:
                top = np.argsort(day_sym)[:8]
                bad_days.append((str(data.idx[i - 1].date()), round(day_sym.sum() / day_start_eq * 100, 2),
                                 [(str(data.cols[j]), round(day_sym[j] / day_start_eq * 100, 2)) for j in top]))
            day_sym[:] = 0.0
            day_code = dc; day_start_eq = eq
        # ---- funding at hour i on positions held into i
        if i > i0:
            fr = fund[i]
            if fr.any():
                fv = -pos * np.where(np.isfinite(fr), fr, 0.0)
                f = float(np.nansum(fv))
                fund_frac += f / max(eq, 1e-9)
                eq += f; fund_total += f
                if A is not None and i >= af:
                    A['funding'] += fv
        # ---- forced closes: gap/delisting events at this bar (close at bar close, plus shock)
        fc = (pos != 0) & cev[i]
        if fc.any():
            loss = delist_shock * np.abs(pos[fc]).sum()
            eq -= loss; delist_total += loss
            pos[fc] = 0.0
        # ---- rebalance?
        if (hh[i] - reb_offset) % reb_h == 0 and (halted_day != dc):
            mask = data.universe(i, **uni_kwargs)
            mask &= (cooldown_until <= i)
            data.cur_pos = pos; data.cur_eq = eq; data.cur_lev = lev; data.cur_gross = gross   # read-only state for strategies
            w = signal_fn(data, i, mask)
            w = np.where(np.isfinite(w) & mask, w, 0.0)
            if smooth > 0 and w_prev_target is not None:
                w = smooth * w_prev_target + (1 - smooth) * w
            w_prev_target = w.copy()
            if vol_target and len(pnl_hist) >= 24 * 10:
                # 'direct': lev = target / realised vol of the UNLEVERED book (stable). 'recursive' (legacy): lev *= target/dvol
                # on levered returns with a lagging window -> bang-bang oscillation between the clips (see exp20g).
                src = pnl_hist_u if vt_mode == 'direct' else pnl_hist
                ph = np.array(src[-vol_win_d * 24:])
                ph = ph[len(ph) % 24:]
                dvol = ph.reshape(-1, 24).sum(1).std(ddof=1) if len(ph) >= 48 else np.nan
                if np.isfinite(dvol) and dvol > 1e-6:
                    if vt_mode == 'direct':
                        lev_new = float(np.clip(vol_target / dvol, 0.05, max_lev))
                        lev = lev_new if vt_smooth <= 0 else float(vt_smooth * lev + (1 - vt_smooth) * lev_new)
                    else:
                        lev = float(np.clip(lev * vol_target / dvol, 0.05, max_lev))
            tgt = w * gross * lev * eq
            capl = pos_cap * eq; caps = cap_short * eq
            if pos_cap_abs:
                capl = min(capl, pos_cap_abs); caps = min(caps, pos_cap_abs)
            capl_v = np.where(exempt, cap_exempt_val * eq, capl); caps_v = np.where(exempt, cap_exempt_val * eq, caps)
            if cap_young:                            # jump-risk cap: young names (segment age < young_h) get a tighter per-name cap
                ym = (data.age[i] < young_h) & ~exempt
                capl_v = np.where(ym, np.minimum(capl_v, cap_young * eq), capl_v); caps_v = np.where(ym, np.minimum(caps_v, cap_young * eq), caps_v)
            tgt = np.where(tgt > 0, np.minimum(tgt, capl_v), np.maximum(tgt, -caps_v))
            ls, ss = tgt[tgt > 0].sum(), -tgt[tgt < 0].sum()
            if ls > 0 and ss > 0:                    # restore dollar neutrality after caps
                mm = min(ls, ss)
                tgt[tgt > 0] *= mm / ls; tgt[tgt < 0] *= mm / ss
            if max_gross_x:
                g = np.abs(tgt).sum()
                if g > max_gross_x * eq:
                    tgt *= max_gross_x * eq / g; n_clip += 1
            n_reb += 1
            # dust rule: targets below the minimum trade size are zero -> full close (no tails)
            tgt = np.where(np.abs(tgt) < min_trade, 0.0, tgt)
            # realised leverage of the book (after caps / neutrality / gross cap / dust) relative to the raw weights:
            # the vol target de-levers by THIS, not by the intended lev (Opus: otherwise a capped book under-estimates
            # its unlevered vol and lev creeps up — a weak version of the recursion fixed in exp20h)
            gw = float(np.abs(w).sum())
            lev_eff = float(np.abs(tgt).sum() / max(eq * gw, 1e-9)) if gw > 1e-12 else lev
            diff = tgt - pos
            full_close = (tgt == 0) & (pos != 0)
            bandv = band * band_fn(data, i) if band_fn is not None else band     # optional per-name band (cost-dependent)
            trade = full_close | (np.abs(diff) >= np.maximum(min_trade, bandv * np.abs(tgt)))
            dpos = np.where(trade, diff * trade_frac, 0.0)
            dpos = np.where(full_close, diff, dpos)
            traded = np.abs(dpos)
            tv = traded.sum()
            if tv > 0:
                slv, feev = trade_cost(traded, i)
                sl, fee = float(slv.sum()), float(feev.sum())
                e_ = max(eq, 1e-9)
                turn_frac += tv / e_; fee_frac += fee / e_; slip_frac += sl / e_
                eq -= fee + sl
                fee_total += fee; slip_total += sl; turn_total += tv
                newpos = pos + dpos
                px = cff[i]
                # entry price bookkeeping (VWAP on adds, reset on open/flip)
                flip = (np.sign(newpos) != np.sign(pos)) & (newpos != 0)
                add = (np.sign(newpos) == np.sign(pos)) & (np.abs(newpos) > np.abs(pos)) & (pos != 0)
                entry = np.where(flip | ((pos == 0) & (newpos != 0)), px, entry)
                with np.errstate(all='ignore'):
                    entry = np.where(add, (np.abs(pos) * entry + np.abs(dpos) * px) / np.abs(newpos), entry)
                pos = newpos
                if A is not None and i >= af:
                    A['cost'] += slv + feev; A['traded'] += traded
            n_pos_hist.append((pos != 0).sum()); gross_hist.append(np.abs(pos).sum() / max(eq, 1e-9))
            lev_hist.append((i, lev, float(np.abs(pos).sum() / max(eq, 1e-9)), int((pos != 0).sum())))
        eq_path[i] = eq
        # ---- P&L over next bar (i+1)
        r = ret[i + 1]
        r = np.where(np.isfinite(r), r, 0.0)
        pv = pos * r
        # ---- intra-bar stop-loss on bar i+1 (conservative fill: worse of stop level and close)
        if stop_pct and high is not None and (pos != 0).any():
            hi = high[i + 1]; lo = low[i + 1]; c1 = cff[i + 1]
            sh_hit = (pos < 0) & np.isfinite(hi) & (hi >= entry * (1 + stop_pct)) if stop_side in ('both', 'short') else np.zeros(N, bool)
            lg_hit = (pos > 0) & np.isfinite(lo) & (lo <= entry * (1 - stop_pct)) if stop_side in ('both', 'long') else np.zeros(N, bool)
            hit = sh_hit | lg_hit
            if hit.any():
                # fill price: shorts at max(stop, close), longs at min(stop, close)
                fill = np.where(sh_hit, np.maximum(entry * (1 + stop_pct), c1),
                                np.minimum(entry * (1 - stop_pct), c1))
                prev_px = cff[i]
                r_fill = np.where(hit, fill / prev_px - 1.0, r)
                pv = pos * r_fill
                # cost of the stop exit (taker + slippage) on the position notional at fill
                notional = np.abs(pos * (1 + r_fill)) * hit
                slv, feev = trade_cost(notional, i)
                c_ = float(slv.sum() + feev.sum())
                eq -= c_; stop_total += c_; fee_total += float(feev.sum()); slip_total += float(slv.sum())
                turn_frac += notional.sum() / max(eq, 1e-9)
                n_stops += int(hit.sum())
                cooldown_until[hit] = i + 1 + stop_cooldown_h
                if A is not None and i >= af:
                    A['stops'] += hit; A['cost'] += slv + feev
        pnl = float(pv.sum())
        pnl_hist.append(pnl / max(eq, 1e-9)); pnl_hist_u.append(pnl / max(eq * (lev_eff if vt_delever == 'realised' else lev * gross), 1e-9))
        hourly_pnl[i + 1] = pnl
        eq += pnl
        if A is not None and i >= af:
            day_sym += pv
            A['price'] += pv
            A['hours_held'] += (pos != 0)
            A['long_pnl'] += np.where(pos > 0, pv, 0.0)
            A['short_pnl'] += np.where(pos < 0, pv, 0.0)
        pos = pos * (1 + r)
        if stop_pct and high is not None and (pos != 0).any():
            pos = np.where(hit, 0.0, pos) if 'hit' in dir() and hit.any() else pos
        # ---- daily stop (flatten at this bar's close, no new trades until next UTC day)
        if daily_stop and eq <= day_start_eq * (1 - daily_stop) and (pos != 0).any():
            traded = np.abs(pos)
            slv, feev = trade_cost(traded, i + 1)
            c_ = float(slv.sum() + feev.sum())
            eq -= c_; fee_total += float(feev.sum()); slip_total += float(slv.sum())
            turn_frac += traded.sum() / max(eq, 1e-9)
            pos[:] = 0.0
            halted_day = hh[i + 1] // 24
        if eq <= 0:
            eq_path[i + 1:] = 0.0
            break
    eq_path[i_end] = eq
    eqs = pd.Series(eq_path, index=data.idx).iloc[i0:i_end + 1]
    yrs = (eqs.index[-1] - eqs.index[0]).days / 365.25
    m = metrics(eqs, label=label, verbose=verbose)
    m.update(fees=fee_frac / yrs, slip=slip_frac / yrs, funding=-fund_frac / yrs,
             delist=delist_total / start_eq / yrs, fees_usd=fee_total, slip_usd=slip_total,
             funding_usd=fund_total, delist_usd=delist_total, n_stops=n_stops, stop_cost_usd=stop_total)
    m['turnover_x'] = turn_frac / yrs
    m['avg_npos'] = float(np.mean(n_pos_hist)) if n_pos_hist else 0
    m['avg_gross'] = float(np.mean(gross_hist)) if gross_hist else 0
    if verbose:
        print(f'  costs %/yr: fees {m["fees"]*100:.2f}  slip {m["slip"]*100:.2f}  funding {m["funding"]*100:+.2f}  '
              f'delist {m["delist"]*100:.2f}  | stops {n_stops}')
        print(f'  turnover {m["turnover_x"]:.0f}x/yr  avg positions {m["avg_npos"]:.0f}  avg gross {m["avg_gross"]:.2f}x  end {eq:,.0f}')
    m['lev_hist'] = lev_hist
    m['clip_frac'] = n_clip / max(n_reb, 1)          # share of rebalances where the gross cap bound
    if A is not None:
        m['attrib'] = pd.DataFrame(A, index=data.cols)
        m['bad_days'] = sorted(bad_days, key=lambda x: x[1])
        if verbose and bad_days:
            print(f'  worst days (<= -1.5%): {len(bad_days)}')
            for dte, pct, top in m['bad_days'][:8]:
                print(f'    {dte} {pct:+.2f}%  ' + ', '.join(f'{s} {v:+.2f}' for s, v in top[:5]))
    if ret_diag:
        return m, eqs, pd.Series(hourly_pnl, index=data.idx).iloc[i0:]
    return m, eqs
