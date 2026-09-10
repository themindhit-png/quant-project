#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Risk engine for PROP-SLEEVES v4: firm floors (static/trailing), CPPI multiplier, daily throttle/halt,
vol targeting, caps & neutrality, notional/margin/low-cap guards, consistency & target logic."""
import os, sys, math, time
from datetime import datetime, timezone
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C


def utc_today():
    return datetime.now(timezone.utc).date().isoformat()


class RiskEngine:
    def __init__(self, mem, log=print):
        """mem: persistent dict (see state.py) — engine reads/writes its fields there."""
        self.mem = mem
        self.log = log
        m = self.mem
        m.setdefault('initial', float(C.ACCOUNT_SIZE))
        m.setdefault('eq_peak', float(C.ACCOUNT_SIZE))
        m.setdefault('day_date', None); m.setdefault('day_anchor', None); m.setdefault('day_peak', None)
        m.setdefault('halted_day', None); m.setdefault('halt_total', False)
        m.setdefault('phase_base', None); m.setdefault('target_hit', False)
        m.setdefault('trading_days', 0); m.setdefault('td_date', None)
        m.setdefault('day_pnls', []); m.setdefault('day_pnl_dates', [])
        m.setdefault('lev', 1.0); m.setdefault('w_prev', {}); m.setdefault('eq_daily', [])
        m.setdefault('mult_daily', []); m.setdefault('mult_day_sum', 0.0); m.setdefault('mult_day_n', 0)
        m.setdefault('flow_daily', []); m.setdefault('flow_today', 0.0)      # cash flows (payouts < 0, deposits > 0) per UTC day

    def note_cash_flow(self, amount):
        """Record a cash flow (payout negative). The daily anchor/peak move with it so a payout is not a 'loss' for the
        daily rule, the trading-day/consistency PnL, or the vol target (NAV return = (eq_t - flow_t)/eq_{t-1} - 1)."""
        m = self.mem; amount = float(amount)
        m['flow_today'] = float(m.get('flow_today', 0.0)) + amount
        if m.get('day_anchor') is not None:
            m['day_anchor'] = float(m['day_anchor']) + amount
        if m.get('day_peak') is not None:
            m['day_peak'] = float(m['day_peak']) + amount
        if C.RULES['total_dd_mode'] == 'static':          # static floors are on the initial balance; a payout does not create a drawdown
            m['eq_peak'] = float(m['eq_peak']) + amount
        m.setdefault('cooldown', {}); m.setdefault('k_last', 0.0); m.setdefault('k_reason', '')

    # ------------------------------------------------------------ firm geometry
    @property
    def initial(self):
        return float(self.mem['initial'])

    def notional_base(self, equity):
        """Reference balance for the firm's notional/margin/per-position limits (see config.NOTIONAL_BASE)."""
        return float(equity) if C.NOTIONAL_BASE == 'equity' else self.initial

    def dd_amount(self):
        return self.initial * C.RULES['total_dd'] / 100.0

    def daily_amount(self):
        return self.initial * C.RULES['daily_dd'] / 100.0

    def total_floor(self):
        base = self.mem['eq_peak'] if C.RULES['total_dd_mode'] == 'trailing' else self.initial
        return float(base) - self.dd_amount()

    def daily_floor(self):
        ref = self.mem['day_peak'] if C.RULES['daily_dd_mode'] == 'trailing' else self.mem['day_anchor']
        if not ref:
            return -1e18
        return float(ref) - self.daily_amount()

    def phase_target(self):
        base = float(C.PHASE_BASE or self.mem.get('phase_base') or self.initial)
        tgt_pct = C.RULES['targets'][min(C.PHASE, len(C.RULES['targets'])) - 1]
        return base * (1 + tgt_pct / 100.0), base * (1 + (tgt_pct + C.TARGET_BUFFER_PCT) / 100.0)

    # ------------------------------------------------------------ marks (called by watchdog and cycle)
    def observe_equity(self, equity, now=None):
        """Update peaks/anchors; returns dict of live risk metrics. Must be called frequently (watchdog)."""
        now = now or datetime.now(timezone.utc)
        today = now.date().isoformat()
        m = self.mem
        if m['day_date'] != today:
            # new UTC day: finalise yesterday's pnl for consistency tracking
            if m['day_anchor'] is not None and m['day_date'] is not None:
                pnl = equity - float(m['day_anchor'])
                m['day_pnls'].append(pnl); m['day_pnl_dates'].append(m['day_date'])
                m['eq_daily'].append(equity)
                m['eq_daily'] = m['eq_daily'][-120:]
                m['flow_daily'].append(float(m.get('flow_today', 0.0))); m['flow_daily'] = m['flow_daily'][-120:]; m['flow_today'] = 0.0
                # exposure multiplier (lev x K) that generated yesterday's return -> de-levered vol for the vol target
                n_ = int(m.get('mult_day_n', 0))
                mult = float(m['mult_day_sum']) / n_ if n_ > 0 else float(m.get('lev', 1.0)) * max(float(m.get('k_last', C.BASE_SCALE)), 0.05)
                m['mult_daily'].append(max(mult, 1e-6)); m['mult_daily'] = m['mult_daily'][-120:]
                m['mult_day_sum'] = 0.0; m['mult_day_n'] = 0
                if abs(pnl) >= 0.0025 * self.initial and m['td_date'] != m['day_date']:
                    m['trading_days'] = int(m['trading_days']) + 1; m['td_date'] = m['day_date']
            m['day_date'] = today; m['day_anchor'] = equity; m['day_peak'] = equity; m['halted_day'] = None
        m['day_peak'] = max(float(m['day_peak'] or equity), equity)
        m['eq_peak'] = max(float(m['eq_peak']), equity)
        tf, df_ = self.total_floor(), self.daily_floor()
        return dict(equity=equity, total_floor=tf, daily_floor=df_, buffer_total=equity - tf, buffer_daily=equity - df_,
                    day_pnl=equity - float(m['day_anchor']), day_dd=(equity - float(m['day_peak'])) if C.RULES['daily_dd_mode'] == 'trailing'
                    else equity - float(m['day_anchor']))

    # ------------------------------------------------------------ risk multiplier
    def multiplier(self, equity, now=None):
        m = self.mem
        now = now or datetime.now(timezone.utc)
        if m['halt_total']:
            return 0.0, 'TOTAL HALT (latched)'
        dd_amt = self.dd_amount()
        buf = equity - self.total_floor()
        if buf <= C.FLOOR_GUARD_FRAC * dd_amt:
            return 0.0, f'floor guard: buffer {buf:,.0f} <= {C.FLOOR_GUARD_FRAC * 100:.0f}% of allowance'
        base = C.BASE_SCALE
        if C.MODE == 'challenge' and C.PHASE >= 2 and C.BASE_SCALE_P2 > 0:
            base = C.BASE_SCALE_P2
        kmax = max(C.K_MAX, base) if C.K_MAX > 0 else base
        k = base * min(kmax / base, buf / (C.CPPI_FRAC * dd_amt)) if C.CPPI_FRAC > 0 else base
        reason = f'K {k / base:.2f}x base (buffer {buf / dd_amt * 100:.0f}% of allowance)'
        today = now.date().isoformat()
        if m['halted_day'] == today:
            return 0.0, 'daily halt (flat until 00:00 UTC)'
        ref = float(m['day_peak']) if C.RULES['daily_dd_mode'] == 'trailing' else float(m['day_anchor'] or equity)
        day_loss = (equity - ref) / self.initial
        if day_loss <= -C.DAILY_HALT_FRAC * C.RULES['daily_dd'] / 100.0:
            return 0.0, f'daily halt trigger: {day_loss * 100:.2f}% of initial'
        if day_loss <= -C.DAILY_THROTTLE_PCT / 100.0:
            k *= 0.5; reason += f' + daily throttle ({day_loss * 100:.2f}%)'
        if k > base * 0.999 and C.MODE == 'challenge':
            reason += ' [convex]' if kmax > base else ''
        if C.MODE == 'challenge':
            tgt, trig = self.phase_target()
            min_days = C.RULES['min_days'][min(C.PHASE, len(C.RULES['min_days'])) - 1]
            if equity >= tgt and int(m['trading_days']) < min_days:
                k *= C.RISK_AFTER_TARGET; reason += ' + target reached, collecting min days'
            if C.RULES['consistency'] > 0 and equity >= tgt and not self.consistency_ok():
                k *= 0.5; reason += ' + consistency rule not yet satisfied'
        m['k_last'] = k; m['k_reason'] = reason
        return k, reason

    def consistency_ok(self):
        dp = list(self.mem['day_pnls'])
        if not dp:
            return True
        tot = sum(dp)
        if tot <= 0:
            return True
        return max(dp) <= C.RULES['consistency'] / 100.0 * tot

    # ------------------------------------------------------------ vol targeting
    def update_lev(self, exante_dvol=None):
        """DIRECT vol target: lev = target / realised daily vol of the UNLEVERED book, where each day's equity return
        is de-levered by the exposure multiplier (lev x K) that was in force that day (mult_daily). The former
        multiplicative form (lev *= target/realised) fed levered returns through a lagging window and oscillated
        between the clips (research exp20g: lev at 0.05 35% of the time, at max 20%). Optional EMA via VT_SMOOTH.
        Cold start (< 10 days): ex-ante estimate from panel if provided."""
        m = self.mem
        eqs = np.array(m['eq_daily'][-(C.VT_WIN_D + 1):], dtype=float)
        if len(eqs) >= 11:
            flows = np.array((m.get('flow_daily') or [])[-(C.VT_WIN_D + 1):], dtype=float)
            flows = flows[-(len(eqs) - 1):] if len(flows) >= len(eqs) - 1 else np.concatenate([np.zeros(len(eqs) - 1 - len(flows)), flows])
            r = (eqs[1:] - flows) / eqs[:-1] - 1.0          # NAV return: payouts/deposits are not performance
            mults = np.array(m.get('mult_daily', [])[-(C.VT_WIN_D + 1):], dtype=float)
            if len(mults) >= len(r):
                mults = mults[-len(r):]
            else:                                     # legacy memory without multipliers: fall back to k_avg x lev
                mults = np.full(len(r), max(float(m.get('k_avg', C.BASE_SCALE)), 0.05) * max(float(m.get('lev', 1.0)), 0.05))
            r_u = r / np.maximum(mults, 1e-6)
            realised_unscaled = float(np.std(r_u, ddof=1)); realised = float(np.std(r, ddof=1))
            if realised_unscaled > 1e-6:
                lev_new = float(np.clip(C.VT_TARGET_DVOL / realised_unscaled, 0.05, C.VT_MAX_LEV))
                m['lev'] = lev_new if C.VT_SMOOTH <= 0 else float(C.VT_SMOOTH * float(m['lev']) + (1 - C.VT_SMOOTH) * lev_new)
                return m['lev'], f'VT realised {realised * 100:.2f}%/d (unlevered {realised_unscaled * 100:.2f}%)'
        if exante_dvol and exante_dvol > 1e-6:
            m['lev'] = float(np.clip(C.VT_TARGET_DVOL / exante_dvol, 0.05, C.VT_MAX_LEV))
            return m['lev'], f'VT ex-ante {exante_dvol * 100:.2f}%/d'
        return m['lev'], 'VT unchanged'

    @staticmethod
    def exante_dvol(w, panel, days=30):
        """Daily vol of the current weight vector applied to the last `days` of hourly returns."""
        cff = panel['cff']; H = cff.shape[0]
        n = min(days * 24, H - 1)
        r = cff[H - n:] / cff[H - n - 1:-1] - 1.0
        r = np.where(np.isfinite(r), r, 0.0)
        pr = r @ w
        d = pr.reshape(-1, 24).sum(1) if len(pr) % 24 == 0 else pr[len(pr) % 24:].reshape(-1, 24).sum(1)
        return float(np.std(d, ddof=1)) if len(d) > 5 else None

    # ------------------------------------------------------------ targets
    def build_targets(self, w_raw, symbols, equity, t24_now, k, majors, lowcap_mask=None):
        """w_raw: combined sleeve weights (N,). Returns target notionals (N,) in USDT after smoothing,
        leverage, risk multiplier, caps, neutrality, gross/margin/low-cap guards and dust rule.
        lowcap_mask: bool (N,) — names the firm counts as low-cap (market cap below its threshold); when None, a
        turnover proxy (RULES.lowcap_turn) is used. Their combined gross is capped at RULES.lowcap_pct % of the notional base."""
        m = self.mem
        N = len(symbols)
        # EMA smoothing of weights (state keyed by symbol)
        w = np.array(w_raw, dtype=float)
        if C.SMOOTH > 0 and m['w_prev']:
            prev = np.array([m['w_prev'].get(s, 0.0) for s in symbols])
            w = C.SMOOTH * prev + (1 - C.SMOOTH) * w
        m['w_prev'] = {s: float(v) for s, v in zip(symbols, w) if abs(v) > 1e-6}
        lev = float(m['lev'])
        tgt = w * lev * k * equity
        is_major = np.array([s in majors for s in symbols])
        capl = np.where(is_major, C.MAJOR_CAP_PCT, C.POS_CAP_PCT) / 100.0 * equity
        caps = np.where(is_major, C.MAJOR_CAP_PCT, C.SHORT_CAP_PCT) / 100.0 * equity
        nb = self.notional_base(equity)
        firm_pos_cap = C.RULES['position_x'] * nb * C.SAFETY_MARGIN
        capl = np.minimum(capl, firm_pos_cap); caps = np.minimum(caps, firm_pos_cap)
        tgt = np.where(tgt > 0, np.minimum(tgt, capl), np.maximum(tgt, -caps))
        # low-cap exposure guard (HyroTrader: 'low-cap altcoins (market cap < $100M) <= 5% of balance'): the combined gross of
        # low-cap names is scaled down to RULES.lowcap_pct % of the notional base (with a safety margin); the rest of the
        # book is untouched. Market caps (CoinGecko) when available, else the turnover proxy; unknown names count as low-cap.
        lowcap_pct = float(C.RULES.get('lowcap_pct', 100.0))
        if lowcap_pct < 100.0:
            low = np.asarray(lowcap_mask, bool) if lowcap_mask is not None else (np.isfinite(t24_now) & (t24_now < C.RULES['lowcap_turn']))
            g_low = float(np.abs(tgt[low]).sum()); g_low_max = lowcap_pct / 100.0 * nb * C.SAFETY_MARGIN
            if g_low > g_low_max > 0:
                tgt[low] *= g_low_max / g_low
            elif g_low > 0 and g_low_max <= 0:
                tgt[low] = 0.0
            self.last_lowcap = dict(n=int(low.sum()), gross=round(g_low, 0), cap=round(g_low_max, 0))
        # dollar neutrality after caps
        ls, ss = tgt[tgt > 0].sum(), -tgt[tgt < 0].sum()
        if ls > 0 and ss > 0:
            mm = min(ls, ss); tgt[tgt > 0] *= mm / ls; tgt[tgt < 0] *= mm / ss
        # gross caps: equity multiple, firm notional (of INITIAL), firm margin (gross / leverage)
        g = float(np.abs(tgt).sum())
        g_max = min(C.MAX_GROSS_X_EQ * equity, C.RULES['notional_x'] * nb * C.SAFETY_MARGIN,
                    C.RULES['margin_pct'] / 100.0 * nb * C.SAFETY_MARGIN * C.SET_LEVERAGE)
        if g > g_max > 0:
            tgt *= g_max / g
        tgt = np.where(np.abs(tgt) < C.MIN_TRADE_USDT, 0.0, tgt)
        info = dict(lev=round(lev, 3), k=round(k, 3), gross=round(float(np.abs(tgt).sum()), 0),
                    gross_x_initial=round(float(np.abs(tgt).sum()) / self.initial, 3),
                    n_long=int((tgt > 0).sum()), n_short=int((tgt < 0).sum()), g_max=round(g_max, 0), lowcap=getattr(self, 'last_lowcap', None))
        return tgt, info
