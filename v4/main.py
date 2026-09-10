#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PROP-SLEEVES v4 — multi-sleeve market-neutral portfolio bot for prop-firm accounts on Bybit (demo/live).

Sleeves: listing-drift short vs majors, cross-sectional trend quality (core336), ML rankers (3d/7d).
Risk: firm presets (static/trailing floors), CPPI, daily throttle/halt, vol targeting, caps, guards, stops.
Cycle: daily rebalance at REBAL_HOUR_UTC:REBAL_MINUTE (after the 00:00 funding + bar close); hourly
scale pass (proportional de-risking when K drops); watchdog every WATCHDOG_SEC.
"""
import os, sys, time, json, threading, traceback
from datetime import datetime, timezone, timedelta
from http.server import BaseHTTPRequestHandler, HTTPServer
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
from bybit import Bybit, ApiError
from data import MarketData, NEED_H
from signals import Sleeves
from risk import RiskEngine
from execution import Executor, round_qty, fmt_price
from state import Memory, log, tg, tg_once, set_status, get_status


class Bot:
    def __init__(self, api, md=None, sleeves=None, mem=None):
        self.api = api
        self.mem = mem or Memory()
        self.md = md or MarketData(api, log, C.USE_BINANCE_CLOCK)
        self.sleeves = sleeves or Sleeves(log)
        self.risk = RiskEngine(self.mem, log)
        self.ex = Executor(api, log, tg, C.DRY_RUN)
        self.instr = {}
        self.lock = threading.Lock()
        self.reb_every_base = int(C.REBAL_EVERY_H)         # cadence with ml8 on; the kill rule switches to daily when ml8 is off

    # ------------------------------------------------------------ helpers
    def now(self):
        return datetime.now(timezone.utc)

    def equity(self):
        return self.api.equity()

    def recover_day_anchor(self, equity):
        """State lost mid-day: anchor = equity - realised change today (tx log); conservative floor guard."""
        d = self.now().date()
        day_start = datetime(d.year, d.month, d.day, tzinfo=timezone.utc)
        try:
            anchor = equity - self.api.transaction_log_sum(int(day_start.timestamp() * 1000))
            if anchor < equity * 0.99:
                anchor = equity
            return anchor, 'txlog'
        except Exception as e:
            log(f'txlog anchor fail: {e}')
            return equity, 'fallback'

    # ------------------------------------------------------------ guards
    def guards(self, equity):
        """Returns True if trading may continue this cycle."""
        m = self.mem
        if C.KILL_SWITCH:
            self.ex.flatten_all(self.instr, 'KILL_SWITCH'); return False
        if m['halt_total']:
            return False
        met = self.risk.observe_equity(equity, self.now())
        if met['buffer_total'] <= C.FLOOR_GUARD_FRAC * self.risk.dd_amount():
            m['halt_total'] = True; m.save()
            self.ex.flatten_all(self.instr, f'FLOOR GUARD: equity {equity:,.0f} within {C.FLOOR_GUARD_FRAC * 100:.0f}% of allowance to floor {met["total_floor"]:,.0f}')
            tg(f'🛑 ОБЩИЙ СТОП: equity {equity:,.2f}, пол фирмы {met["total_floor"]:,.2f}. Торговля остановлена (латч). Сброс: RESET_TOTAL_STOP=1')
            return False
        if C.MODE == 'challenge' and C.ABANDON_FRAC > 0 and met['buffer_total'] <= C.ABANDON_FRAC * self.risk.dd_amount():
            m['halt_total'] = True; m['abandoned'] = True; m.save()
            self.ex.flatten_all(self.instr, f'ABANDON: buffer {met["buffer_total"]:,.0f} <= {C.ABANDON_FRAC * 100:.0f}% of allowance')
            tg(f'🏳️ Челлендж сдан по правилу перезапуска: equity {equity:,.2f}, буфер до пола {met["buffer_total"]:,.0f} '
               f'(≤{C.ABANDON_FRAC * 100:.0f}% допуска). Быстрее купить новый челлендж, чем выползать. Бот остановлен (латч).')
            return False
        today = self.now().date().isoformat()
        if m['halted_day'] == today:
            return False
        day_loss = met['day_dd'] / self.risk.initial
        if day_loss <= -C.DAILY_HALT_FRAC * C.RULES['daily_dd'] / 100.0:
            m['halted_day'] = today; m.save()
            self.ex.flatten_all(self.instr, f'DAILY HALT: day drawdown {day_loss * 100:.2f}% of initial (firm limit {C.RULES["daily_dd"]}%)')
            tg(f'🛑 ДНЕВНОЙ СТОП {day_loss * 100:.2f}% (лимит фирмы −{C.RULES["daily_dd"]}%). Флэт до 00:00 UTC.')
            return False
        if C.MODE == 'challenge' and not m['target_hit']:
            tgt, trig = self.risk.phase_target()
            min_days = C.RULES['min_days'][min(C.PHASE, len(C.RULES['min_days'])) - 1]
            if equity >= trig and int(m['trading_days']) >= min_days and (C.RULES['consistency'] <= 0 or self.risk.consistency_ok()):
                n = self.ex.flatten_all(self.instr, f'PHASE TARGET {tgt:,.0f} reached')
                time.sleep(15)
                try:
                    eq2 = self.api.equity()
                except Exception:
                    eq2 = equity
                if eq2 >= tgt or n == 0:
                    m['target_hit'] = True; m.save()
                    tg(f'🎉 ФАЗА {C.PHASE} ПРОЙДЕНА: equity после закрытия {eq2:,.2f} ≥ цель {tgt:,.2f}. Торговых дней {m["trading_days"]}. '
                       f'Бот остановлен. Дальше: подтверждение фирмы → PHASE={C.PHASE + 1} или MODE=funded.')
                else:
                    tg(f'⚠️ Комиссии закрытия откусили проход ({eq2:,.2f} < {tgt:,.2f}) — продолжаю торговать.')
                return False
        if m['target_hit']:
            return False
        return True

    # ------------------------------------------------------------ daily rebalance
    # ------------------------------------------------------------ per-sleeve attribution & ml8 kill rule (PREREG.md rule 1)
    def attrib_settle(self, panel, syms, equity, now_):
        """Split the PnL since the last rebalance (notional x price return, from the panel) across sleeves by the signed
        weight shares recorded then; roll the daily buckets into history at each new UTC day."""
        m = self.mem; today = now_.date().isoformat()
        m.setdefault('sleeve_pnl_day', {}); m.setdefault('sleeve_pnl_hist', []); m.setdefault('attrib_day', today)
        if m['attrib_day'] != today:
            m['sleeve_pnl_hist'].append({'date': m['attrib_day'], 'eq': float(equity), **{k: float(v) for k, v in m['sleeve_pnl_day'].items()}})
            m['sleeve_pnl_hist'] = m['sleeve_pnl_hist'][-400:]; m['sleeve_pnl_day'] = {}; m['attrib_day'] = today
        share, px_prev, notion = m.get('attrib_share') or {}, m.get('attrib_px') or {}, m.get('attrib_notional') or {}
        if not share:
            return
        i = panel['cff'].shape[0] - 1; pos_now = {s: j for j, s in enumerate(syms)}
        for s, sh_ in share.items():
            j = pos_now.get(s); p0 = px_prev.get(s)
            if j is None or not p0 or p0 <= 0:
                continue
            p1 = float(panel['cff'][i][j])
            if not np.isfinite(p1) or p1 <= 0:
                continue
            ret = p1 / p0 - 1.0
            for k, expo in sh_.items():                # expo = signed $ exposure attributed to sleeve k (sums to the position)
                m['sleeve_pnl_day'][k] = float(m['sleeve_pnl_day'].get(k, 0.0)) + float(expo) * ret
        # shadow ml8: the sleeve at its full pre-registered weight, whatever multiplier is in force -> the kill rule keeps
        # measuring the sleeve after it was scaled down or switched off, so it can be restored (Opus round 3)
        for s, expo in (m.get('attrib_shadow_ml8') or {}).items():
            j = pos_now.get(s); p0 = px_prev.get(s)
            if j is None or not p0 or p0 <= 0:
                continue
            p1 = float(panel['cff'][i][j])
            if np.isfinite(p1) and p1 > 0:
                m['sleeve_pnl_day']['ml8_shadow'] = float(m['sleeve_pnl_day'].get('ml8_shadow', 0.0)) + float(expo) * (p1 / p0 - 1.0)

    def attrib_summary(self, equity, days=90):
        hist = self.mem.get('sleeve_pnl_hist') or []
        if not hist:
            return {}
        recent = hist[-days:]; out = {}
        for k in set().union(*[set(h.keys()) - {'date', 'eq'} for h in recent]):
            r = np.array([h.get(k, 0.0) / max(h['eq'], 1e-9) for h in recent])
            out[k] = dict(pnl_pct=round(float(r.sum()) * 100, 2), sharpe=round(float(r.mean() / r.std(ddof=1) * np.sqrt(365)), 2) if len(r) > 10 and r.std(ddof=1) > 0 else None, days=len(r))
        return out

    def apply_ml8_kill_rule(self, equity, now_):
        """Pre-registered: trailing-window Sharpe of the ml8 contribution <= 0 -> multiplier 1.0 -> 0.5; <= HARD -> 0.0;
        restore one step after RESTORE_D days of contribution Sharpe > 0.5; at most one change per MIN_GAP_D days."""
        m = self.mem; hist = m.get('sleeve_pnl_hist') or []
        mult = float(m.get('ml8_mult', 1.0)); last = m.get('ml8_mult_date')
        if len(hist) < C.ML8_KILL_WIN_D:
            C.SLEEVE_SCALES['ml8'] = C.ML8_BASE_SCALE * mult; return
        win = hist[-C.ML8_KILL_WIN_D:]
        shadow = all('ml8_shadow' in h for h in win)        # shadow = full-weight sleeve (measurable when off), net of its own costs
        if shadow:
            r = np.array([(h.get('ml8_shadow', 0.0) - h.get('ml8_shadow_cost', 0.0)) / max(h['eq'], 1e-9) for h in win])
        else:
            r = np.array([h.get('ml8', 0.0) / max(h['eq'], 1e-9) for h in win])
        s = float(r.mean() / r.std(ddof=1) * np.sqrt(365)) if r.std(ddof=1) > 0 else 0.0
        if m.get('ml8_audit_day') != now_.date().isoformat() and now_.day == 1:      # monthly audit (state + log; the harness prints it)
            g = np.array([h.get('ml8_shadow', 0.0) for h in win]); c = np.array([h.get('ml8_shadow_cost', 0.0) for h in win])
            m['ml8_audit_day'] = now_.date().isoformat()
            m.setdefault('ml8_audit', []).append((now_.date().isoformat(), round(s, 2), round(float(c.sum() / max(abs(g.sum()), 1e-9)), 2), mult))
            m['ml8_audit'] = m['ml8_audit'][-72:]
            log(f'ML8 KILL RULE check: net Sharpe {s:.2f} over {len(win)}d | shadow gross {g.sum():,.0f} cost {c.sum():,.0f} '
                f'(cost/gross {c.sum() / max(abs(g.sum()), 1e-9):.2f}) | mult {mult} | measure {"net shadow" if shadow else "marginal"}')
        gap_ok = (last is None) or ((now_.date() - datetime.fromisoformat(last).date()).days >= C.ML8_KILL_MIN_GAP_D)
        new = mult
        if gap_ok:
            # v2 thresholds (PREREG rule 1): net Sharpe <= HARD (0) -> off; <= SOFT (0.5) -> half (second time -> off);
            # restore one step after RESTORE_D days with net Sharpe > RESTORE_SHARPE (1.0)
            if s <= C.ML8_KILL_HARD or (s <= C.ML8_KILL_SOFT and mult <= 0.5):
                new = 0.0
            elif s <= C.ML8_KILL_SOFT and mult > 0.5:
                new = 0.5
            elif mult < 1.0 and s > C.ML8_KILL_RESTORE_SHARPE and last is not None and (now_.date() - datetime.fromisoformat(last).date()).days >= C.ML8_KILL_RESTORE_D:
                new = min(1.0, mult + 0.5)
        if new != mult:
            m['ml8_mult'] = new; m['ml8_mult_date'] = now_.date().isoformat()
            m.setdefault('ml8_changes', []).append((now_.date().isoformat(), mult, new, round(s, 2))); m['ml8_changes'] = m['ml8_changes'][-50:]
            log(f'ML8 KILL RULE: {"net shadow" if shadow else "marginal"} contribution Sharpe {s:.2f} over {C.ML8_KILL_WIN_D}d -> ml8 multiplier {mult} -> {new}')
            tg(f'⚙️ Правило отключения ml8: Sharpe вклада за {C.ML8_KILL_WIN_D} дн = {s:.2f} → множитель {mult} → {new} (вес рукава {C.ML8_BASE_SCALE * new:.1f})')
            mult = new
        C.SLEEVE_SCALES['ml8'] = C.ML8_BASE_SCALE * mult
        # cadence follows the sleeve: without ml8 there is nothing to trade intraday -> daily rebalances (fcarry becomes daily)
        want_every = 24 if mult == 0.0 else self.reb_every_base
        if int(C.REBAL_EVERY_H) != want_every:
            log(f'ML8 KILL RULE: rebalance cadence {C.REBAL_EVERY_H}h -> {want_every}h'); C.REBAL_EVERY_H = want_every
            tg(f'⚙️ Каденция ребаланса: {want_every}ч (ml8 {"выключен" if mult == 0 else "включён"})')

    def rebalance(self, slot=None):
        with self.lock:
            t0 = time.time()
            equity = self.equity()
            m = self.mem
            if C.MODE == 'challenge' and not m.get('phase_base'):
                m['phase_base'] = float(C.ACCOUNT_SIZE) if C.PHASE == 1 else float(equity)
            if not self.guards(equity):
                log('rebalance skipped by guards'); return
            k, why = self.risk.multiplier(equity, self.now())
            if k <= 0:
                self.ex.flatten_all(self.instr, f'K=0: {why}'); return
            # ---- data
            self.md.refresh_instruments()
            self.instr = self.md.instr
            tickers = self.api.tickers()
            positions = self.api.positions()
            held = set(positions)
            now_ms = int(self.now().timestamp() * 1000)
            cands = [s for s, t in tickers.items() if s in self.instr and s.endswith('USDT') and s not in C.BLACKLIST
                     and not any(p in s for p in C.BLACKLIST_PATTERNS) and t['turnover24h'] >= min(C.UNI_MIN_TURN, C.LIST_MIN_TURN) * 0.8]
            cands.sort(key=lambda s: -tickers[s]['turnover24h'])
            # candidate pool: top-N by turnover for core/ML + ALL young liquid names for the listing sleeve + held
            young = [s for s in cands if tickers[s]['turnover24h'] >= C.LIST_MIN_TURN
                     and 0 <= self.md.age_hours(s, now_ms, 'listing') <= C.LIST_AGE_MAX_D * 24]
            symbols = list(dict.fromkeys(C.MAJORS + cands[:max(C.UNI_TOP_ML + 60, C.CAND_POOL)] + young + sorted(held)))
            last_closed = self.md.last_closed_hour_ms(now_ms)
            ok, bad = self.md.refresh(symbols, last_closed)
            panel = self.md.panel([s for s in symbols if s in self.md.bars], last_closed)
            panel['delist'] = self.md.delist
            syms = panel['symbols']
            age_h = np.array([self.md.age_hours(s, now_ms, 'listing') for s in syms])       # earliest listing
            age_ml = np.array([self.md.age_hours(s, now_ms, 'ml') for s in syms])            # Binance-clock (training)
            now_ = self.now()
            # ---- per-sleeve attribution & ml8 kill rule BEFORE the signals, so a multiplier/cadence change applies this cycle
            if C.ATTRIB_SLEEVES:
                self.attrib_settle(panel, syms, equity, now_)
                if C.ML8_KILL_RULE and 'ml8' in C.SLEEVES:
                    self.apply_ml8_kill_rule(equity, now_)
            # ---- signals (daily sleeves recomputed at the daily hour, held otherwise; ml8/fcarry every rebalance)
            daily_update = (C.REBAL_EVERY_H >= 24) or (now_.hour == C.REBAL_HOUR_UTC) or (m.get('last_daily_date') != now_.date().isoformat() and now_.hour > C.REBAL_HOUR_UTC)
            panel_daily = None
            if daily_update and C.REBAL_EVERY_H < 24 and now_.hour != C.REBAL_HOUR_UTC:
                # late catch-up (the 01:10 cycle was missed): daily sleeves on the 00:00 bar, as in research, not on the current bar
                day0_ms = int(datetime(now_.year, now_.month, now_.day, tzinfo=timezone.utc).timestamp() * 1000)
                if last_closed > day0_ms:
                    try:
                        panel_daily = self.md.panel([s for s in symbols if s in self.md.bars], day0_ms); panel_daily['delist'] = self.md.delist
                    except Exception as e:
                        log(f'daily panel on the 00:00 bar failed ({e}); using the current bar')
            w, parts, info = self.sleeves.combined(panel, age_h, held, age_ml=age_ml, daily_update=daily_update, hour_utc=now_.hour, panel_daily=panel_daily)
            # fail-safe: a signal-bearing sleeve that produced NO weights means broken data/model/alignment -> alert and
            # do not trade this cycle (keep current positions) instead of silently trading the remainder of the book.
            empty = [s for s in C.SLEEVES if s in parts and not np.any(parts[s] != 0.0)]
            critical = [s for s in empty if s != 'listing']          # the listing sleeve is legitimately empty in quiet periods
            if empty:
                tg_once(f'empty_sleeve_{"_".join(empty)}', f'⚠️ Рукав(а) без целей: {", ".join(empty)} | uni {info}', hours=6.0)
            if critical:
                log(f'REBALANCE ABORTED: sleeve(s) without weights {critical} | info {info}')
                set_status(ok=False, error=f'empty sleeves {critical}', sleeves=info, last_cycle_utc=self.now().isoformat())
                return
            if daily_update:
                m['last_daily_date'] = now_.date().isoformat()
                if C.FUNDING_CHECK:
                    self.funding_check(panel, syms, now_)
            m['held_w'] = self.sleeves.held
            # ---- masks BEFORE neutralisation / caps (Opus: masking after build_targets broke the neutrality of the rest):
            # names in stop cooldown, with stale or bad data get weight 0 here; their open positions are closed by the planner
            now_ts = self.now().timestamp()
            for s, until in list(m['cooldown'].items()):
                if until < now_ts:
                    m['cooldown'].pop(s, None)
            blocked = set(m['cooldown']) | {s for s, st in zip(syms, panel['stale']) if st} | set(bad)
            if blocked:
                jb = [j for j, s in enumerate(syms) if s in blocked]
                w = w.copy(); w[jb] = 0.0; info['blocked_n'] = len(jb)
            # ---- vol target & targets
            exante = self.risk.exante_dvol(w, panel) if len(m['eq_daily']) < 11 else None
            lev, vt_why = self.risk.update_lev(exante)
            i = panel['cff'].shape[0] - 1
            lowcap_mask = None
            if float(C.RULES.get('lowcap_pct', 100.0)) < 100.0:
                # firm's low-cap set = market cap below threshold OR thin 24h turnover; without market caps (harness, API down)
                # every name younger than LOWCAP_YOUNG_D days also counts as low-cap (the listing sleeve's names) — conservative
                t24_now = np.asarray(panel['t24'][i], float)
                lowcap_mask = np.isfinite(t24_now) & (t24_now < C.RULES['lowcap_turn'])
                mc = self.md.refresh_mcap() if (C.USE_MCAP and hasattr(self.md, 'refresh_mcap')) else None
                if mc:
                    lowcap_mask |= np.array([mc.get(MarketData.base_symbol(s), 0.0) < C.LOWCAP_MCAP_USD for s in syms])
                else:
                    lowcap_mask |= np.asarray(age_h, float) < C.LOWCAP_YOUNG_D * 24.0
                lowcap_mask &= np.array([s not in C.MAJORS for s in syms])
                info['lowcap_n'] = int(lowcap_mask.sum()); info['lowcap_src'] = 'mcap+turnover' if mc else 'turnover+age proxy'
            tgt, tinfo = self.risk.build_targets(w, syms, equity, panel['t24'][i], k, set(C.MAJORS), lowcap_mask=lowcap_mask)
            # exposure multiplier actually in force = realised gross / (equity x gross of raw weights) — not lev*K, because
            # caps / neutrality / firm limits may have cut the book (Opus review); used by the direct vol target
            mult_real = tinfo['gross'] / max(equity * max(info.get('gross_w', 0.0), 1e-9), 1e-9) if info.get('gross_w', 0.0) > 1e-9 else lev * k
            m['mult_day_sum'] = float(m.get('mult_day_sum', 0.0)) + mult_real; m['mult_day_n'] = int(m.get('mult_day_n', 0)) + 1
            m['clip_n'] = int(m.get('clip_n', 0)) + int(tinfo['gross'] >= tinfo['g_max'] - 1.0); m['reb_n'] = int(m.get('reb_n', 0)) + 1
            targets = {s: float(v) for s, v in zip(syms, tgt) if v != 0.0}
            if C.ATTRIB_SLEEVES:
                # marginal attribution: each sleeve is credited with its own SIGNED exposure scaled to the actual position
                # (kappa = target$ / sum of scaled sleeve weights), so offsetting sleeves get opposite-sign PnL and the
                # contributions add up to the position's PnL (internal crossing accounted, unlike |share| normalisation)
                scales = {k: C.SLEEVE_SCALES.get(k, 2.0 if k == 'ml8' else 1.0) for k in parts}
                share = {}
                for j, s in enumerate(syms):
                    if s not in targets:
                        continue
                    contrib = {k: float(v[j]) * scales[k] for k, v in parts.items() if v[j] != 0.0}
                    tot = sum(contrib.values())
                    if abs(tot) > 1e-12:
                        kappa = targets[s] / tot                       # $ per unit of scaled weight actually held
                        share[s] = {k: kappa * x for k, x in contrib.items()}   # signed $ exposure attributed to each sleeve
                m['attrib_share'] = share; m['attrib_notional'] = dict(targets)
                # shadow ml8 at its full pre-registered weight (kill rule measures the sleeve even when scaled down / off)
                shadow = {}
                if 'ml8' in parts:
                    tot_full = sum(scales[k] for k in parts if k != 'ml8') + C.ML8_BASE_SCALE
                    kappa_book = sum(abs(v) for v in targets.values()) / max(float(np.abs(w).sum()), 1e-9)   # $ per unit of book weight
                    shadow = {s: kappa_book * C.ML8_BASE_SCALE * float(parts['ml8'][j]) / tot_full for j, s in enumerate(syms) if parts['ml8'][j] != 0.0}
                    # estimated trading cost of the shadow sleeve (turnover x blended bps) -> the kill rule measures NET contribution
                    prev_sh = m.get('attrib_shadow_ml8') or {}
                    turn = sum(abs(shadow.get(s_, 0.0) - prev_sh.get(s_, 0.0)) for s_ in set(shadow) | set(prev_sh))
                    m.setdefault('sleeve_pnl_day', {})
                    m['sleeve_pnl_day']['ml8_shadow_cost'] = float(m['sleeve_pnl_day'].get('ml8_shadow_cost', 0.0)) + turn * C.ML8_COST_BPS_EST / 1e4
                m['attrib_shadow_ml8'] = shadow
                m['attrib_px'] = {s: float(panel['cff'][i][j]) for j, s in enumerate(syms) if (s in targets or s in shadow) and np.isfinite(panel['cff'][i][j])}
            for s in blocked:
                targets.pop(s, None)                       # weight was zero already; open positions in these names get closed
            # ---- execution
            legs = self.ex.plan(targets, positions, tickers, self.instr, C.REBAL_BAND, C.MIN_TRADE_USDT)
            stats, errors = self.ex.execute(legs, self.instr, tickers, self.api.positions)
            # ---- bookkeeping
            m['last_targets'] = targets; m['k_ref'] = k; m['last_rebal_date'] = self.now().date().isoformat()
            m['last_rebal_slot'] = slot or f'{self.now().date().isoformat()}T{self.now().hour:02d}'
            m['k_avg'] = 0.8 * float(m.get('k_avg', k)) + 0.2 * k
            m['cycles'] = int(m.get('cycles', 0)) + 1
            m.save()
            poss2 = self.api.positions()
            if C.REQUIRE_SL:
                self.ensure_stops(poss2)
            g = sum(abs(p['qty'] * p['mark']) for p in poss2.values()); net = sum(p['qty'] * p['mark'] for p in poss2.values())
            set_status(equity=round(equity, 2), k=round(k, 3), k_reason=why, lev=round(lev, 3), vt=vt_why, mult_real=round(mult_real, 3),
                       clip_share=round(m['clip_n'] / max(m['reb_n'], 1), 3), sleeves=info, targets=tinfo, sleeve_attrib=self.attrib_summary(equity),
                       ml8_mult=m.get('ml8_mult', 1.0),
                       n_targets=len(targets), n_positions=len(poss2), gross=round(g, 0), net=round(net, 0), gross_x_initial=round(g / self.risk.initial, 3),
                       orders=stats, errors=[e[2] for e in errors][:10], data_bad=bad[:20], cycles=m['cycles'],
                       last_cycle_utc=self.now().isoformat(), total_floor=round(self.risk.total_floor(), 2), daily_floor=round(self.risk.daily_floor(), 2),
                       trading_days=m['trading_days'], day_pnls_n=len(m['day_pnls']), consistency_ok=self.risk.consistency_ok(), ok=True)
            log(f'REBALANCE done in {time.time() - t0:.0f}s | eq {equity:,.2f} K={k:.2f} ({why}) lev {lev:.2f} ({vt_why}) | '
                f'targets {len(targets)} gross {tinfo["gross"]:,.0f} ({tinfo["gross_x_initial"]:.2f}x) L{tinfo["n_long"]}/S{tinfo["n_short"]} | '
                f'orders maker {stats["maker_sent"]} market {stats["market_sent"]} errors {len(errors)} | positions {len(poss2)} gross {g:,.0f} net {net:+,.0f}')
            tg(f'📊 Ребаланс: equity {equity:,.2f} | K {k:.2f} ({why}) | lev {lev:.2f} | целей {len(targets)} (L{tinfo["n_long"]}/S{tinfo["n_short"]}) '
               f'gross {tinfo["gross_x_initial"]:.2f}x | позиций {len(poss2)}, net {net / equity * 100:+.1f}% | ордера maker/market {stats["maker_sent"]}/{stats["market_sent"]}'
               + (f' | ошибок {len(errors)}' if errors else ''))

    # ------------------------------------------------------------ funding diagnostics (PREREG rule 6)
    def funding_check(self, panel, syms, now_):
        """Once a day: (1) funding settlements actually booked on THIS account in the last 24h (transaction log) and since the
        start — if none after FUNDING_ALERT_D days the account does not accrue funding and the carry book must be replaced by F;
        (2) Bybit-vs-Binance 24h funding on the 20 largest names — the basis between the trading exchange and the research one."""
        m = self.mem
        if not hasattr(self.api, 'funding_since'):
            return                                        # simulation (harness FakeApi): nothing to check
        try:
            day_ms = int((now_.timestamp() - 24 * 3600) * 1000)
            f24, n24 = self.api.funding_since(day_ms)
            m['fund_days'] = int(m.get('fund_days', 0)) + 1
            m['fund_settlements'] = int(m.get('fund_settlements', 0)) + n24
            m['fund_total'] = float(m.get('fund_total', 0.0)) + f24
            msg = f'💱 Фандинг за сутки: {f24:+.2f} USDT ({n24} расчётов); с начала: {m["fund_total"]:+.2f} USDT за {m["fund_days"]} дн.'
            if m['fund_days'] >= C.FUNDING_ALERT_D and m['fund_settlements'] == 0:
                msg += f'\n🛑 За {m["fund_days"]} дней ни одного расчёта фандинга: счёт НЕ начисляет фандинг → книга F2 здесь не тестируется, ' \
                       f'переключиться на книгу F (SLEEVES без fchg,fcarry; см. PREREG правило 6).'
                tg_once('no_funding', msg, 24.0)
            elif m['fund_days'] <= C.FUNDING_ALERT_D or now_.weekday() == 0:
                tg(msg)
            set_status(funding_24h=round(f24, 2), funding_settlements_24h=n24, funding_days=m['fund_days'], funding_total=round(m['fund_total'], 2))
        except Exception as e:
            log(f'funding check (transaction log) failed: {e}')
        try:
            i = panel['cff'].shape[0] - 1; t24 = np.asarray(panel['t24'][i], float)
            top = [s for s in [syms[j] for j in np.argsort(-np.where(np.isfinite(t24), t24, -1))] if s.endswith('USDT')][:20]
            bn = self.md.binance_funding_24h(top) if hasattr(self.md, 'binance_funding_24h') else {}
            if bn:
                fund = np.asarray(panel['fund'][-24:], float); by = {s: float(np.nansum(fund[:, syms.index(s)])) for s in bn}
                a = np.array([by[s] for s in bn]); b = np.array([bn[s] for s in bn])
                corr = float(np.corrcoef(a, b)[0, 1]) if len(a) > 3 and a.std() > 0 and b.std() > 0 else float('nan')
                basis = dict(n=len(bn), corr=round(corr, 2), mean_bybit_bps=round(a.mean() * 1e4, 2), mean_binance_bps=round(b.mean() * 1e4, 2),
                             mean_abs_diff_bps=round(float(np.abs(a - b).mean()) * 1e4, 2))
                set_status(funding_basis=basis); log(f'funding basis Bybit vs Binance (24h, top {len(bn)}): {basis}')
                if now_.weekday() == 0:
                    tg(f'📐 Фандинг Bybit vs Binance за сутки (топ-{len(bn)}): корр {basis["corr"]}, средняя разница {basis["mean_abs_diff_bps"]} бп')
        except Exception as e:
            log(f'funding basis check failed: {e}')

    # ------------------------------------------------------------ exchange-side stop loss on every position (firm rule)
    def ensure_stops(self, positions):
        """Some firms (HyroTrader) require a Bybit 'TP/SL' stop on every position at open. Set a far position-level stop
        (SL_PCT_LONG / SL_PCT_SHORT from entry) on any position without one; the bot's own risk logic remains the real control."""
        n = 0
        for sym, p in positions.items():
            if p['qty'] == 0 or any(str(l.get('sl') or '') not in ('', '0', '0.0') for l in p.get('legs', [])):
                continue
            meta = self.instr.get(sym)
            if not meta or not meta.get('tickSize') or p['entry'] <= 0:
                continue
            px = p['entry'] * (1 - C.SL_PCT_LONG / 100.0) if p['qty'] > 0 else p['entry'] * (1 + C.SL_PCT_SHORT / 100.0)
            try:
                if not C.DRY_RUN:
                    self.api.set_trading_stop(sym, fmt_price(px, meta['tickSize'], 'Sell' if p['qty'] > 0 else 'Buy'))
                n += 1
            except Exception as e:
                log(f'trading-stop {sym}: {e}')
        if n:
            log(f'exchange-side stops set on {n} positions')
        return n

    # ------------------------------------------------------------ hourly proportional scale pass
    def scale_pass(self):
        with self.lock:
            m = self.mem
            if not m.get('last_targets') or m.get('k_ref') in (None, 0):
                return
            equity = self.equity()
            if not self.guards(equity):
                return
            k, why = self.risk.multiplier(equity, self.now())
            ratio = k / float(m['k_ref'])
            if ratio >= 0.85:
                return                      # only de-risk intraday; increases wait for the daily rebalance
            log(f'SCALE PASS: K {float(m["k_ref"]):.2f} -> {k:.2f} ({why}); scaling book x{ratio:.2f}')
            targets = {s: v * ratio for s, v in m['last_targets'].items()}
            tickers = self.api.tickers(); positions = self.api.positions()
            legs = [l for l in self.ex.plan(targets, positions, tickers, self.instr, C.REBAL_BAND, C.MIN_TRADE_USDT) if l[3]]
            self.ex.execute(legs, self.instr, tickers, self.api.positions, maker_attempts=1, wait_s=60)
            m['last_targets'] = targets; m['k_ref'] = k; m.save()

    # ------------------------------------------------------------ watchdog
    def watchdog(self):
        m = self.mem
        if m['halt_total'] or m['target_hit']:
            return
        try:
            equity = self.equity()
        except Exception as e:
            log(f'watchdog equity fail: {e}'); return
        with self.lock:
            if not self.guards(equity):
                return
            try:
                poss = self.api.positions()
            except Exception as e:
                log(f'watchdog positions fail: {e}'); return
            if not poss:
                return
            # short stop-loss: mark >= entry * (1 + STOP)
            for sym, p in poss.items():
                if C.STOP_SHORT_PCT > 0 and p['qty'] < 0 and p['entry'] > 0 and p['mark'] >= p['entry'] * (1 + C.STOP_SHORT_PCT / 100.0):   # 0 = stop disabled
                    meta = self.instr.get(sym)
                    if not meta:
                        continue
                    q = round_qty(abs(p['qty']), meta)
                    try:
                        if not C.DRY_RUN:
                            self.api.market(sym, q, 'Buy', reduce_only=True)
                        m['cooldown'][sym] = self.now().timestamp() + C.STOP_COOLDOWN_H * 3600
                        m['last_targets'].pop(sym, None); m.save()
                        log(f'STOP: {sym} short closed at mark {p["mark"]} (entry {p["entry"]}, upnl {p["upnl"]:,.0f})')
                        tg(f'✂️ Стоп по шорту {sym}: цена +{(p["mark"] / p["entry"] - 1) * 100:.0f}% от входа, убыток {p["upnl"]:,.0f} USDT. Карантин {C.STOP_COOLDOWN_H}ч.')
                    except Exception as e:
                        tg(f'🛑 Стоп {sym} не исполнен: {e} — ЗАКРОЙ ВРУЧНУЮ'); log(f'stop {sym}: {e}')
            # per-position loss guard (firm 'per trade loss' rules; RULES.per_trade_loss % of initial, 0 = off)
            ptl = float(C.RULES.get('per_trade_loss', 0.0) or 0.0)
            if ptl > 0:
                for sym, p in list(poss.items()):
                    if p.get('upnl', 0.0) < -0.9 * ptl / 100.0 * self.risk.initial and p['qty'] != 0:
                        meta = self.instr.get(sym)
                        if not meta:
                            continue
                        try:
                            if not C.DRY_RUN:
                                self.api.market(sym, round_qty(abs(p['qty']), meta), 'Sell' if p['qty'] > 0 else 'Buy', reduce_only=True)
                            m['cooldown'][sym] = self.now().timestamp() + C.STOP_COOLDOWN_H * 3600
                            m['last_targets'].pop(sym, None); m.save()
                            log(f'PER-TRADE LOSS GUARD: {sym} closed, upnl {p["upnl"]:,.0f} (limit {ptl}% of {self.risk.initial:,.0f})')
                            tg(f'✂️ Лимит убытка по позиции {sym}: {p["upnl"]:,.0f} USDT (правило фирмы {ptl}% от initial). Закрыта, карантин {C.STOP_COOLDOWN_H}ч.')
                        except Exception as e:
                            tg(f'🛑 Не смог закрыть {sym} по лимиту убытка: {e} — ЗАКРОЙ ВРУЧНУЮ'); log(f'per-trade guard {sym}: {e}')
            # notional / margin guards vs firm limits (of INITIAL)
            g = sum(abs(p['qty'] * p['mark']) for p in poss.values())
            im = sum(p.get('im', 0.0) for p in poss.values())
            lim_g = C.RULES['notional_x'] * self.risk.initial
            lim_m = C.RULES['margin_pct'] / 100.0 * self.risk.initial
            if g > 0.95 * lim_g or (im > 0 and im > 0.95 * lim_m):
                cut = 1.0 - min(0.90 * lim_g / g if g > 0 else 1.0, 0.90 * lim_m / im if im > 0 else 1.0)
                log(f'GROSS GUARD: gross {g:,.0f} (limit {lim_g:,.0f}), IM {im:,.0f} (limit {lim_m:,.0f}) — cutting {cut * 100:.0f}%')
                tg(f'⚠️ Объём {g:,.0f} / маржа {im:,.0f} у лимитов фирмы — сокращаю все позиции на {cut * 100:.0f}%')
                for sym, p in poss.items():
                    meta = self.instr.get(sym)
                    if not meta:
                        continue
                    q = round_qty(abs(p['qty']) * cut, meta)
                    if q >= meta['minQty'] and not C.DRY_RUN:
                        try:
                            self.api.market(sym, q, 'Sell' if p['qty'] > 0 else 'Buy', reduce_only=True)
                        except Exception as e:
                            log(f'gross guard {sym}: {e}')

    # ------------------------------------------------------------ startup
    def start(self):
        m = self.mem
        cfg, h = C.effective_config(); log(f'effective config hash {h} ({len(cfg)} keys)'); set_status(config_hash=h)
        if C.EXPECTED_CONFIG_HASH and h != C.EXPECTED_CONFIG_HASH:
            tg(f'🛑 Конфигурация ({h}) не совпадает с пред-регистрацией ({C.EXPECTED_CONFIG_HASH}). Бот НЕ запущен.')
            raise SystemExit(f'FATAL: effective config hash {h} != EXPECTED_CONFIG_HASH {C.EXPECTED_CONFIG_HASH}')
        m.load()
        # latches are cleared one by one (Opus: one flag used to clear the total stop AND the phase-passed latch)
        if os.environ.get('RESET_TOTAL_STOP', '0') == '1':
            m['halt_total'] = False; m.save(); log('RESET_TOTAL_STOP=1: total-stop latch cleared')
        if os.environ.get('RESET_TARGET_HIT', '0') == '1':
            m['target_hit'] = False; m.save(); log('RESET_TARGET_HIT=1: phase-passed latch cleared')
        if os.environ.get('RESET_ABANDONED', '0') == '1':
            m['abandoned'] = False; m['halt_total'] = False; m.save(); log('RESET_ABANDONED=1: abandon latch cleared')
        try:
            self.api.switch_one_way()
        except ApiError as e:
            if e.code != 110025:
                log(f'switch one-way: {e}')
        self.md.refresh_instruments(force=True); self.instr = self.md.instr
        if m.get('held_w'):
            self.sleeves.held = {k: dict(v) for k, v in m['held_w'].items()}      # restore held daily sleeves
        equity = self.equity()
        if not m.loaded:
            anchor, src = self.recover_day_anchor(equity)
            m['day_date'] = self.now().date().isoformat(); m['day_anchor'] = anchor; m['day_peak'] = max(anchor, equity)
            if C.RULES['total_dd_mode'] == 'trailing':
                m['eq_peak'] = max(float(os.environ.get('EQ_PEAK', 0) or 0), float(C.ACCOUNT_SIZE), equity)
                tg_once('peak_reset', f'⚠️ Память бота пуста: пик equity для trailing-просадки взят как {m["eq_peak"]:,.2f}. '
                                      f'Если реальный пик выше — задай EQ_PEAK в env и перезапусти.', 24)
            m.save()
            log(f'fresh state: anchor {anchor:,.2f} ({src}), peak {m["eq_peak"]:,.2f}')
        elif m.get('day_date') != self.now().date().isoformat():
            # restarted on a new UTC day after downtime: today's realised PnL so far is not in memory -> roll the day over,
            # then anchor from the exchange transaction log instead of the current equity (which may already be down)
            self.risk.observe_equity(equity, self.now())
            anchor, src = self.recover_day_anchor(equity)
            m['day_anchor'] = anchor; m['day_peak'] = max(anchor, equity); m.save()
            log(f'day anchor after downtime: {anchor:,.2f} ({src}), equity {equity:,.2f}')
        # leverage on the tradable set (best effort)
        if not C.DRY_RUN and C.SET_LEVERAGE > 0:
            tk = self.api.tickers()
            top = sorted([s for s in tk if s in self.instr], key=lambda s: -tk[s]['turnover24h'])[:260]
            n = 0
            for s in top:
                try:
                    self.api.set_leverage(s, min(C.SET_LEVERAGE, self.instr[s]['maxLeverage'] or C.SET_LEVERAGE)); n += 1
                except ApiError as e:
                    if e.code != 110043:
                        log(f'leverage {s}: {e}')
                except Exception as e:
                    log(f'leverage {s}: {e}')
                time.sleep(0.03)
            log(f'leverage set on {n} symbols')
        tf = self.risk.total_floor()
        tg(f'✅ PROP-SLEEVES v{C.VERSION} запущен | {C.FIRM} | {C.MODE} фаза {C.PHASE} | initial {C.ACCOUNT_SIZE:,.0f} | equity {equity:,.2f}\n'
           f'Правила: total DD {C.RULES["total_dd"]}% ({C.RULES["total_dd_mode"]}), daily {C.RULES["daily_dd"]}% ({C.RULES["daily_dd_mode"]}), '
           f'цели {C.RULES["targets"]}, consistency {C.RULES["consistency"]}%, notional ≤{C.RULES["notional_x"]}x\n'
           f'Пол сейчас: {tf:,.2f} | рукава {C.SLEEVES} | BASE {C.BASE_SCALE} CPPI {C.CPPI_FRAC} VT {C.VT_TARGET_DVOL * 100:.2f}%/д\n'
           f'Ребаланс {C.REBAL_HOUR_UTC:02d}:{C.REBAL_MINUTE:02d} UTC | {"DRY_RUN" if C.DRY_RUN else "LIVE"} | {C.BASE_URL}')

    def due_slot(self, now):
        """Most recent scheduled rebalance time <= now on the REBAL_HOUR_UTC + k*REBAL_EVERY_H grid (at REBAL_MINUTE)."""
        base = now.replace(hour=C.REBAL_HOUR_UTC, minute=C.REBAL_MINUTE, second=0, microsecond=0)
        if base > now:
            base -= timedelta(days=1)
        k = int((now - base).total_seconds() // (int(C.REBAL_EVERY_H) * 3600))
        return base + timedelta(hours=k * int(C.REBAL_EVERY_H))

    def loop(self):
        self.start()
        while True:
            try:
                now = self.now()
                due = self.due_slot(now); slot = f'{due.date().isoformat()}T{due.hour:02d}'
                if self.mem.get('last_rebal_slot') != slot:       # covers a missed slot (downtime): rebalance at the next wake-up
                    self.rebalance(slot)
                elif now.minute >= C.REBAL_MINUTE:
                    self.scale_pass()
            except Exception as e:
                log(f'CYCLE FAILED: {e}\n{traceback.format_exc()[:1500]}')
                tg_once('cycle_crash', f'⚠️ Цикл упал: {e}', 3)
            now2 = self.now(); nxt = now2.replace(minute=C.REBAL_MINUTE, second=0, microsecond=0)
            if nxt <= now2:
                nxt += timedelta(hours=1)                          # (a start at hh:05 waits for hh:10, not for the next hour)
            time.sleep(max(30.0, (nxt - now2).total_seconds() + 5))

    def watchdog_loop(self):
        while True:
            time.sleep(C.WATCHDOG_SEC)
            try:
                self.watchdog()
                lc = self.mem.get('last_rebal_date')
                if lc and (self.now().date() - datetime.fromisoformat(lc).date()).days >= 2:
                    tg_once('silent', '⚠️ Ребаланс не выполнялся 2 дня — проверь логи.', 12)
            except Exception as e:
                log(f'watchdog: {e}')


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        tok = ''
        if '?' in self.path:
            from urllib.parse import parse_qs
            tok = (parse_qs(self.path.split('?', 1)[1]).get('token') or [''])[0]
        ok = bool(C.STATUS_TOKEN) and (tok == C.STATUS_TOKEN or self.headers.get('X-Status-Token', '') == C.STATUS_TOKEN)
        body = json.dumps(get_status(ok and self.path.startswith('/status')), ensure_ascii=False, default=str).encode()
        self.send_response(200); self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body))); self.end_headers(); self.wfile.write(body)

    def log_message(self, *a):
        pass


def main():
    if not C.API_KEY or not C.API_SECRET:
        log('NO KEYS: set BYBIT_API_KEY / BYBIT_API_SECRET'); sys.exit(1)
    api = Bybit(C.BASE_URL, C.API_KEY, C.API_SECRET, log)
    bot = Bot(api)
    threading.Thread(target=bot.watchdog_loop, daemon=True).start()
    threading.Thread(target=bot.loop, daemon=True).start()
    HTTPServer(('0.0.0.0', C.PORT), Handler).serve_forever()


if __name__ == '__main__':
    main()
