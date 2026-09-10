#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Execution for PROP-SLEEVES v4: converge positions to target notionals.
 * every open position is considered (also names outside the universe) — no silent skips;
 * reductions/closures first, then openings; PostOnly at the touch with re-quotes, market fallback;
 * fills verified by re-reading positions (robust to demo quirks); dust closed fully."""
import os, sys, time, math
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
from bybit import ApiError


def round_qty(qty, meta):
    step = meta['qtyStep']
    q = math.floor(abs(qty) / step + 1e-9) * step
    prec = max(0, int(round(-math.log10(step)))) if step < 1 else 0
    return round(q, prec)


def fmt_price(price, tick, side):
    n = price / tick
    n = math.floor(n + 1e-9) if side == 'Buy' else math.ceil(n - 1e-9)
    val = n * tick
    s = f'{tick:.10f}'.rstrip('0')
    prec = len(s.split('.')[1]) if '.' in s else 0
    return f'{val:.{prec}f}'


class Executor:
    def __init__(self, api, log=print, tg=None, dry_run=False):
        self.api, self.log, self.tg, self.dry = api, log, (tg or (lambda *_: None)), dry_run
        self.stats = dict(maker_sent=0, market_sent=0, closes=0, opens=0, errors=0)          # per execute() call (cycle)
        self.stats_total = dict(self.stats)                                                  # cumulative since start

    def _reset_stats(self):
        for k, v in self.stats.items():
            self.stats_total[k] = self.stats_total.get(k, 0) + v
        self.stats = dict(maker_sent=0, market_sent=0, closes=0, opens=0, errors=0)

    # ------------------------------------------------------------ planning
    def plan(self, targets, positions, tickers, instr, band, min_trade, no_touch=()):
        """targets: {sym: usd}. Returns list of legs (sym, side, usd, reduce_only, full_close)."""
        legs, missing = [], []
        for sym in sorted(set(targets) | set(positions)):
            if sym in no_touch:
                continue
            meta, tk, pos = instr.get(sym), tickers.get(sym), positions.get(sym)
            px = (tk or {}).get('mark') or (tk or {}).get('last') or (pos or {}).get('mark') or 0.0
            if not meta or px <= 0:
                if pos:
                    missing.append(sym)
                continue
            cur = (pos['qty'] if pos else 0.0) * px
            tgt = float(targets.get(sym, 0.0))
            if 0 < abs(tgt) < min_trade:
                tgt = 0.0
            full_close = tgt == 0.0 and cur != 0.0
            diff = tgt - cur
            if not full_close and abs(diff) < max(min_trade, band * abs(tgt)):
                continue
            if cur != 0 and (tgt == 0 or np.sign(tgt) != np.sign(cur)):
                legs.append((sym, 'Sell' if cur > 0 else 'Buy', abs(cur), True, True))          # close
                if tgt != 0:
                    legs.append((sym, 'Buy' if tgt > 0 else 'Sell', abs(tgt), False, False))   # flip open
            elif abs(tgt) < abs(cur):
                legs.append((sym, 'Sell' if cur > 0 else 'Buy', abs(cur) - abs(tgt), True, False))  # reduce
            else:
                legs.append((sym, 'Buy' if diff > 0 else 'Sell', abs(diff), False, False))       # open/add
        if missing:
            self.log(f'NO PRICE/META for open positions: {missing} — closing at mark')
            self.tg(f'⚠️ Позиции без цены/меты: {missing}. Закрываю по марку.')
            for sym in missing:
                pos = positions[sym]
                legs.insert(0, (sym, 'Sell' if pos['qty'] > 0 else 'Buy', abs(pos['qty'] * (pos.get('mark') or 1.0)), True, True))
        legs.sort(key=lambda l: (not l[3], l[0]))      # reductions first
        return legs

    # ------------------------------------------------------------ sending
    def _send(self, sym, side, qty, ro, meta, price_str=None):
        if self.dry:
            self.log(f'DRY {sym} {side} {qty} {"RO" if ro else ""} {"PO@" + price_str if price_str else "MKT"}')
            return True
        left = float(qty); sent = False
        while True:
            q = round_qty(min(left, meta['maxQty']), meta)
            if q < meta['minQty'] or q <= 0:
                break
            if price_str:
                self.api.post_only(sym, q, side, price_str, reduce_only=ro); self.stats['maker_sent'] += 1
            else:
                self.api.market(sym, q, side, reduce_only=ro); self.stats['market_sent'] += 1
            sent = True
            left = round(left - q, 12)
            if left < meta['qtyStep'] - 1e-9:
                break
            time.sleep(0.05)
        return sent

    def _remaining(self, sym, side, ro, target_qty_signed, positions):
        """Remaining signed qty to trade for this leg given current position."""
        cur = positions.get(sym, {}).get('qty', 0.0)
        return target_qty_signed - cur

    def _order_params(self, sym, side, q, ro, price_str=None):
        p = {'category': 'linear', 'symbol': sym, 'side': side, 'qty': str(q), 'positionIdx': 0}
        if price_str:
            p.update(orderType='Limit', price=price_str, timeInForce='PostOnly')
        else:
            p['orderType'] = 'Market'
        if ro:
            p['reduceOnly'] = True
        return p

    def _send_quotes(self, quotes):
        """Send PostOnly quotes in batches (fallback: one by one). quotes: list of (sym, side, q, ro, price_str, meta).
        Returns {sym: (code, msg)} with code 0 = accepted."""
        res = {}
        if self.dry:
            for sym, side, q, ro, ps, meta in quotes:
                self.log(f'DRY {sym} {side} {q} {"RO" if ro else ""} PO@{ps}'); res[sym] = (0, '')
            return res
        singles = [x for x in quotes if x[2] > x[5]['maxQty']]                 # oversized legs go through _send (splitting)
        batch = [x for x in quotes if x[2] <= x[5]['maxQty']]
        if len(batch) >= 2 and hasattr(self.api, 'create_batch'):
            try:
                codes = self.api.create_batch([self._order_params(s, sd, q, ro, ps) for s, sd, q, ro, ps, _ in batch])
                for (sym, *_), (c, msg) in zip(batch, codes):
                    res[sym] = (c, msg); self.stats['maker_sent'] += 1
                batch = []
            except Exception as e:
                self.log(f'batch quotes failed ({e}); falling back to single orders')
        for sym, side, q, ro, ps, meta in singles + batch:
            try:
                self._send(sym, side, q, ro, meta, ps); res[sym] = (0, '')
            except ApiError as e:
                res[sym] = (e.code, str(e))
            except Exception as e:
                res[sym] = (-1, str(e))
        return res

    def execute(self, legs, instr, tickers, positions_fn, maker_attempts=None, wait_s=None):
        """Execute legs: PostOnly at touch (batched), re-quote, then market. Returns (stats, errors); stats are per cycle."""
        maker_attempts = C.MAKER_ATTEMPTS if maker_attempts is None else maker_attempts
        wait_s = C.MAKER_WAIT_S if wait_s is None else wait_s
        self._reset_stats()
        errors, dead = [], set()
        maker_notional = market_notional = 0.0
        # desired final signed qty per symbol from legs (apply sequentially per symbol)
        pos0 = positions_fn()
        want = {}
        for sym, side, usd, ro, full in legs:
            meta = instr.get(sym); tk = tickers.get(sym, {})
            px = tk.get('mark') or tk.get('last') or pos0.get(sym, {}).get('mark') or 0.0
            if not meta or px <= 0:
                continue
            cur = want.get(sym, pos0.get(sym, {}).get('qty', 0.0))
            dq = usd / px * (1 if side == 'Buy' else -1)
            if full:
                want[sym] = 0.0
            else:
                want[sym] = cur + dq
        active = dict(want)
        # execution quality instrumentation (Opus review): decision price per symbol, fill-time proxy price, timing
        t_start = time.time(); dec_px = {}; fill_bps = []; n_maker_filled = 0; n_market = 0; t_last_fill = t_start
        for sym in want:
            tk = tickers.get(sym, {}); dec_px[sym] = tk.get('mark') or tk.get('last') or 0.0
        prev_active = {}
        for attempt in range(maker_attempts + 1):
            if not active:
                break
            final = attempt == maker_attempts
            try:
                self.api.cancel_all()
            except Exception as e:
                self.log(f'cancel_all: {e}')
            time.sleep(0.5)
            pos = positions_fn()
            tks = tickers if attempt == 0 else self.api.tickers()
            if attempt > 0:                       # legs that were quoted last round and no longer need trading -> filled as maker
                for sym, tq in prev_active.items():
                    cur = pos.get(sym, {}).get('qty', 0.0); px_now = (tks.get(sym) or {}).get('mark') or 0.0
                    if abs(tq - cur) * max(px_now, 1e-9) < C.MIN_TRADE_USDT * 0.5 and dec_px.get(sym, 0) > 0 and px_now > 0:
                        side = 1.0 if tq > pos0.get(sym, {}).get('qty', 0.0) else -1.0
                        fill_bps.append(side * (px_now / dec_px[sym] - 1.0) * 1e4); n_maker_filled += 1; t_last_fill = time.time()
                        maker_notional += abs(tq - prev_qty.get(sym, pos0.get(sym, {}).get('qty', 0.0))) * px_now
            prev_active = dict(active); prev_qty = {s: pos.get(s, {}).get('qty', 0.0) for s in active}
            still = {}; quotes = []
            for sym, tq in active.items():
                if sym in dead:
                    continue
                meta = instr[sym]
                cur = pos.get(sym, {}).get('qty', 0.0)
                dq = tq - cur
                tk = tks.get(sym, {})
                px = tk.get('mark') or tk.get('last') or 0.0
                if px <= 0:
                    continue
                if abs(dq) * px < max(C.MIN_TRADE_USDT * 0.5, meta.get('minNotional', 5.0)) and not (tq == 0 and cur != 0):
                    continue
                q = round_qty(abs(dq), meta)
                if q < meta['minQty']:
                    if tq == 0 and cur != 0:
                        q = round_qty(abs(cur), meta)
                        if q < meta['minQty']:
                            continue
                    else:
                        continue
                side = 'Buy' if dq > 0 else 'Sell'
                ro = (cur != 0) and (np.sign(dq) != np.sign(cur)) and abs(dq) <= abs(cur) + 1e-12
                if tq == 0 and cur != 0:
                    q = round_qty(abs(cur), meta); ro = True
                if final or not meta.get('tickSize'):
                    try:
                        self._send(sym, side, q, ro, meta, None); n_market += 1; t_last_fill = time.time(); market_notional += q * px
                        if dec_px.get(sym, 0) > 0:
                            fill_bps.append((1.0 if side == 'Buy' else -1.0) * (px / dec_px[sym] - 1.0) * 1e4)
                        self.stats['closes' if ro else 'opens'] += 1
                    except ApiError as e:
                        if e.code in (110123, 110125, 110126, 10029):
                            dead.add(sym); errors.append((sym, e.code, str(e))); self.log(f'{sym}: permanent block {e}')
                        else:
                            errors.append((sym, e.code, str(e))); self.stats['errors'] += 1; self.log(f'order {sym} {side} {q}: {e}')
                    except Exception as e:
                        errors.append((sym, None, str(e))); self.stats['errors'] += 1; self.log(f'order {sym}: {e}')
                else:
                    touch = tk.get('bid') if side == 'Buy' else tk.get('ask')
                    quotes.append((sym, side, q, ro, fmt_price(touch or px, meta['tickSize'], side), meta)); still[sym] = tq
            # PostOnly quotes for this round, batched (<=10 per request); per-order codes decide what stays active
            if quotes:
                res = self._send_quotes(quotes)
                for sym, side, q, ro, ps, meta in quotes:
                    code, msg = res.get(sym, (0, ''))
                    if code == 0:
                        self.stats['closes' if ro else 'opens'] += 1
                    elif code in (110017, 110019, 30208):                    # post-only would cross -> re-quote next round
                        pass
                    elif code in (110123, 110125, 110126, 10029):
                        dead.add(sym); still.pop(sym, None); errors.append((sym, code, msg)); self.log(f'{sym}: permanent block {code} {msg}')
                    else:
                        errors.append((sym, code, msg)); self.stats['errors'] += 1; self.log(f'quote {sym} {side} {q}: {code} {msg}')
            active = still
            if active and not final:
                time.sleep(wait_s)
        try:
            self.api.cancel_all()
        except Exception:
            pass
        st = self.stats.copy()
        n_fills = n_maker_filled + n_market; tot_notional = maker_notional + market_notional
        st.update(exec_seconds=round(time.time() - t_start, 1), last_fill_seconds=round(t_last_fill - t_start, 1), legs=len(want),
                  maker_fill_share=round(n_maker_filled / n_fills, 3) if n_fills else None,                        # by legs (proxy)
                  maker_notional_share=round(maker_notional / tot_notional, 3) if tot_notional > 0 else None,      # by notional (proxy)
                  market_legs=n_market,
                  adverse_bps_mean=round(float(np.mean(fill_bps)), 2) if fill_bps else None, adverse_bps_p90=round(float(np.percentile(fill_bps, 90)), 2) if fill_bps else None)
        # real fills from the exchange (best effort): notional-weighted maker share and slippage vs the decision price
        if not self.dry and hasattr(self.api, 'executions'):
            try:
                fills = self.api.executions(int(t_start * 1000) - 2000)
                mk = tot = 0.0; wbps = []; wts = []
                for f in fills:
                    sym = f.get('symbol'); q = float(f.get('execQty') or 0); px = float(f.get('execPrice') or 0); notional = q * px
                    if notional <= 0:
                        continue
                    tot += notional; mk += notional if str(f.get('isMaker')).lower() in ('true', '1') else 0.0
                    if dec_px.get(sym, 0) > 0:
                        wbps.append((1.0 if f.get('side') == 'Buy' else -1.0) * (px / dec_px[sym] - 1.0) * 1e4); wts.append(notional)
                if tot > 0:
                    st.update(fills_n=len(fills), fills_notional=round(tot, 0), maker_share_real=round(mk / tot, 3),
                              adverse_bps_real=round(float(np.average(wbps, weights=wts)), 2) if wbps else None)
            except Exception as e:
                self.log(f'executions fetch failed: {e}')
        return st, errors

    # ------------------------------------------------------------ emergency
    def flatten_all(self, instr, reason):
        try:
            self.api.cancel_all()
        except Exception as e:
            self.log(f'cancel_all: {e}')
        try:
            poss = self.api.positions()
        except Exception as e:
            self.tg(f'🛑 Не смог прочитать позиции для закрытия ({reason}): {e}. ЗАКРОЙ ВРУЧНУЮ.')
            return -1
        if not poss:
            return 0
        self.log(f'FLATTEN ALL: {reason} ({len(poss)} symbols)')
        self.tg(f'🛑 Закрываю все позиции ({len(poss)}). Причина: {reason}')
        fails = []
        for sym, p in poss.items():
            meta = instr.get(sym)
            if not meta:
                fails.append(sym); continue
            for leg in p.get('legs') or [dict(qty=p['qty'], positionIdx=0)]:
                q = round_qty(abs(leg['qty']), meta)
                if q < meta['minQty']:
                    continue
                try:
                    if not self.dry:
                        self.api.market(sym, q, 'Sell' if leg['qty'] > 0 else 'Buy', reduce_only=True, position_idx=leg.get('positionIdx', 0))
                except Exception as e:
                    fails.append(f'{sym}: {e}')
        if fails:
            self.tg('⚠️ НЕ ЗАКРЫЛИСЬ: ' + ', '.join(str(f) for f in fails[:10]) + '. Закрой вручную.')
        return len(poss)
