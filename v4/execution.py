#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Execution for PROP-SLEEVES v4: converge positions to target notionals.
 * every open position is considered (also names outside the universe) — no silent skips;
 * reductions/closures first, then openings; PostOnly at the touch with re-quotes, market fallback;
 * fills verified by re-reading positions (robust to demo quirks); dust closed fully."""
import os, sys, time, math
from decimal import Decimal, ROUND_FLOOR
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
from bybit import ApiError, fmt_qty


def round_qty(qty, meta):
    """Floor |qty| to a whole multiple of qtyStep (exact decimal arithmetic: 1.5 at step 0.5 stays 1.5, never rounds up)."""
    d = Decimal(str(meta['qtyStep']))
    n = (Decimal(str(abs(qty))) / d).to_integral_value(ROUND_FLOOR)
    return float(n * d)


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
        self.abort = None                         # threading.Event set by the watchdog when a hard risk limit is hit mid-rebalance
        self.tg_once = None                       # optional throttled sender tg_once(key, text, hours) (set by the bot); falls back to tg

    def _reset_stats(self):
        for k, v in self.stats.items():
            self.stats_total[k] = self.stats_total.get(k, 0) + v
        self.stats = dict(maker_sent=0, market_sent=0, closes=0, opens=0, errors=0)

    # ------------------------------------------------------------ alerts / cancel
    def _alert(self, key, text, hours=1.0):
        """Throttled Telegram alert (an unattended loop must not repeat the same message every cycle)."""
        if self.tg_once is not None:
            self.tg_once(key, text, hours)
        else:
            self.tg(text)

    def _cancel_verified(self):
        """cancel_all, then check open_orders is empty (Bybit.cancel_all swallows API errors): retry once, else Telegram. True when the book is clean
        (or cannot be checked: dry run / a client without open_orders)."""
        try:
            self.api.cancel_all()
        except Exception as e:
            self.log(f'cancel_all: {e}')
        if self.dry or not hasattr(self.api, 'open_orders'):
            return True
        left = None
        for att in range(2):
            try:
                left = self.api.open_orders()
            except Exception as e:
                self.log(f'open_orders check: {e}'); left = None
            if left is not None and not left:
                return True
            if att == 0:
                self.log(f'cancel_all left {len(left) if left is not None else "?"} open orders; retrying once')
                time.sleep(0.5)
                try:
                    self.api.cancel_all()
                except Exception as e:
                    self.log(f'cancel_all (retry): {e}')
                time.sleep(0.5)
        syms = sorted({str(o.get('symbol')) for o in (left or [])})[:10]
        self._alert('cancel_left', f'⚠️ После cancel_all на бирже остались лимитные ордера ({len(left) if left is not None else "не удалось проверить"}): {syms}. '
                                   f'Они могут исполниться вне плана — проверь вручную.', 1.0)
        return False

    # ------------------------------------------------------------ planning
    def plan(self, targets, positions, tickers, instr, band, min_trade, no_touch=(), frac=1.0, start=None):
        """targets: {sym: usd}. Returns list of legs (sym, side, usd, reduce_only, full_close).
        frac < 1 (REBAL_PARTIAL, reg. 5): partial rebalance — the book moves `frac` of the gap between the position at the START of the
        rebalance slot and the target (full close stays full). start = {sym: qty at slot start}: the band decision and the step are
        taken from it, so a retry of the same slot completes the same step instead of stepping again (idempotent). Same arithmetic as
        bt.run(trade_frac=...). frac >= 1: unchanged behaviour."""
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
            if frac < 1.0:
                ref = cur if start is None else float(start.get(sym, 0.0)) * px
                if not full_close and abs(tgt - ref) < max(min_trade, band * abs(tgt)):
                    continue
                if not full_close:
                    tgt = ref + frac * (tgt - ref)
                    diff = tgt - cur
                    if abs(diff) < min_trade:         # step already done (retry) or too small to place
                        continue
            elif not full_close and abs(diff) < max(min_trade, band * abs(tgt)):
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

    def _order_params(self, sym, side, q, ro, price_str=None, meta=None):
        p = {'category': 'linear', 'symbol': sym, 'side': side, 'qty': fmt_qty(q, (meta or {}).get('qtyStep')), 'positionIdx': 0}
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
                codes = self.api.create_batch([self._order_params(s, sd, q, ro, ps, mt) for s, sd, q, ro, ps, mt in batch])
                for (sym, *_), (c, msg) in zip(batch, codes):
                    res[sym] = (c, msg); self.stats['maker_sent'] += 1
                batch = []
            except Exception as e:
                # F4: a later chunk failed: the chunks before it are ON THE BOOK (BatchError.done). Resend only the legs whose orders were not
                # accepted, and even of those skip any that already rests (the failed request may have been processed: timeout after the fact)
                done = list(getattr(e, 'done', None) or [])
                self.log(f'batch quotes failed after {len(done)} accepted ({e}); resending only the rest as single orders')
                for (sym, *_), (c, msg) in zip(batch, done):
                    res[sym] = (c, msg); self.stats['maker_sent'] += 1
                rest = batch[len(done):]
                resting = self._resting(rest)
                for x in rest:
                    if (x[0], x[1]) in resting:
                        res[x[0]] = (0, ''); self.stats['maker_sent'] += 1
                        self.log(f'{x[0]} {x[1]}: quote already resting, not resent')
                batch = [x for x in rest if (x[0], x[1]) not in resting]
        for sym, side, q, ro, ps, meta in singles + batch:
            try:
                self._send(sym, side, q, ro, meta, ps); res[sym] = (0, '')
            except ApiError as e:
                res[sym] = (e.code, str(e))
            except Exception as e:
                res[sym] = (-1, str(e))
        return res

    def _resting(self, quotes):
        """{(symbol, side)} of the given quotes that already rest on the book (best effort: unknown -> empty set -> they are resent)."""
        if not quotes or not hasattr(self.api, 'open_orders'):
            return set()
        try:
            want = {(x[0], x[1]) for x in quotes}
            return {(o.get('symbol'), o.get('side')) for o in self.api.open_orders() if (o.get('symbol'), o.get('side')) in want}
        except Exception as e:
            self.log(f'open_orders (dedupe) failed: {e}')
            return set()

    def execute(self, legs, instr, tickers, positions_fn, maker_attempts=None, wait_s=None, maker_only=None, force_market=()):
        """Execute legs: PostOnly at touch (batched), re-quote, then market. Returns (stats, errors); stats are per cycle.
        maker_only (default C.MAKER_ONLY, reg. 5c): the unfilled remainder of a normal leg is NOT sent to market (cancelled, re-evaluated at the
        next slot); symbols in force_market (delisting / cooldown / blocked names) still finish at market. Callers of risk passes use maker_only=False."""
        maker_only = bool(C.MAKER_ONLY) if maker_only is None else bool(maker_only)
        force_market = set(force_market)
        n_unfilled = 0; unfilled_usd = 0.0
        maker_attempts = C.MAKER_ATTEMPTS if maker_attempts is None else maker_attempts
        wait_s = C.MAKER_WAIT_S if wait_s is None else wait_s
        self._reset_stats()
        errors, dead = [], set()
        maker_notional = market_notional = 0.0
        # desired final signed qty per symbol from legs (apply sequentially per symbol)
        pos0 = positions_fn()
        want = {}
        no_meta, no_px = [], []
        for sym, side, usd, ro, full in legs:
            meta = instr.get(sym); tk = tickers.get(sym, {})
            px = tk.get('mark') or tk.get('last') or pos0.get(sym, {}).get('mark') or 0.0
            if not meta or px <= 0:
                (no_px if meta else no_meta).append(sym)           # F10: never dropped silently
                continue
            cur = want.get(sym, pos0.get(sym, {}).get('qty', 0.0))
            dq = usd / px * (1 if side == 'Buy' else -1)
            if full:
                want[sym] = 0.0
            else:
                want[sym] = cur + dq
        if no_meta or no_px:
            self.log(f'LEGS DROPPED: no instrument meta {no_meta}, no price {no_px}')
            self._alert('legs_no_meta', f'⚠️ Ноги не выставлены: нет данных инструмента {sorted(set(no_meta))[:10]}, нет цены {sorted(set(no_px))[:10]}. '
                                        f'Если это открытая позиция — она осталась на бирже, закрой вручную.', 1.0)
        active = dict(want)
        # execution quality instrumentation (Opus review): decision price per symbol, fill-time proxy price, timing
        t_start = time.time(); dec_px = {}; fill_bps = []; n_maker_filled = 0; n_market = 0; t_last_fill = t_start
        for sym in want:
            tk = tickers.get(sym, {}); dec_px[sym] = tk.get('mark') or tk.get('last') or 0.0
        prev_active = {}; aborted = False
        try:
            for attempt in range(maker_attempts + 1):
                if not active:
                    break
                if self.abort is not None and self.abort.is_set():      # checked between quote rounds: no new orders, open quotes cancelled below
                    aborted = True; errors.append(('*', None, 'aborted by watchdog (risk limit)'))
                    self.log(f'EXECUTE ABORTED (watchdog: risk limit hit): round {attempt}, {len(active)} legs not traded'); break
                final = attempt == maker_attempts
                self._cancel_verified()
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
                    if final and maker_only and sym not in force_market and meta.get('tickSize'):
                        n_unfilled += 1; unfilled_usd += abs(dq) * px     # reg. 5c: no market chasing; the remainder is carried to the next slot
                        continue
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
        finally:                                       # F4: an exception anywhere in the rounds must not leave resting quotes on the book
            self._cancel_verified()
        st = self.stats.copy()
        n_fills = n_maker_filled + n_market; tot_notional = maker_notional + market_notional
        st.update(exec_seconds=round(time.time() - t_start, 1), last_fill_seconds=round(t_last_fill - t_start, 1), legs=len(want),
                  maker_fill_share=round(n_maker_filled / n_fills, 3) if n_fills else None,                        # by legs (proxy)
                  maker_notional_share=round(maker_notional / tot_notional, 3) if tot_notional > 0 else None,      # by notional (proxy)
                  market_legs=n_market, unfilled_legs=n_unfilled, unfilled_notional=round(unfilled_usd, 0), aborted=aborted,
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
    def flatten_all(self, instr, reason, quiet=False):
        """Close everything at market. quiet=True (the watchdog's repeated attempt while a halt is latched): no 'closing' message, and a failure
        message at most once an hour per reason instead of every attempt."""
        self._cancel_verified()
        try:
            poss = self.api.positions()
        except Exception as e:
            msg = f'🛑 Не смог прочитать позиции для закрытия ({reason}): {e}. ЗАКРОЙ ВРУЧНУЮ.'
            if quiet:
                self._alert('flatten_read', msg, 1.0)
            else:
                self.tg(msg)
            return -1
        if not poss:
            return 0
        self.log(f'FLATTEN ALL: {reason} ({len(poss)} symbols)')
        if not quiet:
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
            msg = '⚠️ НЕ ЗАКРЫЛИСЬ: ' + ', '.join(str(f) for f in fails[:10]) + '. Закрой вручную.'
            if quiet:
                self._alert('flatten_fail', msg, 1.0)
            else:
                self.tg(msg)
        return len(poss)
