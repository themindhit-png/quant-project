#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Market data cache for PROP-SLEEVES v4: hourly close/high/low/turnover + funding + premium index per
symbol, aligned on a common hourly grid (hours since epoch). Incremental refresh via Bybit v5 public API.
Also token age from Bybit launchTime and (optionally) Binance onboardDate (two-clock age)."""
import time, math
import numpy as np, requests

NEED_H = 2200            # hours of history kept (ML needs 2161)


class MarketData:
    def __init__(self, api, log=print, use_binance_clock=True):
        self.api = api
        self.log = log
        self.bars = {}          # sym -> dict(h=np.int64[], close, high, low, turn) ascending, closed bars only
        self.fund = {}          # sym -> dict(h -> rate)
        self.prem = {}          # sym -> dict(h=np.int64[], close)
        self.instr = {}
        self.instr_ts = 0.0
        self.binance_onboard = {}   # sym -> ms
        self.binance_ts = 0.0
        self.use_binance_clock = use_binance_clock
        self.delist = set()
        self.delist_ts = 0.0

    # ------------------------------------------------------------ instruments / clocks
    def refresh_instruments(self, force=False):
        if force or time.time() - self.instr_ts > 6 * 3600:
            self.instr = self.api.instruments()
            self.instr_ts = time.time()
            self.log(f'instruments: {len(self.instr)}')
        if self.use_binance_clock and time.time() - self.binance_ts > 24 * 3600:
            try:
                r = requests.get('https://fapi.binance.com/fapi/v1/exchangeInfo', timeout=20)
                if r.status_code == 200:
                    for s in r.json().get('symbols', []):
                        if s.get('contractType') == 'PERPETUAL' and s.get('quoteAsset') == 'USDT':
                            self.binance_onboard[s['symbol']] = int(s.get('onboardDate') or 0)
                    self.log(f'binance onboard dates: {len(self.binance_onboard)}')
                else:
                    self.log(f'binance exchangeInfo HTTP {r.status_code} — Bybit clock only')
            except Exception as e:
                self.log(f'binance exchangeInfo unavailable ({e}) — Bybit clock only')
            self.binance_ts = time.time()
        if time.time() - self.delist_ts > 6 * 3600:
            self.delist = self.api.announcements_delisting()
            self.delist_ts = time.time()
            if self.delist:
                self.log(f'delisting announcements mention: {sorted(self.delist)[:10]}')

    def refresh_mcap(self, max_age_s=6 * 3600):
        """Market caps (USD) by upper-case base symbol from CoinGecko's public API (top 1000 coins, 4 pages). Cached
        for max_age_s; returns the dict or None when unavailable (callers then fall back to the turnover proxy)."""
        now = time.time()
        if getattr(self, 'mcap', None) is not None and now - getattr(self, 'mcap_ts', 0) < max_age_s:
            return self.mcap
        out = {}
        try:
            for page in range(1, 5):
                r = requests.get('https://api.coingecko.com/api/v3/coins/markets',
                                 params=dict(vs_currency='usd', order='market_cap_desc', per_page=250, page=page, sparkline='false'), timeout=20)
                if r.status_code != 200:
                    raise RuntimeError(f'HTTP {r.status_code}')
                for c in r.json():
                    s = str(c.get('symbol') or '').upper(); mc = c.get('market_cap')
                    if s and mc:
                        out[s] = max(float(mc), out.get(s, 0.0))       # duplicate tickers: keep the largest (the one an exchange lists)
                time.sleep(1.5)
        except Exception as e:
            self.log(f'mcap refresh failed: {e}')
            if getattr(self, 'mcap', None) is not None:
                return self.mcap                                          # stale cache is better than nothing
            return None
        self.mcap, self.mcap_ts = out, now
        self.log(f'mcap refreshed: {len(out)} coins')
        return out

    def binance_funding_24h(self, symbols, max_n=20):
        """Sum of the last 24h funding rates on Binance USDT-M for the given symbols (best effort, public endpoint) — used only
        as a diagnostic of the Bybit-vs-Binance funding basis (the research was measured on Binance rates)."""
        out = {}
        for s in symbols[:max_n]:
            try:
                r = requests.get('https://fapi.binance.com/fapi/v1/fundingRate', params={'symbol': s, 'limit': 6}, timeout=10)
                if r.status_code != 200:
                    continue
                now_ms = int(time.time() * 1000)
                out[s] = float(sum(float(x['fundingRate']) for x in r.json() if now_ms - int(x['fundingTime']) <= 24 * 3600 * 1000))
                time.sleep(0.05)
            except Exception:
                continue
        return out

    @staticmethod
    def base_symbol(sym):
        """'1000PEPEUSDT' -> 'PEPE', 'BTCUSDT' -> 'BTC'."""
        s = sym[:-4] if sym.endswith('USDT') else sym
        return s.lstrip('0123456789') or s

    def age_hours(self, sym, now_ms, mode='listing'):
        """Token age in hours. mode='listing': since the EARLIEST known listing (Bybit launchTime or
        Binance onboardDate) — for the listing-drift sleeve. mode='ml': Binance onboardDate when known
        (the ML models were trained with Binance listing age), else Bybit launchTime."""
        m = self.instr.get(sym)
        by = m['launch'] if (m and m.get('launch')) else None
        bn = self.binance_onboard.get(sym) or None
        if mode == 'ml':
            ref = bn or by
        else:
            cands = [x for x in (by, bn) if x]
            ref = min(cands) if cands else None
        if not ref:
            return -1.0
        return (now_ms - ref) / 3.6e6

    # ------------------------------------------------------------ bars
    @staticmethod
    def last_closed_hour_ms(now_ms=None):
        now_ms = now_ms or int(time.time() * 1000)
        return (now_ms // 3_600_000 - 1) * 3_600_000       # start of the last fully closed hourly bar

    def refresh_symbol(self, sym, last_closed_ms, need_h=NEED_H):
        """Ensure bars/funding/premium for sym are complete up to last_closed_ms."""
        b = self.bars.get(sym)
        if b is None or len(b['h']) == 0:
            rows = self.api.klines_long(sym, need_h)
            rows = [r for r in rows if r[0] <= last_closed_ms]
            if len(rows) < 48:
                return False
            self.bars[sym] = dict(h=np.array([r[0] // 3_600_000 for r in rows], dtype=np.int64),
                                  close=np.array([r[4] for r in rows]), high=np.array([r[2] for r in rows]),
                                  low=np.array([r[3] for r in rows]), turn=np.array([r[6] for r in rows]))
            self._refresh_funding(sym, rows[0][0], last_closed_ms + 3_600_000)
            self._refresh_premium(sym, need_h, last_closed_ms)
            return True
        last_h = int(b['h'][-1])
        want_h = last_closed_ms // 3_600_000
        if want_h <= last_h:
            return True
        n_new = min(want_h - last_h + 2, 1000)
        rows = self.api.klines(sym, '60', n_new, start=(last_h - 1) * 3_600_000, end=last_closed_ms + 3_599_999)
        rows = [r for r in rows if r[0] <= last_closed_ms and r[0] // 3_600_000 > last_h]
        if rows:
            for k, col in (('close', 4), ('high', 2), ('low', 3), ('turn', 6)):
                b[k] = np.concatenate((b[k], [r[col] for r in rows]))[-need_h:]
            b['h'] = np.concatenate((b['h'], [r[0] // 3_600_000 for r in rows]))[-need_h:]
        self._refresh_funding(sym, (last_h - 48) * 3_600_000, last_closed_ms + 3_600_000)
        self._refresh_premium(sym, min(n_new + 2, 1000), last_closed_ms, incremental=True)
        return True

    def _refresh_funding(self, sym, start_ms, end_ms):
        try:
            for ts, rate in self.api.funding_history(sym, start_ms, end_ms):
                self.fund.setdefault(sym, {})[ts // 3_600_000] = rate
        except Exception as e:
            self.log(f'funding {sym}: {e}')

    def _refresh_premium(self, sym, need, last_closed_ms, incremental=False):
        try:
            rows = self.api.premium_klines_long(sym, need)
            rows = [r for r in rows if r[0] <= last_closed_ms]
            if not rows:
                return
            p = self.prem.get(sym)
            h = np.array([r[0] // 3_600_000 for r in rows], dtype=np.int64)
            c = np.array([r[4] for r in rows])
            if p is None or not incremental:
                self.prem[sym] = dict(h=h, close=c)
            else:
                m = h > p['h'][-1]
                if m.any():
                    self.prem[sym] = dict(h=np.concatenate((p['h'], h[m]))[-NEED_H:], close=np.concatenate((p['close'], c[m]))[-NEED_H:])
        except Exception as e:
            self.log(f'premium {sym}: {e}')

    def refresh(self, symbols, last_closed_ms, sleep=0.03):
        ok, bad = [], []
        for s in symbols:
            try:
                (ok if self.refresh_symbol(s, last_closed_ms) else bad).append(s)
            except Exception as e:
                bad.append(s); self.log(f'bars {s}: {e}')
            time.sleep(sleep)
        # drop symbols no longer needed to bound memory
        for s in list(self.bars):
            if s not in symbols:
                self.bars.pop(s, None); self.fund.pop(s, None); self.prem.pop(s, None)
        return ok, bad

    # ------------------------------------------------------------ aligned panel
    def panel(self, symbols, last_closed_ms, need_h=NEED_H):
        """Aligned arrays (need_h, N) ending at the last closed hour: cff, high, low, t24, fund, prem, valid,
        plus per-symbol data-age (hours of available data) and 'stale' flag."""
        end_h = last_closed_ms // 3_600_000
        grid = np.arange(end_h - need_h + 1, end_h + 1, dtype=np.int64)
        N = len(symbols)
        close = np.full((need_h, N), np.nan); high = np.full((need_h, N), np.nan); low = np.full((need_h, N), np.nan)
        turn = np.full((need_h, N), np.nan); fund = np.zeros((need_h, N)); prem = np.full((need_h, N), np.nan)
        stale = np.zeros(N, dtype=bool); data_age = np.zeros(N)
        for j, s in enumerate(symbols):
            b = self.bars.get(s)
            if b is None or len(b['h']) == 0:
                stale[j] = True; continue
            pos = b['h'] - grid[0]
            m = (pos >= 0) & (pos < need_h)
            close[pos[m], j] = b['close'][m]; high[pos[m], j] = b['high'][m]; low[pos[m], j] = b['low'][m]; turn[pos[m], j] = b['turn'][m]
            stale[j] = (end_h - int(b['h'][-1])) > 8
            data_age[j] = float(end_h - int(b['h'][0]))
            for hh_, rate in self.fund.get(s, {}).items():
                p = hh_ - grid[0]
                if 0 <= p < need_h:
                    fund[p, j] += rate
            pr = self.prem.get(s)
            if pr is not None:
                pos = pr['h'] - grid[0]; m = (pos >= 0) & (pos < need_h)
                prem[pos[m], j] = pr['close'][m]
        valid = np.isfinite(close) & (close > 0)
        # forward fill closes
        cff = close.copy()
        for j in range(N):
            col = cff[:, j]
            idx = np.where(np.isfinite(col), np.arange(need_h), 0)
            np.maximum.accumulate(idx, out=idx)
            col2 = col[idx]; col2[~np.isfinite(col[0]) & (idx == 0)] = np.nan
            cff[:, j] = col2
        t24 = np.full_like(turn, np.nan)
        tt = np.where(np.isfinite(turn), turn, 0.0)
        cs = np.cumsum(tt, axis=0)
        t24[23:] = cs[23:] - np.concatenate((np.zeros((1, N)), cs[:-24]), axis=0)[:need_h - 23]
        return dict(grid=grid, cff=cff, high=high, low=low, t24=t24, fund=fund, prem=prem, valid=valid,
                    stale=stale, data_age=data_age, symbols=list(symbols))
