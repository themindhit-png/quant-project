#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bybit v5 REST client (linear USDT perps, UTA). Market data is public; account/order calls signed."""
import time, hmac, hashlib, uuid, json
from urllib.parse import urlencode
import requests

RETRY_RETCODES = {10002, 10006, 10016, 10018}


class ApiError(RuntimeError):
    def __init__(self, code, msg):
        super().__init__(f'retCode {code}: {msg}')
        self.code = int(code or 0)


class Bybit:
    def __init__(self, base, key='', secret='', log=print):
        self.base = base.rstrip('/')
        self.key, self.secret = key, secret
        self.s = requests.Session()
        self.log = log
        self.n_req = 0
        self._last_t = {}                       # simple per-method pacing (Bybit: 10 order requests/s per UID, 120 GET/s)
        self.MIN_GAP = {'POST': 0.12, 'GET': 0.05}

    def _pace(self, method):
        gap = self.MIN_GAP.get(method, 0.05); dt = time.time() - self._last_t.get(method, 0.0)
        if dt < gap:
            time.sleep(gap - dt)
        self._last_t[method] = time.time()

    # ------------------------------------------------------------ core
    def _sign(self, ts, payload):
        raw = f'{ts}{self.key}20000{payload}'
        return hmac.new(self.secret.encode(), raw.encode(), hashlib.sha256).hexdigest()

    def _req(self, method, path, params=None, auth=False, retries=4, raw=False):
        params = params or {}
        last = None
        for att in range(retries):
            try:
                self._pace(method)
                self.n_req += 1
                ts = str(int(time.time() * 1000))
                if method == 'GET':
                    qs = urlencode(sorted(params.items()))
                    url = f'{self.base}{path}?{qs}' if qs else f'{self.base}{path}'
                    headers = {}
                    if auth:
                        headers = {'X-BAPI-API-KEY': self.key, 'X-BAPI-TIMESTAMP': ts, 'X-BAPI-RECV-WINDOW': '20000',
                                   'X-BAPI-SIGN': self._sign(ts, qs)}
                    r = self.s.get(url, headers=headers, timeout=20)
                else:
                    body = json.dumps(params)
                    headers = {'Content-Type': 'application/json'}
                    if auth:
                        headers.update({'X-BAPI-API-KEY': self.key, 'X-BAPI-TIMESTAMP': ts, 'X-BAPI-RECV-WINDOW': '20000',
                                        'X-BAPI-SIGN': self._sign(ts, body)})
                    r = self.s.post(f'{self.base}{path}', data=body, headers=headers, timeout=20)
                if r.status_code == 403:
                    raise RuntimeError(f'HTTP 403 (geo-block / IP ban?) at {path}')
                if r.status_code == 429:
                    time.sleep(1.0 + att); continue
                j = r.json()
                code = int(j.get('retCode', -1))
                if code == 0:
                    return j if raw else j.get('result', {})
                if code == 10006 and att < retries - 1:                 # rate limit -> back off
                    time.sleep(1.0 + att); continue
                if code in RETRY_RETCODES and att < retries - 1:
                    time.sleep(0.5 + att); continue
                raise ApiError(code, j.get('retMsg'))
            except ApiError:
                raise
            except Exception as e:
                last = e
                time.sleep(0.5 + att)
        raise RuntimeError(f'{path}: {last}')

    # ------------------------------------------------------------ market data (public)
    def instruments(self):
        out, cursor = {}, ''
        while True:
            p = {'category': 'linear', 'status': 'Trading', 'limit': 1000}
            if cursor:
                p['cursor'] = cursor
            r = self._req('GET', '/v5/market/instruments-info', p)
            for it in r.get('list', []):
                if it.get('settleCoin') != 'USDT' or it.get('contractType') != 'LinearPerpetual':
                    continue
                lf, pf = it.get('lotSizeFilter', {}), it.get('priceFilter', {})
                try:
                    out[it['symbol']] = dict(
                        qtyStep=float(lf.get('qtyStep') or 0), minQty=float(lf.get('minOrderQty') or 0),
                        maxQty=float(lf.get('maxMktOrderQty') or lf.get('maxOrderQty') or 1e12),
                        minNotional=max(float(lf.get('minNotionalValue') or 0), 5.0),
                        tickSize=float(pf.get('tickSize') or 0), launch=int(it.get('launchTime') or 0),
                        maxLeverage=float((it.get('leverageFilter') or {}).get('maxLeverage') or 0),
                        fundingInterval=int(it.get('fundingInterval') or 480))
                except (TypeError, ValueError):
                    continue
            cursor = r.get('nextPageCursor', '')
            if not cursor:
                return out

    def tickers(self):
        r = self._req('GET', '/v5/market/tickers', {'category': 'linear'})
        out = {}
        for t in r.get('list', []):
            def f(k):
                try:
                    return float(t.get(k) or 0)
                except (TypeError, ValueError):
                    return 0.0
            out[t['symbol']] = dict(last=f('lastPrice'), mark=f('markPrice'), index=f('indexPrice'),
                                    bid=f('bid1Price'), ask=f('ask1Price'), turnover24h=f('turnover24h'),
                                    funding=f('fundingRate'), oi_value=f('openInterestValue'))
        return out

    def klines(self, symbol, interval='60', limit=1000, start=None, end=None):
        """Returns rows [start_ms, open, high, low, close, volume, turnover] ascending."""
        p = {'category': 'linear', 'symbol': symbol, 'interval': interval, 'limit': min(limit, 1000)}
        if start: p['start'] = int(start)
        if end: p['end'] = int(end)
        r = self._req('GET', '/v5/market/kline', p)
        rows = [[int(x[0])] + [float(v) for v in x[1:7]] for x in r.get('list', [])]
        rows.sort(key=lambda x: x[0])
        return rows

    def klines_long(self, symbol, need_bars, interval='60'):
        rows, end = [], None
        while len(rows) < need_bars:
            chunk = self.klines(symbol, interval, 1000, end=end)
            if not chunk:
                break
            rows = chunk + rows
            if len(chunk) < 1000:
                break
            end = chunk[0][0] - 1
        seen, out = set(), []
        for r in rows:
            if r[0] not in seen:
                seen.add(r[0]); out.append(r)
        out.sort(key=lambda x: x[0])
        return out[-need_bars:]

    def premium_klines_long(self, symbol, need_bars, interval='60'):
        rows, end = [], None
        while len(rows) < need_bars:
            p = {'category': 'linear', 'symbol': symbol, 'interval': interval, 'limit': 1000}
            if end: p['end'] = int(end)
            r = self._req('GET', '/v5/market/premium-index-price-kline', p)
            chunk = [[int(x[0])] + [float(v) for v in x[1:5]] for x in r.get('list', [])]
            chunk.sort(key=lambda x: x[0])
            if not chunk:
                break
            rows = chunk + rows
            if len(chunk) < 1000:
                break
            end = chunk[0][0] - 1
        seen, out = set(), []
        for r in rows:
            if r[0] not in seen:
                seen.add(r[0]); out.append(r)
        out.sort(key=lambda x: x[0])
        return out[-need_bars:]

    def funding_history(self, symbol, start_ms, end_ms):
        """List of (ts_ms, rate) between start and end (paginated backwards, 200 per call)."""
        out, end = [], end_ms
        for _ in range(20):
            r = self._req('GET', '/v5/market/funding/history', {'category': 'linear', 'symbol': symbol,
                                                                 'startTime': int(start_ms), 'endTime': int(end), 'limit': 200})
            lst = r.get('list', [])
            if not lst:
                break
            for it in lst:
                out.append((int(it['fundingRateTimestamp']), float(it['fundingRate'])))
            oldest = min(int(it['fundingRateTimestamp']) for it in lst)
            if oldest <= start_ms or len(lst) < 200:
                break
            end = oldest - 1
        return sorted(set(out))

    def announcements_delisting(self):
        """Symbols mentioned in recent delisting announcements (best effort)."""
        try:
            r = self._req('GET', '/v5/announcements/index', {'locale': 'en-US', 'type': 'delistings', 'limit': 50})
        except Exception:
            return set()
        syms = set()
        for it in r.get('list', []):
            title = (it.get('title') or '') + ' ' + (it.get('description') or '')
            for w in title.replace(',', ' ').replace('/', ' ').split():
                w = w.strip().upper()
                if w.endswith('USDT') and len(w) > 5:
                    syms.add(w)
                elif w.endswith('PERP') or w.endswith('PERPETUAL'):
                    pass
        return syms

    # ------------------------------------------------------------ account (signed)
    def wallet(self):
        r = self._req('GET', '/v5/account/wallet-balance', {'accountType': 'UNIFIED'}, auth=True)
        lst = r.get('list') or []
        if not lst:
            raise RuntimeError('wallet-balance: empty')
        acct = lst[0]
        eq_usdt = None
        for c in (acct.get('coin') or []):
            if c.get('coin') == 'USDT':
                try:
                    eq_usdt = float(c.get('equity') or 0)
                except (TypeError, ValueError):
                    pass
        return dict(equity=eq_usdt if eq_usdt else float(acct.get('totalEquity') or 0),
                    total_equity=float(acct.get('totalEquity') or 0),
                    im=float(acct.get('totalInitialMargin') or 0),
                    mm=float(acct.get('totalMaintenanceMargin') or 0),
                    available=float(acct.get('totalAvailableBalance') or 0))

    def equity(self):
        return self.wallet()['equity']

    def positions(self):
        """{symbol: {qty (signed, net), mark, entry, upnl, legs:[...]}}"""
        raw, cursor = {}, ''
        while True:
            p = {'category': 'linear', 'settleCoin': 'USDT', 'limit': 200}
            if cursor:
                p['cursor'] = cursor
            r = self._req('GET', '/v5/position/list', p, auth=True)
            for pos in r.get('list', []):
                sz = float(pos.get('size') or 0)
                if sz == 0:
                    continue
                sign = 1 if pos.get('side') == 'Buy' else -1
                def f(k, _p=pos):
                    try:
                        return float(_p.get(k) or 0)
                    except (TypeError, ValueError):
                        return 0.0
                leg = dict(qty=sign * sz, positionIdx=int(pos.get('positionIdx') or 0), mark=f('markPrice'),
                           entry=f('avgPrice'), upnl=f('unrealisedPnl'), value=f('positionValue'), im=f('positionIM'),
                           sl=str(pos.get('stopLoss') or ''))
                raw.setdefault(pos['symbol'], []).append(leg)
            cursor = r.get('nextPageCursor', '')
            if not cursor:
                break
        out = {}
        for sym, legs in raw.items():
            q = sum(l['qty'] for l in legs)
            mark = next((l['mark'] for l in legs if l['mark']), 0.0)
            ent = sum(abs(l['qty']) * l['entry'] for l in legs) / max(sum(abs(l['qty']) for l in legs), 1e-12)
            out[sym] = dict(qty=q, mark=mark, entry=ent, upnl=sum(l['upnl'] for l in legs),
                            im=sum(l['im'] for l in legs), legs=legs)
        return out

    def funding_since(self, start_ms):
        """(sum, count) of funding settlements in the account's transaction log since start_ms (type SETTLEMENT / non-zero
        'funding' field). This is how 'does this account accrue funding?' is answered (PREREG rule 6)."""
        total, n, cursor = 0.0, 0, ''
        while True:
            p = {'accountType': 'UNIFIED', 'startTime': int(start_ms), 'limit': 50}
            if cursor:
                p['cursor'] = cursor
            r = self._req('GET', '/v5/account/transaction-log', p, auth=True)
            for it in r.get('list', []):
                t = str(it.get('type') or '').upper()
                try:
                    f = float(it.get('funding') or 0.0)
                except (TypeError, ValueError):
                    f = 0.0
                if t == 'SETTLEMENT' or f != 0.0:
                    total += f if f != 0.0 else float(it.get('change') or 0.0); n += 1
            cursor = r.get('nextPageCursor', '')
            if not cursor:
                return total, n

    def transaction_log_sum(self, start_ms):
        total, cursor = 0.0, ''
        while True:
            p = {'accountType': 'UNIFIED', 'startTime': int(start_ms), 'limit': 50}
            if cursor:
                p['cursor'] = cursor
            r = self._req('GET', '/v5/account/transaction-log', p, auth=True)
            for it in r.get('list', []):
                t = str(it.get('type') or '').upper()
                if 'TRANSFER' in t or t == 'BONUS':
                    continue
                try:
                    total += float(it.get('change') or 0)
                except (TypeError, ValueError):
                    pass
            cursor = r.get('nextPageCursor', '')
            if not cursor:
                return total

    # ------------------------------------------------------------ orders (signed)
    @staticmethod
    def oid():
        return 'ps4-' + uuid.uuid4().hex[:20]

    def create(self, p):
        p = dict(p); p['orderLinkId'] = self.oid()
        try:
            return self._req('POST', '/v5/order/create', p, auth=True)
        except ApiError as e:
            if e.code == 110072 or 'duplicate' in str(e).lower():
                return {'duplicate': True}
            raise

    def create_batch(self, orders):
        """Batch order creation (/v5/order/create-batch, <=10 per request). Returns a list of (retCode, retMsg) per order,
        in input order; a transport-level failure raises so the caller can fall back to single orders."""
        out = []
        for k in range(0, len(orders), 10):
            chunk = []
            for p in orders[k:k + 10]:
                p = dict(p); p['orderLinkId'] = self.oid(); chunk.append(p)
            j = self._req('POST', '/v5/order/create-batch', {'category': 'linear', 'request': chunk}, auth=True, retries=2, raw=True)
            codes = ((j.get('retExtInfo') or {}).get('list') or [])
            for n in range(len(chunk)):
                c = codes[n] if n < len(codes) else {}
                out.append((int(c.get('code', 0) or 0), str(c.get('msg', ''))))
        return out

    def executions(self, start_ms, limit=100):
        """Own fills since start_ms (/v5/execution/list): list of dicts with symbol, side, execQty, execPrice, isMaker, execFee."""
        out, cursor = [], ''
        while True:
            p = {'category': 'linear', 'startTime': int(start_ms), 'limit': limit}
            if cursor:
                p['cursor'] = cursor
            r = self._req('GET', '/v5/execution/list', p, auth=True, retries=2)
            out.extend(r.get('list') or [])
            cursor = r.get('nextPageCursor', '')
            if not cursor or len(out) >= 2000:
                return out

    def market(self, symbol, qty, side, reduce_only=False, position_idx=0):
        p = {'category': 'linear', 'symbol': symbol, 'side': side, 'orderType': 'Market', 'qty': str(qty),
             'positionIdx': position_idx}
        if reduce_only:
            p['reduceOnly'] = True
        return self.create(p)

    def post_only(self, symbol, qty, side, price_str, reduce_only=False, position_idx=0):
        p = {'category': 'linear', 'symbol': symbol, 'side': side, 'orderType': 'Limit', 'qty': str(qty),
             'price': price_str, 'timeInForce': 'PostOnly', 'positionIdx': position_idx}
        if reduce_only:
            p['reduceOnly'] = True
        return self.create(p)

    def cancel_all(self, symbol=None):
        p = {'category': 'linear'}
        if symbol:
            p['symbol'] = symbol
        else:
            p['settleCoin'] = 'USDT'
        try:
            return self._req('POST', '/v5/order/cancel-all', p, auth=True, retries=2)
        except ApiError as e:
            self.log(f'cancel-all: {e}')
            return None

    def open_orders(self):
        out, cursor = [], ''
        while True:
            p = {'category': 'linear', 'settleCoin': 'USDT', 'limit': 50}
            if cursor:
                p['cursor'] = cursor
            r = self._req('GET', '/v5/order/realtime', p, auth=True)
            out.extend(r.get('list') or [])
            cursor = r.get('nextPageCursor', '')
            if not cursor:
                return out

    def set_trading_stop(self, symbol, stop_loss_price_str, position_idx=0):
        """Position-level stop loss via /v5/position/trading-stop (Bybit 'TP/SL' — the form some firms require on every
        position). Mark-price trigger, full position ('Full' tpslMode)."""
        return self._req('POST', '/v5/position/trading-stop',
                         {'category': 'linear', 'symbol': symbol, 'stopLoss': str(stop_loss_price_str), 'slTriggerBy': 'MarkPrice',
                          'tpslMode': 'Full', 'positionIdx': position_idx}, auth=True, retries=2)

    def set_leverage(self, symbol, lev):
        s = str(int(lev)) if float(lev).is_integer() else str(lev)
        return self._req('POST', '/v5/position/set-leverage',
                         {'category': 'linear', 'symbol': symbol, 'buyLeverage': s, 'sellLeverage': s}, auth=True, retries=1)

    def switch_one_way(self):
        return self._req('POST', '/v5/position/switch-mode', {'category': 'linear', 'coin': 'USDT', 'mode': 0},
                         auth=True, retries=1)
