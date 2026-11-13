#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Market-wide liquidation CASCADE FILTER (registration 5b).

Rule: every UTC day D the market intensity C_D = sum of daily liquidations (long + short, USD) / sum of open interest (USD) over the
bot universe on Binance + Bybit USDT perps (free Coinalyze API) is compared with the 95th percentile of the trailing 365 days
(day D included, at least 180 valid days). C_D above it = trigger day. The book is scaled by CASCADE_MULT (0.5) on the next
CASCADE_DAYS (2) days, D+1 and D+2; otherwise 1.0. A decision on day X therefore uses only data through X-1.
Any data problem (no key, API down, incomplete day, stale cache) -> 1.0 and a single Telegram warning; trading is never blocked.

The API key is read ONLY from the environment variable COINALYZE_API_KEY at call time; it is never stored, logged, cached or hashed.
Pure functions (series / trigger / multiplier) have no side effects and are shared with the research engine (exp_common.cascade_wrap).

CLI (needs COINALYZE_API_KEY and the bot env, run from the VPS):  python3 v4/cascade.py backfill | status | export OUT.csv
"""
import os, sys, json, time, math, threading
from collections import deque
from datetime import datetime, timezone, timedelta

BASE_URL = 'https://api.coinalyze.net/v1/'
KEY_ENV = 'COINALYZE_API_KEY'
DAY = timedelta(days=1)


# ------------------------------------------------------------------ pure logic (shared with research)
def day_str(d):
    return d.isoformat()


def intensity_series(liq, oi):
    """liq, oi: {code: {'YYYY-MM-DD': usd}}. Returns {day: C} with C = sum(liq)/sum(oi) over the codes present that day (NaN-free:
    missing values are skipped, like the research sum with fill 0); days without positive OI are omitted."""
    L, O = {}, {}
    for src, dst in ((liq, L), (oi, O)):
        for per in src.values():
            for dd, v in per.items():
                if v is not None and math.isfinite(v):
                    dst[dd] = dst.get(dd, 0.0) + float(v)
    return {dd: L.get(dd, 0.0) / O[dd] for dd in sorted(O) if O[dd] > 0 and dd in L}


def threshold_series(C, pct=0.95, win=365, min_n=180):
    """Trailing threshold per day: pct-quantile (linear interpolation, as pandas) of the valid C over the `win` calendar days ending at
    the day itself. None when fewer than min_n valid days. Returns {day: thr|None}."""
    import numpy as np
    if not C:
        return {}
    days = sorted(C)
    d0 = datetime.strptime(days[0], '%Y-%m-%d').date(); d1 = datetime.strptime(days[-1], '%Y-%m-%d').date()
    n = (d1 - d0).days + 1
    arr = np.full(n, np.nan)
    for dd, v in C.items():
        arr[(datetime.strptime(dd, '%Y-%m-%d').date() - d0).days] = v
    out = {}
    for k in range(n):
        w = arr[max(0, k - win + 1):k + 1]; w = w[np.isfinite(w)]
        out[(d0 + timedelta(days=k)).isoformat()] = float(np.quantile(w, pct)) if len(w) >= min_n else None
    return out


def trigger_days(C, pct=0.95, win=365, min_n=180):
    """Set of trigger days (C > threshold, threshold known)."""
    thr = threshold_series(C, pct, win, min_n)
    return {dd for dd, v in C.items() if thr.get(dd) is not None and v > thr[dd]}


def multiplier_for(day, trig, days=2, mult=0.5):
    """Book multiplier for UTC date `day` (date or 'YYYY-MM-DD'): `mult` if one of the `days` preceding days is a trigger day, else 1.0."""
    if isinstance(day, str):
        day = datetime.strptime(day, '%Y-%m-%d').date()
    for k in range(1, int(days) + 1):
        if day_str(day - timedelta(days=k)) in trig:
            return float(mult)
    return 1.0


# ------------------------------------------------------------------ HTTP / throttle
class CascadeError(Exception):
    pass


class RateLimiter:
    """At most `per_min` symbol-calls in any 60 s window (a request for n symbols counts n)."""
    def __init__(self, per_min=38, clock=time.time, sleep=time.sleep):
        self.per_min, self.clock, self.sleep, self.q = per_min, clock, sleep, deque()

    def acquire(self, n=1):
        n = min(max(1, n), self.per_min)
        while True:
            now = self.clock()
            while self.q and now - self.q[0] >= 60.0:
                self.q.popleft()
            if len(self.q) + n <= self.per_min:
                for _ in range(n):
                    self.q.append(now)
                return
            self.sleep(max(0.2, 60.0 - (now - self.q[0]) + 0.05))


def _http_get(path, params, key, timeout=60):
    """-> (status, json|None, retry_after). The key travels in a header only."""
    import requests
    r = requests.get(BASE_URL + path, params=params, headers={'api_key': key, 'User-Agent': 'propsleeves-cascade'}, timeout=timeout)
    try:
        body = r.json()
    except Exception:
        body = None
    try:
        ra = float(r.headers.get('Retry-After', '5') or 5)
    except Exception:
        ra = 5.0
    return r.status_code, body, ra


class CascadeFilter:
    def __init__(self, pct=0.95, mult=0.5, days=2, win=365, min_n=180, exch=('A', '6'), nuni=150, directory='', backfill_d=400,
                 per_min=38, http=None, clock=time.time, sleep=time.sleep, notify=None, log=print, universe_fn=None, tries=5):
        self.pct, self.mult, self.days, self.win, self.min_n = float(pct), float(mult), int(days), int(win), int(min_n)
        self.exch, self.nuni, self.backfill_d, self.tries = tuple(exch), int(nuni), int(backfill_d), int(tries)
        self.dir = directory; self.path = os.path.join(directory, 'cascade_cache.json')
        self.http = http or _http_get; self.clock, self.sleep = clock, sleep
        self.rl = RateLimiter(per_min, clock, sleep); self.notify = notify or (lambda t: None); self.log = log
        self.universe_fn = universe_fn; self.lock = threading.Lock(); self.cache = None; self.running = False; self.last_try = 0.0

    # ---- persistence
    def _load(self):
        if self.cache is None:
            try:
                self.cache = json.load(open(self.path, encoding='utf-8'))
            except Exception:
                self.cache = {}
            for k, v in (('liq', {}), ('oi', {}), ('series', {}), ('universe', []), ('codes', {}), ('uni_ts', 0), ('warn', False), ('last_ok', ''), ('err', '')):
                self.cache.setdefault(k, v)
        return self.cache

    def _save(self):
        os.makedirs(self.dir, exist_ok=True)
        tmp = self.path + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(self.cache, f, separators=(',', ':'))
        os.replace(tmp, self.path)

    def _clean(self, s):
        k = os.environ.get(KEY_ENV, '')
        s = str(s)
        return s.replace(k, '***') if k else s

    # ---- API
    def _get(self, path, params, nsym=1):
        key = os.environ.get(KEY_ENV, '').strip()
        if not key:
            raise CascadeError(f'нет переменной {KEY_ENV}')
        last = 'retries'
        for i in range(self.tries):
            self.rl.acquire(nsym)
            try:
                st, body, ra = self.http(path, params, key)
            except Exception as e:                                        # network: message may not carry the key, but clean anyway
                last = f'сеть: {type(e).__name__}'; self.sleep(5 * (i + 1)); continue
            if st == 200 and body is not None:
                return body
            if st == 429:
                last = '429'; self.sleep(min(ra, 90.0) + 1 + 2 * i); continue
            if st in (500, 502, 503, 504):
                last = f'HTTP {st}'; self.sleep(5 * (i + 1)); continue
            raise CascadeError(f'HTTP {st}')                              # 400/401/403/404: retrying will not help
        raise CascadeError(last)

    def _history(self, endpoint, field_fn, codes, frm, to):
        """One request for <= 20 codes -> {code: {day: value}} (only complete days, t < today 00:00 UTC)."""
        extra = {'convert_to_usd': 'true'}
        body = self._get(endpoint, {'symbols': ','.join(codes), 'interval': 'daily', 'from': int(frm), 'to': int(to), **extra}, nsym=len(codes))
        if not isinstance(body, list):
            raise CascadeError('неожиданный ответ API')
        out = {}
        today0 = datetime.fromtimestamp(to, timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
        for it in body:
            per = {}
            for h in it.get('history', []) or []:
                t = h.get('t')
                if t is None or t >= today0:
                    continue
                v = field_fn(h)
                if v is not None and math.isfinite(v):
                    per[datetime.fromtimestamp(t, timezone.utc).date().isoformat()] = float(v)
            out[it.get('symbol')] = per
        return out

    @staticmethod
    def _liq(h):
        return (float(h['l']) + float(h['s'])) if h.get('l') is not None and h.get('s') is not None else None

    @staticmethod
    def _oi(h):
        return float(h['c']) if h.get('c') is not None else None

    # ---- universe -> Coinalyze codes
    def _refresh_universe(self, c, now):
        if not self.universe_fn:
            return
        if c['codes'] and now - c['uni_ts'] < 30 * 86400:
            return
        syms = list(self.universe_fn())[:self.nuni]
        if not syms:
            raise CascadeError('пустая вселенная')
        fm = self._get('future-markets', {})
        idx = {}
        for x in fm:
            if x.get('is_perpetual') and x.get('quote_asset') == 'USDT' and x.get('exchange') in self.exch:
                idx[(x['exchange'], x['symbol_on_exchange'])] = x['symbol']
        codes = {}
        for ex in self.exch:
            codes[ex] = {s: idx[(ex, s)] for s in syms if (ex, s) in idx}
        if sum(len(v) for v in codes.values()) < 20:
            raise CascadeError('вселенная не сопоставилась с Coinalyze')
        c['universe'] = syms; c['codes'] = codes; c['uni_ts'] = now; self.log(f'cascade: вселенная {len(syms)} монет, кодов {sum(len(v) for v in codes.values())}')

    def all_codes(self):
        c = self._load()
        return sorted({code for ex in c['codes'].values() for code in ex.values()})

    # ---- daily update
    def update(self, now=None):
        """Bring the cache up to yesterday (UTC). Returns True on success. Never raises."""
        now_dt = now or datetime.now(timezone.utc)
        with self.lock:
            self.running = True; self.last_try = self.clock()
            try:
                c = self._load()
                yest = day_str(now_dt.date() - DAY)
                if c['last_ok'] >= yest:
                    return True
                ts = now_dt.timestamp()
                self._refresh_universe(c, ts)
                codes = self.all_codes()
                if not codes:
                    raise CascadeError('нет кодов')
                have = [x for x in codes if c['oi'].get(x) and c['liq'].get(x)]
                new = [x for x in codes if x not in have]
                groups = []
                if new:
                    groups.append((new, ts - self.backfill_d * 86400))
                if have:
                    last = min(max(c['oi'][x]) for x in have)
                    frm = datetime.strptime(last, '%Y-%m-%d').replace(tzinfo=timezone.utc).timestamp() - 2 * 86400
                    frm = max(frm, ts - self.backfill_d * 86400)
                    groups.append((have, frm))
                for grp, frm in groups:
                    for i in range(0, len(grp), 20):
                        chunk = grp[i:i + 20]
                        for ep, fn, key in (('liquidation-history', self._liq, 'liq'), ('open-interest-history', self._oi, 'oi')):
                            r = self._history(ep, fn, chunk, frm, ts)
                            for code, per in r.items():
                                if code in chunk and per:
                                    c[key].setdefault(code, {}).update(per)
                # completeness of yesterday: enough names with both liquidations and OI
                ref = sorted(x for x in codes if c['oi'].get(x, {}).get(day_str(now_dt.date() - 3 * DAY)) is not None)
                got = [x for x in codes if yest in c['oi'].get(x, {}) and yest in c['liq'].get(x, {})]
                need = max(20, int(0.9 * len(ref)))
                if len(got) < need:
                    raise CascadeError(f'за {yest} данных только по {len(got)} из ожидаемых {need}')
                self._prune(c)
                C = intensity_series({x: c['liq'][x] for x in codes if x in c['liq']}, {x: c['oi'][x] for x in codes if x in c['oi']})
                thr = threshold_series(C, self.pct, self.win, self.min_n)
                tr = {dd for dd, v in C.items() if thr.get(dd) is not None and v > thr[dd]}
                c['series'] = {dd: [round(v, 8), (round(thr[dd], 8) if thr.get(dd) is not None else None), dd in tr] for dd, v in C.items()}
                if yest not in c['series'] or c['series'][yest][1] is None:
                    raise CascadeError(f'история короче {self.min_n} дн. — порог ещё не определён')
                c['last_ok'] = yest; c['err'] = ''
                was_warn = c['warn']; c['warn'] = False
                self._save()
                self.log(f'cascade: данные до {yest}, C={c["series"][yest][0]:.5f}, порог {c["series"][yest][1]:.5f}, триггер: {c["series"][yest][2]}')
                if was_warn:
                    self.notify('✅ Каскад-фильтр: данные Coinalyze снова доступны.')
                return True
            except Exception as e:
                msg = self._clean(e)
                c = self._load(); c['err'] = msg[:200]
                self.log(f'cascade: ОШИБКА обновления ({msg}); множитель 1.0')
                if not c.get('warn'):
                    c['warn'] = True
                    try:
                        self._save()
                    except Exception:
                        pass
                    self.notify(f'⚠️ Каскад-фильтр: нет данных Coinalyze ({msg[:120]}). Множитель книги 1.0, торговля не блокируется.')
                return False
            finally:
                self.running = False

    def _prune(self, c, keep_d=800):
        cut = day_str(datetime.now(timezone.utc).date() - timedelta(days=keep_d))
        for key in ('liq', 'oi'):
            for code in list(c[key]):
                per = c[key][code]
                for dd in [dd for dd in per if dd < cut]:
                    del per[dd]

    # ---- the multiplier the bot uses (reads the cache only; never calls the API)
    def multiplier(self, now=None):
        """-> (mult, reason). 1.0 whenever yesterday's value is missing or anything is wrong."""
        try:
            now_dt = now or datetime.now(timezone.utc)
            c = self._load(); day = now_dt.date(); yest = day_str(day - DAY)
            ser = c.get('series') or {}
            last_ok = c.get('last_ok', '')
            trig = {dd for dd, v in ser.items() if v[2]}
            if yest not in ser or last_ok < yest:
                # F10: the cache is 1-2 days behind (Coinalyze late / down at the slot): the trigger days we DO know still scale the book —
                # the window d-1, d-2 is applied as far as it is known (a missing day cannot be a known trigger). More than 2 days behind -> 1.0.
                lag = (datetime.strptime(yest, '%Y-%m-%d').date() - datetime.strptime(last_ok, '%Y-%m-%d').date()).days if last_ok else None
                if lag is not None and 1 <= lag <= 2:
                    m = multiplier_for(day, trig, self.days, self.mult)
                    if m < 1.0:
                        td = [dd for dd in (day_str(day - timedelta(days=k)) for k in range(1, self.days + 1)) if dd in trig]
                        return m, f'каскад ликвидаций {td[0]} → ×{m:g} (кэш отстаёт на {lag} дн., данные до {last_ok})'
                return 1.0, f'каскад: нет данных за {yest} → ×1.0'
            m = multiplier_for(day, trig, self.days, self.mult)
            if m < 1.0:
                td = [dd for dd in (day_str(day - timedelta(days=k)) for k in range(1, self.days + 1)) if dd in trig]
                return m, f'каскад ликвидаций {td[0]} → ×{m:g}'
            return 1.0, f'каскад: нет (C {ser[yest][0]:.4f} / порог {ser[yest][1]:.4f})'
        except Exception as e:
            return 1.0, f'каскад: ошибка ({self._clean(e)}) → ×1.0'

    # ---- background job: ~00:05 UTC daily, retries every 20 min until yesterday is complete
    def due(self, now_dt):
        c = self._load()
        if c['last_ok'] >= day_str(now_dt.date() - DAY):
            return False
        if (now_dt.hour, now_dt.minute) < (0, 5):
            return False
        # F10: one extra attempt from 00:50 UTC, ahead of the 01:01 slot, whatever the 20-minute spacing says (once per day)
        today = day_str(now_dt.date())
        if (now_dt.hour, now_dt.minute) >= (0, 50) and now_dt.hour == 0 and getattr(self, 'extra_day', '') != today:
            self.extra_day = today
            return True
        return self.clock() - self.last_try >= 20 * 60 or self.last_try == 0.0

    def start_background(self):
        if getattr(self, 'thread', None) is not None:
            return self.thread                                           # idempotent
        def loop():
            while True:
                try:
                    now_dt = datetime.now(timezone.utc)
                    if self.due(now_dt):
                        self.update(now_dt)
                except Exception as e:
                    self.log(f'cascade thread: {self._clean(e)}')
                time.sleep(60)
        self.thread = threading.Thread(target=loop, name='cascade', daemon=True); self.thread.start()
        return self.thread

    def export_csv(self, out):
        c = self._load()
        with open(out, 'w') as f:
            f.write('date,C\n')
            for dd, v in sorted(c['series'].items()):
                f.write(f'{dd},{v[0]}\n')


def from_config(C, notify=None, log=print, universe_fn=None):
    d = C.CASCADE_DIR or os.path.join(os.path.dirname(os.path.abspath(C.STATE_FILE)), 'cascade')
    return CascadeFilter(pct=C.CASCADE_PCT, mult=C.CASCADE_MULT, days=C.CASCADE_DAYS, win=C.CASCADE_WIN_D, min_n=C.CASCADE_MIN_D,
                         exch=tuple(x for x in C.CASCADE_EXCH.split(',') if x), nuni=C.CASCADE_NUNI, directory=d,
                         backfill_d=C.CASCADE_BACKFILL_D, per_min=C.CASCADE_RATE, notify=notify, log=log, universe_fn=universe_fn)


if __name__ == '__main__':
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import config as C
    from state import log, tg
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'status'
    cf = from_config(C, notify=tg, log=log)
    if cmd == 'backfill':
        if not cf.universe_fn:
            from bybit import Bybit
            api = Bybit(C.BASE_URL, C.API_KEY, C.API_SECRET)
            def uni():
                tk = api.tickers()
                ok = [s for s, t in tk.items() if s.endswith('USDT') and s not in C.BLACKLIST and not any(p in s for p in C.BLACKLIST_PATTERNS)]
                return sorted(ok, key=lambda s: -tk[s]['turnover24h'])
            cf.universe_fn = uni
        print('ok' if cf.update() else 'FAILED (см. лог)')
    elif cmd == 'export':
        cf.export_csv(sys.argv[2]); print('written', sys.argv[2])
    else:
        print(cf.multiplier())
