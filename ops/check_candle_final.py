#!/usr/bin/env python3
"""Reg. 5 step R-3 (docs/REG5_PLAN.md): is the hourly Bybit candle final a few seconds after the hour? (needed before moving the
rebalance from hh:10 to hh:01). Public API, no keys, does not touch the bot. Waits for the next full hour, then reads the just-closed 1h
candle of 12 symbols at +3, +10, +20, +30, +45, +60, +120 and +300 s and reports whether close / volume / turnover still changed after
each moment (the +300 s value is taken as final). Repeat over 3 hours by default.
Owner, on the VPS:   python3 ops/check_candle_final.py            (takes ~3 h; or --hours 1)
Result: the line "ФИНАЛЬНА с +N с" — the rebalance minute must be later than that."""
import argparse, json, time, urllib.request, urllib.parse
SYMS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'DOGEUSDT', 'ADAUSDT', 'LINKUSDT', 'AVAXUSDT', 'SUIUSDT', 'LTCUSDT', 'WIFUSDT', 'ENAUSDT']
OFFS = [3, 10, 20, 30, 45, 60, 120, 300]

def kline(sym, start_ms):
    q = urllib.parse.urlencode({'category': 'linear', 'symbol': sym, 'interval': '60', 'start': start_ms, 'end': start_ms, 'limit': 1})
    for i in range(4):                                                # Bybit sometimes answers retCode!=0 (rate limit) -> retry
        try:
            with urllib.request.urlopen('https://api.bybit.com/v5/market/kline?' + q, timeout=10) as r:
                js = json.load(r)
            lst = (js.get('result') or {}).get('list')
            if lst is not None:
                return tuple(float(x) for x in lst[0][4:7]) if lst else None   # close, volume, turnover
            err = f"retCode {js.get('retCode')} {js.get('retMsg')}"
        except Exception as e:
            err = repr(e)
        time.sleep(0.5 * (i + 1))
    print(f'  ! {sym}: {err}', flush=True)
    return 'ERR'

def one_hour():
    now = time.time(); top = (int(now) // 3600 + 1) * 3600; start_ms = (top - 3600) * 1000; snaps = {}
    for off in OFFS:
        time.sleep(max(0.0, top + off - time.time()))
        snaps[off] = {s: kline(s, start_ms) for s in SYMS}
    fin = snaps[OFFS[-1]]; first_ok = None
    for off in OFFS[:-1]:
        diff = [s for s in SYMS if 'ERR' not in (snaps[off][s], fin[s]) and snaps[off][s] != fin[s]]
        print(f'  {time.strftime("%Y-%m-%d %H:00", time.gmtime(top - 3600))} свеча, +{off:>3} с: отличается от финальной у {len(diff)} из {len(SYMS)} {diff[:4]}', flush=True)
        if not diff and first_ok is None: first_ok = off
        if diff: first_ok = None
    return first_ok

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--hours', type=int, default=3); a = ap.parse_args(); res = []
    for _ in range(a.hours):
        res.append(one_hour())
    worst = None if any(r is None for r in res) else max(res)
    print(f'ИТОГ: свеча ФИНАЛЬНА с +{worst} с (по {a.hours} ч)' if worst is not None else 'ИТОГ: свеча менялась до +120 с хотя бы раз — минуту ребаланса 1 НЕ ставить, сообщить в чат')
