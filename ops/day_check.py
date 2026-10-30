#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Why does the Mubite dashboard count 0 trading days? Read-only: Bybit closed-PnL records per UTC day vs the two definitions of a
'trading day' found for Mubite (docs/FIRMS.md):
  A (Mubite FAQ): a close with volume >= 5 % of the starting balance AND |P/L| >= 1 % of that position's margin;
  B (third-party review): |closed P/L of the day| >= 0.25 % of the starting balance.
Also the maker share of fills. No orders, no keys printed.
Run on the VPS:  /opt/propsleeves/venv/bin/python /root/quant-project/ops/day_check.py [days=4] [env=/opt/propsleeves/bot.env]"""
import os, sys
from collections import defaultdict
from datetime import datetime, timezone, timedelta
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'v4'))
from opslib import read_env
from bybit import Bybit


def closed_pnl(api, start_ms, end_ms):
    out, cursor = [], ''
    while True:
        p = {'category': 'linear', 'startTime': int(start_ms), 'endTime': int(end_ms), 'limit': 100}
        if cursor:
            p['cursor'] = cursor
        r = api._req('GET', '/v5/position/closed-pnl', p, auth=True, retries=2)
        out.extend(r.get('list') or [])
        cursor = r.get('nextPageCursor', '')
        if not cursor or len(out) >= 5000:
            return out


def main():
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    env = read_env(sys.argv[2] if len(sys.argv) > 2 else '/opt/propsleeves/bot.env')
    api = Bybit(env.get('BYBIT_BASE_URL', 'https://api-demo.bybit.com'), env['BYBIT_API_KEY'], env['BYBIT_API_SECRET'], lambda *a: None)
    init = float(env.get('ACCOUNT_SIZE') or 0)
    now = datetime.now(timezone.utc); start = (now - timedelta(days=days)).replace(hour=0, minute=0, second=0, microsecond=0)
    rows = closed_pnl(api, start.timestamp() * 1000, now.timestamp() * 1000)
    by = defaultdict(list)
    for r in rows:
        by[datetime.fromtimestamp(int(r['updatedTime']) / 1000, timezone.utc).date()].append(r)
    print(f'счёт {init:,.0f}: A = закрытие >= {0.05 * init:,.0f} USDT и |P/L| >= 1 % маржи;  B = |P/L дня| >= {0.0025 * init:,.0f} USDT\n')
    print(f'{"день UTC":10} {"закрытий":>8} {"макс. объём":>11} {">=5 %":>6} {"A":>3} {"P/L дня":>9} {"сумма |P/L|":>11} {"B":>3}')
    for d in sorted(by):
        rs = by[d]
        vol = [float(r['cumExitValue']) for r in rs]
        a = [r for r in rs if float(r['cumExitValue']) >= 0.05 * init
             and abs(float(r['closedPnl'])) >= 0.01 * float(r['cumEntryValue']) / max(float(r.get('leverage') or 1), 1e-9)]
        pnl = sum(float(r['closedPnl']) for r in rs); apnl = sum(abs(float(r['closedPnl'])) for r in rs)
        print(f'{d!s:10} {len(rs):>8} {max(vol):>11,.0f} {sum(v >= 0.05 * init for v in vol):>6} {"да" if a else "нет":>3} '
              f'{pnl:>9,.0f} {apnl:>11,.0f} {"да" if abs(pnl) >= 0.0025 * init else "нет":>3}')
    ex = api.executions(start.timestamp() * 1000)
    if ex:
        mk = sum(1 for e in ex if str(e.get('isMaker')).lower() == 'true')
        print(f'\nисполнений {len(ex)}, maker {mk} ({mk / len(ex):.0%})')


if __name__ == '__main__':
    main()
