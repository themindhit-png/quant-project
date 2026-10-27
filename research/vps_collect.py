#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Сбор данных с VPS для аудита (Фаза 2). Запускать на VPS: python3 research/vps_collect.py

Что делает (всё только на чтение, ключи НЕ собираются и не печатаются):
  1. Находит процесс бота (v4/main.py), его параметры (без секретов), файл состояния, лог.
  2. Хэши sha256 файлов v4/*.py и v4/models/*.
  3. /status бота (токен берётся из окружения процесса, в дамп не пишется).
  4. Выжимка логов: счётчики по ключевым словам, REBALANCE ABORTED, tracebacks, последние строки.
  5. Публичный API Bybit: instruments-info и история фандинга по всем символам (с 2025-01-01).
  5б. Часовые свечи (o,h,l,c,turnover) и premium-index close с 2026-01-01 для топ-400 символов по обороту
      (public.bybit.com для USDT-перпов даёт только сырые сделки, готовых свечей нет). Шарды по 50 символов.
  6. Пуш в ветку vps-dump (если на VPS настроен git-доступ), иначе путь к архиву.
"""
import os, sys, re, json, gzip, time, hashlib, subprocess, glob, urllib.request, urllib.parse
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'research', 'vps_dump')
BYBIT = 'https://api.bybit.com'
FUND_FROM = datetime(2025, 1, 1, tzinfo=timezone.utc)
KLINE_FROM = datetime(2026, 1, 1, tzinfo=timezone.utc)
TOP_N = 400
SECRET_KEYS = re.compile(r'(KEY|SECRET|TOKEN|PASS|CHAT)', re.I)
SECRET_VAL = re.compile(r'\b[A-Za-z0-9_\-]{28,}\b')


def sh(cmd, timeout=60):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout).stdout
    except Exception as e:
        return f'ERR {e}'


def redact(text):
    return SECRET_VAL.sub('<redacted>', text)


def http_json(url, tries=4):
    err = None
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'vps-collect'})
            with urllib.request.urlopen(req, timeout=20) as r:
                return json.loads(r.read().decode())
        except Exception as e:
            err = e
            time.sleep(1 + 2 * i)
    return {'error': str(err)}


def config_keys():
    try:
        src = open(os.path.join(ROOT, 'v4', 'config.py'), encoding='utf-8').read()
        return set(re.findall(r"_env\('([A-Z0-9_]+)'", src))
    except Exception:
        return set()


def find_bot():
    info = {'pid': None}
    for line in sh("ps -eo pid,etimes,args | grep 'main.py' | grep -v grep").splitlines():
        parts = line.split(None, 2)
        if len(parts) == 3 and 'main.py' in parts[2]:
            info.update(pid=int(parts[0]), uptime_sec=int(parts[1]), cmd=parts[2])
            break
    if not info['pid']:
        return info
    pid = info['pid']
    try:
        info['cwd'] = os.readlink(f'/proc/{pid}/cwd')
    except Exception:
        pass
    env = {}
    try:
        for kv in open(f'/proc/{pid}/environ', 'rb').read().split(b'\0'):
            if b'=' in kv:
                k, v = kv.decode(errors='replace').split('=', 1)
                env[k] = v
    except Exception as e:
        info['environ_error'] = str(e)
    info['_env'] = env  # приватное, в дамп не пишем
    ck = config_keys()
    info['env_public'] = {k: v for k, v in env.items() if k in ck and not SECRET_KEYS.search(k)}
    info['cgroup'] = sh(f'cat /proc/{pid}/cgroup | tail -1').strip()
    return info


def file_hashes(bot):
    """Сравнение развёрнутого кода (папка процесса) с v4/ из репозитория: хэши + diff."""
    import difflib
    out, diffs = {}, []
    cwd = bot.get('cwd') or ''
    mdir = next((d for d in (os.path.join(cwd, 'models'), os.path.join(os.path.dirname(cwd), 'models')) if os.path.isdir(d)), '')
    out['deployed_dir'], out['deployed_models_dir'] = cwd, mdir
    pairs = [(p, os.path.join(cwd, os.path.basename(p))) for p in sorted(glob.glob(os.path.join(ROOT, 'v4', '*.py')))]
    pairs += [(p, os.path.join(mdir, os.path.basename(p))) for p in sorted(glob.glob(os.path.join(ROOT, 'v4', 'models', '*')))]
    for repo_p, dep_p in pairs:
        name = os.path.relpath(repo_p, ROOT)
        rh = hashlib.sha256(open(repo_p, 'rb').read()).hexdigest()[:16]
        dh = hashlib.sha256(open(dep_p, 'rb').read()).hexdigest()[:16] if os.path.isfile(dep_p) else None
        out[name] = {'repo': rh, 'deployed': dh, 'same': rh == dh}
        if dh and rh != dh and name.endswith('.py'):
            d = difflib.unified_diff(open(repo_p, errors='replace').read().splitlines(), open(dep_p, errors='replace').read().splitlines(),
                                     'repo/' + name, 'deployed/' + name, lineterm='', n=2)
            diffs.append(redact('\n'.join(d)))
    open(os.path.join(OUT, 'code_diff.txt'), 'w', encoding='utf-8').write('\n\n'.join(diffs) or 'no .py differences')
    out['_repo_git_head'] = sh(f'cd {ROOT} && git rev-parse HEAD').strip()
    return out


def get_status(bot):
    if not bot.get('pid'):
        return {'error': 'no bot process'}
    env = bot.get('_env', {})
    port, tok = env.get('PORT', '10000'), env.get('STATUS_TOKEN', '')
    return http_json(f'http://127.0.0.1:{port}/status?token={urllib.parse.quote(tok)}')


def read_logs(bot):
    """Лог: самый длинный из journalctl / файла stdout процесса / log*.txt|log рядом с ботом."""
    cands, pid = [], bot.get('pid')
    if pid:
        m = re.search(r'([^/]+\.service)', bot.get('cgroup', ''))
        if m:
            cands.append((f'journalctl -u {m.group(1)}', sh(f'journalctl -u {m.group(1)} --since "-60 days" --no-pager -o short-iso 2>/dev/null', timeout=120)))
        try:
            fd = os.readlink(f'/proc/{pid}/fd/1')
            if os.path.isfile(fd):
                cands.append((fd, open(fd, errors='replace').read()))
        except Exception:
            pass
    cwd = bot.get('cwd') or ROOT
    for d in {cwd, os.path.dirname(cwd)}:
        for p in glob.glob(os.path.join(d, 'log*')) + glob.glob(os.path.join(d, 'nohup.out')):
            if os.path.isfile(p):
                cands.append((p, open(p, errors='replace').read()))
    if not cands:
        return '', None
    src, text = max(cands, key=lambda x: len(x[1]))
    return text, src + ' (из: ' + ', '.join(c[0] for c in cands) + ')'


KEYWORDS = ['REBALANCE ABORTED', 'Traceback', 'ERROR', 'WARN', 'empty sleeves', 'halt', 'HALT', 'kill', 'funding basis',
            'effective config hash', 'rebalance', 'reject', 'timeout', 'rate limit', 'retCode']


def summarize_logs(text):
    lines = text.splitlines()
    cnt = {k: sum(1 for l in lines if k in l) for k in KEYWORDS}
    aborted = [redact(l) for l in lines if 'REBALANCE ABORTED' in l][-50:]
    tb, i = [], 0
    while i < len(lines):
        if 'Traceback' in lines[i]:
            tb.append(redact('\n'.join(lines[max(0, i - 1):i + 14])))
            i += 14
        else:
            i += 1
    cfg = [redact(l) for l in lines if 'effective config hash' in l][-5:]
    summ = dict(n_lines=len(lines), first=redact(lines[0]) if lines else '', last=redact(lines[-1]) if lines else '',
                counts=cnt, aborted_last50=aborted, tracebacks_last20=tb[-20:], config_hash_lines=cfg)
    return summ, redact('\n'.join(lines[-400:]))


def collect_bybit():
    items, cur = [], ''
    while True:
        u = f'{BYBIT}/v5/market/instruments-info?category=linear&limit=1000'
        if cur:
            u += '&cursor=' + urllib.parse.quote(cur)
        r = http_json(u)
        items += r.get('result', {}).get('list', [])
        cur = r.get('result', {}).get('nextPageCursor')
        if not cur:
            break
    with gzip.open(os.path.join(OUT, 'instruments_all.json.gz'), 'wt') as f:
        json.dump(items, f)
    syms = sorted(it['symbol'] for it in items if it.get('settleCoin') == 'USDT' and it.get('contractType') == 'LinearPerpetual')
    print(f'instruments: {len(items)}, USDT perps: {len(syms)}', flush=True)

    start, end0 = int(FUND_FROM.timestamp() * 1000), int(time.time() * 1000)
    n = 0
    with gzip.open(os.path.join(OUT, 'funding_bybit.csv.gz'), 'wt') as f:
        f.write('symbol,ts_ms,rate\n')
        for si, s in enumerate(syms):
            end, got = end0, set()
            for _ in range(30):
                r = http_json(f'{BYBIT}/v5/market/funding/history?category=linear&symbol={s}&startTime={start}&endTime={end}&limit=200')
                lst = r.get('result', {}).get('list', [])
                if not lst:
                    break
                for it in lst:
                    got.add((int(it['fundingRateTimestamp']), it['fundingRate']))
                oldest = min(int(it['fundingRateTimestamp']) for it in lst)
                if oldest <= start or len(lst) < 200:
                    break
                end = oldest - 1
                time.sleep(0.05)
            for ts, rate in sorted(got):
                f.write(f'{s},{ts},{rate}\n')
                n += 1
            if si % 50 == 0:
                print(f'funding {si}/{len(syms)} rows={n}', flush=True)
            time.sleep(0.05)
    tk = http_json(f'{BYBIT}/v5/market/tickers?category=linear').get('result', {}).get('list', [])
    tv = {t['symbol']: float(t.get('turnover24h') or 0) for t in tk}
    ranked = sorted([x for x in syms if x in tv], key=lambda x: -tv[x])
    res = dict(n_instruments=len(items), n_usdt_perps=len(syms), funding_rows=n)
    res.update(collect_klines(ranked))
    return res


def collect_klines(syms_by_turnover):
    kd = os.path.join(OUT, 'klines')
    os.makedirs(kd, exist_ok=True)
    start, end0 = int(KLINE_FROM.timestamp() * 1000), int(time.time() * 1000)
    syms = syms_by_turnover[:TOP_N]
    total = 0
    for sh_i in range(0, len(syms), 50):
        part = syms[sh_i:sh_i + 50]
        with gzip.open(os.path.join(kd, f'shard_{sh_i // 50:02d}.csv.gz'), 'wt') as f:
            f.write('symbol,ts_ms,open,high,low,close,turnover,prem_close\n')
            for s in part:
                kl, pr = {}, {}
                r_end = end0
                for _ in range(20):
                    r = http_json(f'{BYBIT}/v5/market/kline?category=linear&symbol={s}&interval=60&start={start}&end={r_end}&limit=1000')
                    lst = r.get('result', {}).get('list', [])
                    if not lst:
                        break
                    for it in lst:
                        kl[int(it[0])] = (it[1], it[2], it[3], it[4], it[6])
                    oldest = min(int(it[0]) for it in lst)
                    if oldest <= start or len(lst) < 1000:
                        break
                    r_end = oldest - 1
                    time.sleep(0.05)
                r_end = end0
                for _ in range(20):
                    r = http_json(f'{BYBIT}/v5/market/premium-index-price-kline?category=linear&symbol={s}&interval=60&start={start}&end={r_end}&limit=1000')
                    lst = r.get('result', {}).get('list', [])
                    if not lst:
                        break
                    for it in lst:
                        pr[int(it[0])] = it[4]
                    oldest = min(int(it[0]) for it in lst)
                    if oldest <= start or len(lst) < 1000:
                        break
                    r_end = oldest - 1
                    time.sleep(0.05)
                for ts in sorted(kl):
                    o, h, l, c, t = kl[ts]
                    f.write(f'{s},{ts},{o},{h},{l},{c},{t},{pr.get(ts, "")}\n')
                    total += 1
        print(f'klines shard {sh_i // 50 + 1}/{(len(syms) + 49) // 50} rows={total}', flush=True)
    return dict(kline_symbols=len(syms), kline_rows=total)


def main():
    os.makedirs(OUT, exist_ok=True)
    meta = {'collected_utc': datetime.now(timezone.utc).isoformat(), 'host': sh('hostname').strip(), 'root': ROOT}
    bot = find_bot()
    meta['bot'] = {k: v for k, v in bot.items() if not k.startswith('_')}
    meta['hashes'] = file_hashes(bot)
    meta['status'] = get_status(bot)
    st = (bot.get('_env') or {}).get('STATE_FILE')
    if not st:
        sf = glob.glob(os.path.join(os.path.dirname(bot.get('cwd') or ROOT), 'state', '*.json')) + glob.glob(os.path.join(bot.get('cwd') or ROOT, 'state', '*.json'))
        st = max(sf, key=os.path.getmtime) if sf else None
    if st and os.path.isfile(st):
        try:
            meta['state_file'] = json.load(open(st))
        except Exception as e:
            meta['state_file'] = f'ERR {e}'
    text, src = read_logs(bot)
    meta['log_source'] = src
    summ, tail = summarize_logs(text)
    meta['log_summary'] = summ
    open(os.path.join(OUT, 'log_tail.txt'), 'w', encoding='utf-8').write(tail)
    with gzip.open(os.path.join(OUT, 'log_full_redacted.txt.gz'), 'wt', encoding='utf-8') as f:
        f.write(redact(text))
    meta['bybit'] = collect_bybit()
    # страховка: значения секретов не должны попасть в дамп
    blob = json.dumps(meta, ensure_ascii=False, default=str, indent=1)
    for k, v in (bot.get('_env') or {}).items():
        if SECRET_KEYS.search(k) and len(v) >= 6:
            assert v not in blob and v not in tail, f'secret {k} leaked into dump'
    open(os.path.join(OUT, 'meta.json'), 'w', encoding='utf-8').write(blob)
    print('Дамп готов:', OUT, flush=True)

    out = sh(f'cd {ROOT} && git checkout -B vps-dump && git add -f research/vps_dump && '
             f'git -c user.name=vps -c user.email=vps@example.com commit -m "vps dump {meta["collected_utc"][:16]}" && git push -u origin vps-dump --force 2>&1', timeout=180)
    print(out[-600:])
    sh(f'cd {ROOT} && git checkout -')
    if 'vps-dump' not in out or 'fatal' in out or 'rejected' in out:
        print(f'Пуш не удался. Заархивируйте и пришлите: tar czf /tmp/vps_dump.tgz -C {ROOT}/research vps_dump')


if __name__ == '__main__':
    main()
