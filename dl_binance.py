#!/usr/bin/env python3
"""Bulk downloader for Binance Vision USDT-M futures archive.
Downloads monthly 1h klines, fundingRate and (optionally) premiumIndexKlines for all USDT perps
(including delisted) into data/raw/<kind>/<SYM>/. Incremental: skips files already on disk.

  python3 dl_binance.py klines,fundingRate,premiumIndexKlines   # monthly archives
  python3 dl_binance.py gaps                                     # fill holes of the monthly archives from DAILY archives

`gaps` (audit 2026-10-31, PREREG reg. 4): some monthly archives are truncated (e.g. SOLUSDT-1h-2022-02.zip ends 02-25 23:00,
the April file starts 04-03) although the daily archives for the missing days exist. Every hour missing between a symbol's first
and last monthly bar (klines and premiumIndexKlines) -> its daily archive is fetched into data/raw/daily/<kind>/<SYM>/1h/;
build_panels.py uses daily rows only for hours that no monthly file has. Holes longer than GAP_MAX_D (default 45) days are
left alone (real outages / relistings, not truncation). 404s are remembered in data/raw/daily/_404.json.
"""
import os, re, sys, io, json, glob, time, zipfile, concurrent.futures as cf
import requests

BASE_S3 = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
BASE_DL = "https://data.binance.vision/"
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'raw')
KINDS = sys.argv[1].split(',') if len(sys.argv) > 1 else ['klines', 'fundingRate']
sess = requests.Session()
sess.headers['User-Agent'] = 'Mozilla/5.0 research-downloader'


def list_keys(prefix):
    out, marker = [], None
    for _ in range(50):
        url = f"{BASE_S3}?prefix={prefix}" + (f"&marker={marker}" if marker else "")
        for attempt in range(5):
            try:
                r = sess.get(url, timeout=60)
                if r.status_code == 200:
                    break
            except Exception:
                time.sleep(1 + attempt)
        else:
            raise RuntimeError(f'listing failed {prefix}')
        keys = re.findall(r'<Key>([^<]+)</Key>', r.text)
        out += [k for k in keys if k.endswith('.zip')]
        m = re.search(r'<NextMarker>([^<]+)</NextMarker>', r.text)
        if '<IsTruncated>true' not in r.text:
            break
        marker = m.group(1) if m else (keys[-1] if keys else None)
        if not marker:
            break
    return out


def prefix_for(kind, sym):
    if kind == 'klines':
        return f"data/futures/um/monthly/klines/{sym}/1h/"
    if kind == 'fundingRate':
        return f"data/futures/um/monthly/fundingRate/{sym}/"
    if kind == 'premiumIndexKlines':
        return f"data/futures/um/monthly/premiumIndexKlines/{sym}/1h/"
    raise ValueError(kind)


def download(key):
    dest = os.path.join(ROOT, key.replace('data/futures/um/monthly/', ''))
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return 'skip'
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    for attempt in range(6):
        try:
            r = sess.get(BASE_DL + key, timeout=120)
            if r.status_code == 200 and r.content[:2] == b'PK':
                with open(dest + '.tmp', 'wb') as f:
                    f.write(r.content)
                os.replace(dest + '.tmp', dest)
                return 'ok'
            if r.status_code == 404:
                return '404'
        except Exception:
            pass
        time.sleep(1 + attempt * 2)
    return 'fail'


def _open_hours(path):
    """open_time (ms, hour-floored) of every row of one kline zip."""
    with zipfile.ZipFile(path) as z:
        raw = z.read(z.namelist()[0])
    out = []
    for line in raw.splitlines():
        f = line.split(b',', 1)[0]
        if f[:1].isdigit():
            t = int(f)
            out.append((t // 1000 if t > 1e14 else t) // 3600000)
    return out


def _sym_gap_days(args):
    kind, sym, max_d = args
    hrs = set()
    for f in sorted(glob.glob(os.path.join(ROOT, kind, sym, '1h', '*.zip'))):
        try:
            hrs.update(_open_hours(f))
        except Exception:
            pass
    if len(hrs) < 2:
        return kind, sym, [], []
    h = sorted(hrs); days, long_gaps = set(), []
    for a, b in zip(h[:-1], h[1:]):
        if b - a > 1:
            if (b - a - 1) > max_d * 24:
                long_gaps.append((a + 1, b - 1)); continue
            for x in range(a + 1, b):
                days.add(time.strftime('%Y-%m-%d', time.gmtime(x * 3600)))
    return kind, sym, sorted(days), long_gaps


def fill_gaps(max_d=None):
    max_d = float(os.environ.get('GAP_MAX_D', 45)) if max_d is None else max_d
    droot = os.path.join(ROOT, 'daily'); os.makedirs(droot, exist_ok=True)
    f404 = os.path.join(droot, '_404.json')
    seen404 = set(json.load(open(f404))) if os.path.exists(f404) else set()
    t0 = time.time(); jobs = []
    for kind in ('klines', 'premiumIndexKlines'):
        if os.path.isdir(os.path.join(ROOT, kind)):
            jobs += [(kind, s, max_d) for s in sorted(os.listdir(os.path.join(ROOT, kind))) if s.endswith('USDT')]
    keys, n_long = [], 0
    with cf.ProcessPoolExecutor(min(8, os.cpu_count() or 2)) as ex:
        for kind, sym, days, long_gaps in ex.map(_sym_gap_days, jobs, chunksize=8):
            n_long += len(long_gaps)
            for dd in days:
                keys.append(f'data/futures/um/daily/{kind}/{sym}/1h/{sym}-1h-{dd}.zip')
    keys = [k for k in keys if k not in seen404]
    print(f'gaps: {len(keys)} daily archives to check ({n_long} holes > {max_d:g} d left alone), scan {time.time()-t0:.0f}s', flush=True)

    def dl(key):
        dest = os.path.join(droot, key.replace('data/futures/um/daily/', ''))
        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            return key, 'skip'
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        for attempt in range(5):
            try:
                r = sess.get(BASE_DL + key, timeout=60)
                if r.status_code == 200 and r.content[:2] == b'PK':
                    with open(dest + '.tmp', 'wb') as f:
                        f.write(r.content)
                    os.replace(dest + '.tmp', dest)
                    return key, 'ok'
                if r.status_code == 404:
                    return key, '404'
            except Exception:
                pass
            time.sleep(1 + attempt)
        return key, 'fail'
    stats = {}
    with cf.ThreadPoolExecutor(16) as ex:
        for key, res in ex.map(dl, keys):
            stats[res] = stats.get(res, 0) + 1
            if res == '404':
                seen404.add(key)
    json.dump(sorted(seen404), open(f404, 'w'))
    print(f'gaps DONE {stats} in {time.time()-t0:.0f}s', flush=True)


def main():
    if KINDS == ['gaps']:
        return fill_gaps()
    syms = json.load(open(os.environ.get('SYMBOLS_FILE') or os.path.join(os.path.dirname(ROOT), '..', 'symbols.json')))['klines']
    syms = [s for s in syms if s.isascii()]
    print(f'symbols: {len(syms)}  kinds: {KINDS}', flush=True)
    t0 = time.time()
    keys = []
    with cf.ThreadPoolExecutor(16) as ex:
        futs = {ex.submit(list_keys, prefix_for(k, s)): (k, s) for k in KINDS for s in syms}
        for i, f in enumerate(cf.as_completed(futs)):
            keys += f.result()
            if i % 200 == 0:
                print(f'  listed {i}/{len(futs)} ({time.time()-t0:.0f}s) keys so far {len(keys)}', flush=True)
    keys = sorted(set(keys))
    json.dump(keys, open(os.path.join(ROOT, f'keys_{"_".join(KINDS)}.json'), 'w'))
    print(f'total files: {len(keys)}  listing took {time.time()-t0:.0f}s', flush=True)
    stats = {}
    with cf.ThreadPoolExecutor(24) as ex:
        for i, res in enumerate(ex.map(download, keys)):
            stats[res] = stats.get(res, 0) + 1
            if i % 1000 == 0:
                print(f'  {i}/{len(keys)} {stats} ({time.time()-t0:.0f}s)', flush=True)
    print(f'DONE {stats} in {time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
