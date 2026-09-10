#!/usr/bin/env python3
"""Bulk downloader for Binance Vision USDT-M futures archive.
Downloads monthly 1h klines, fundingRate and (optionally) premiumIndexKlines for all USDT perps
(including delisted) into data/raw/<kind>/<SYM>/. Incremental: skips files already on disk.
"""
import os, re, sys, json, time, concurrent.futures as cf
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


def main():
    syms = json.load(open(os.path.join(os.path.dirname(ROOT), '..', 'symbols.json')))['klines']
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
