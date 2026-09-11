#!/usr/bin/env python3
"""Build symbols.json for dl_binance.py: every USDT-M perp directory in the
Binance Vision archive, INCLUDING delisted ones (otherwise the panel gets
survivorship bias)."""
import re, json, requests

BASE = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
PREFIX = "data/futures/um/monthly/klines/"
s = requests.Session()
s.headers['User-Agent'] = 'Mozilla/5.0 research'
syms, marker = [], None
while True:
    url = f"{BASE}?delimiter=/&prefix={PREFIX}" + (f"&marker={marker}" if marker else "")
    r = s.get(url, timeout=60)
    r.raise_for_status()
    got = re.findall(r"<Prefix>" + re.escape(PREFIX) + r"([^/<]+)/</Prefix>", r.text)
    syms += got
    if "<IsTruncated>true" not in r.text or not got:
        break
    marker = PREFIX + got[-1] + "/"
syms = sorted({x for x in syms if x.endswith("USDT") and x.isascii()})
json.dump({"klines": syms}, open("symbols.json", "w"), indent=0)
print(len(syms), "symbols -> symbols.json")
