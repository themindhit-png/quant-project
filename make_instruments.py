#!/usr/bin/env python3
"""Fetch Bybit linear USDT perp instrument specs into instruments.json.
Format is defined by Bybit.instruments() in v4/bybit.py: qtyStep, minQty, maxQty,
minNotional, tickSize, launch (ms), maxLeverage, fundingInterval. Public endpoint."""
import os, sys, json

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'v4'))
from bybit import Bybit

BASE = os.environ.get('BYBIT_BASE', 'https://api.bybit.com')
api = Bybit(BASE)
instr = api.instruments()
n_launch = sum(1 for v in instr.values() if v.get('launch'))
json.dump(instr, open(os.path.join(HERE, 'instruments.json'), 'w'), indent=0)
print(len(instr), 'instruments ->', 'instruments.json', '| with launchTime:', n_launch)
print('BTCUSDT:', instr.get('BTCUSDT'))
