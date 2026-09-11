#!/usr/bin/env python3
import json, pandas as pd
P = 'data/panels/'
close = pd.read_parquet(P + 'close.parquet')
print('close:', close.shape, close.index[0], '->', close.index[-1])
assert 'BTCUSDT' in close.columns, 'BTCUSDT missing - panel is unusable'
for name in ('qvol', 'funding', 'high', 'low'):
    c = pd.read_parquet(P + name + '.parquet').columns
    print(name, len(c), 'cols aligned:', list(c) == list(close.columns))
prem = pd.read_parquet(P + 'premium.parquet')
miss = [c for c in close.columns if c not in prem.columns]
print('premium cols:', prem.shape[1], '| missing vs close:', len(miss), miss[:10])
if miss:
    prem = prem.reindex(columns=list(close.columns)).astype('float32')
    prem.to_parquet(P + 'premium.parquet')
    print('premium rewritten with', len(miss), 'NaN columns -> aligned to close')
meta = json.load(open(P + 'meta.json'))
print('meta symbols:', len(meta), '| BTC bars:', meta['BTCUSDT']['bars'],
      '| funding points:', meta['BTCUSDT']['n_funding'])
