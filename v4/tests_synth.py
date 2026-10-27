#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Synthetic hourly panels (fixed seed) in the data/panels parquet layout, so that tests_v4.py runs without the 1.3 GB
Binance download (CI, fresh clones). The research-parity tests compare two implementations on the SAME input, so
synthetic prices are enough for them; they say nothing about the strategy's edge."""
import os
import numpy as np, pandas as pd

MAJORS = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT']


def make_panels(out_dir, seed=7, n_old=170, n_young=20, start='2025-03-01', end='2025-09-30 23:00'):
    """Write close/high/low/qvol/funding/premium.parquet to out_dir and return out_dir.
    n_old names trade over the whole window; n_young names list 10-80 days before `end` (listing sleeve, age filters)."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, end, freq='h', tz='UTC')
    T = len(idx)
    syms = MAJORS + [f'A{k:03d}USDT' for k in range(n_old - len(MAJORS))] + [f'Z{k:03d}USDT' for k in range(n_young)]
    N = len(syms)
    mkt = rng.standard_normal(T) * 0.006                                   # common (BTC-like) factor
    beta = np.r_[1.0, rng.uniform(0.6, 1.6, N - 1)]
    vol = np.r_[0.006, rng.uniform(0.006, 0.02, N - 1)]
    drift = rng.normal(0, 2e-5, N)
    r = mkt[:, None] * beta + rng.standard_normal((T, N)) * vol + drift
    r[:, 0] = mkt
    close = 10.0 ** rng.uniform(-2, 3, N) * np.exp(np.cumsum(r, 0))
    wick = np.abs(rng.standard_normal((T, N))) * vol * 0.5
    high = close * np.exp(wick); low = close * np.exp(-np.abs(rng.standard_normal((T, N))) * vol * 0.5)
    base_turn = 10.0 ** rng.uniform(5.3, 7.5, N); base_turn[:len(MAJORS)] = 10.0 ** rng.uniform(7.5, 8.5, len(MAJORS))
    qvol = base_turn * np.exp(rng.normal(0, 0.5, (T, N)))
    fund = np.zeros((T, N))
    f8 = np.asarray(idx.hour % 8 == 0)
    fund[f8] = rng.normal(5e-5, 1.5e-4, (int(f8.sum()), N))
    prem = rng.normal(-1e-4, 4e-4, (T, N)) + rng.normal(0, 2e-4, N)
    for k in range(n_young):                                               # late listings: NaN before the first bar
        j = n_old + k; first = T - int(rng.integers(10 * 24, 80 * 24))
        for a in (close, high, low, qvol, prem):
            a[:first, j] = np.nan
        fund[:first, j] = np.nan
    order = np.argsort(syms, kind='stable'); syms = [syms[j] for j in order]          # alphabetical like the real panels
    close, high, low, qvol, fund, prem = (a[:, order] for a in (close, high, low, qvol, fund, prem))
    os.makedirs(out_dir, exist_ok=True)
    for name, a in dict(close=close, high=high, low=low, qvol=qvol, funding=fund, premium=prem).items():
        pd.DataFrame(a.astype(np.float64), index=idx, columns=syms).to_parquet(os.path.join(out_dir, f'{name}.parquet'))
    return out_dir


if __name__ == '__main__':
    import sys, time
    t0 = time.time(); p = make_panels(sys.argv[1] if len(sys.argv) > 1 else 'synth_panels')
    print(f'{p}: {time.time() - t0:.1f}s')
