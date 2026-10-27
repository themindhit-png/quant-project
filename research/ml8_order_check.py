#!/usr/bin/env python3
# Usage (repo root): PYTHONPATH=$PWD python3 research/ml8_order_check.py <dir with ml_8h.txt + spec_8h.json> [...]
# Order-dependence of ml8 predictions: old (41 feats) vs new (36) model; share of ties per feature on the ml8 universe.
import sys, json, numpy as np, lightgbm as lgb
sys.path.insert(0, 'v4')
from bt import Data
from ml_features_8h import compute_features_8h
from ml_features_v3 import rank_norm
from scipy.stats import spearmanr
d = Data(start='2025-06-01', end='2025-09-30 23:00', min_turn_ever=5e6, load_hl=True, load_premium=True)
jb = int(np.flatnonzero(d.cols == 'BTCUSDT')[0]); rng = np.random.default_rng(0)
for path in sys.argv[1:]:
    mdl = lgb.Booster(model_file=path + '/ml_8h.txt'); feats = json.load(open(path + '/spec_8h.json'))['features']
    sps = []; ties = {}
    for i in range(d.ret.shape[0] - 1 - 24 * 40, d.ret.shape[0] - 1, 24):
        m = d.universe(i, min_age_h=720, min_turn=1e7, top_n=150); ix = np.flatnonzero(m)
        sl = slice(i - 2160, i + 1)
        F = compute_features_8h(d.cff[sl], d.fund[sl], d.t24[sl], d.prem[sl], d.high[sl], d.low[sl], d.age[i].astype(float), jb, int((d.hh[i] + 1) % 24))
        X = np.column_stack([F[f][ix] for f in feats]).astype(np.float32)
        for k, f in enumerate(feats):
            c = X[:, k][np.isfinite(X[:, k])]; ties.setdefault(f, []).append(1 - len(np.unique(c)) / max(len(c), 1))
        a = mdl.predict(rank_norm(X).astype(np.float64))
        for _ in range(3):
            p = rng.permutation(len(ix)); b = mdl.predict(rank_norm(X[p]).astype(np.float64))[np.argsort(p)]; sps.append(spearmanr(a, b).correlation)
    print(path, len(feats), 'perm Spearman mean %.3f min %.3f' % (np.mean(sps), np.min(sps)))
    print('  tie share >5%:', {f: round(float(np.mean(v)), 2) for f, v in ties.items() if np.mean(v) > 0.05})
