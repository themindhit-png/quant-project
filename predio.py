#!/usr/bin/env python3
"""Prediction files (T, N) float32: dense .npy (old format) or sparse .npz (only rows with predictions; used for seed bagging so that
30 walk-forward runs do not take 5 GB of disk). load_pred(base) reads base.npy if it exists, else base.npz; bag(bases, out) averages."""
import os, numpy as np

def save_pred(base, P, sparse=True):
    if not sparse:
        np.save(base + '.npy', P); return
    rows = np.flatnonzero(np.isfinite(P).any(1)); np.savez_compressed(base + '.npz', shape=np.array(P.shape), rows=rows, vals=P[rows])

def load_pred(base, mmap=True):
    if os.path.exists(base + '.npy'):
        return np.load(base + '.npy', mmap_mode='r' if mmap else None)
    z = np.load(base + '.npz'); P = np.full(tuple(z['shape']), np.nan, dtype=np.float32); P[z['rows']] = z['vals']; return P

def bag(bases, out):
    """mean of several prediction files (NaN where any member is NaN -> the same coin set as a single model)"""
    Ps = [load_pred(b, mmap=False) for b in bases]; P = np.mean(np.stack(Ps), 0).astype(np.float32); save_pred(out, P, True); return P

if __name__ == '__main__':
    import sys
    bag(sys.argv[2:], sys.argv[1]); print('bagged', len(sys.argv) - 2, '->', sys.argv[1] + '.npz')
