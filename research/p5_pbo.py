#!/usr/bin/env python3
"""Reg. 5 final checks (docs/REG5_PLAN.md, package rule): PBO by CSCV over the matrix "all forward variants of this plan x 10 half-year
blocks" (lag1_x1 daily returns) and DSR (phase4_gate.dsr, SR0 from the trial-Sharpe spread) of Base-R and of a candidate.
  python3 research/p5_pbo.py <base_tag> <cand_tag> <N_before> <N_after> <tag1> <tag2> ...   (tags = research/phase4_out/P5_<tag>.csv)"""
import os, sys, itertools, math, numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); os.chdir(ROOT); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'research'))
from p5 import load, halves, sharpe
from phase4_gate import dsr, trials_std
base, cand, n0, n1, tags = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5:]
M = pd.DataFrame({t: {k: sharpe(v) for k, v in halves(load(t)['lag1_x1']).items()} for t in tags})   # blocks x variants
B = list(M.index); logits = []
for IS in itertools.combinations(range(len(B)), len(B) // 2):
    OS = [k for k in range(len(B)) if k not in IS]
    best = M.iloc[list(IS)].mean().idxmax(); r = M.iloc[OS].mean().rank()[best] / (M.shape[1] + 1)
    logits.append(math.log(r / (1 - r)))
pbo = float(np.mean(np.array(logits) <= 0)); sd = trials_std()
db, s0b = dsr(load(base)['lag1_x1'], n0, sd); dc, s0c = dsr(load(cand)['lag1_x1'], n1, sd)
print(M.round(2).to_string()); print(f'PBO (CSCV, {len(tags)} variants x {len(B)} half-years, {len(logits)} splits): {pbo:.3f}')
print(f'DSR {base} at N={n0}: {db:.3f} (SR0 {s0b:.2f}); {cand} at N={n1}: {dc:.3f} (SR0 {s0c:.2f}); sd of trial Sharpes {sd:.2f}')
