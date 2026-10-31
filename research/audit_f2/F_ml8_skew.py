#!/usr/bin/env python3
"""Audit F2 (agent F): quantify train/serve skew of the ml8 sleeve on Binance panels with the deployed model v4/models/ml_8h.txt.
Research (exp17 / exp17b, the WF panel used by every backtest and by the harness --ml-panels) rank-normalises the 8h features over
the age>=30d top-150 universe and age_d = Binance data-segment age (data loaded from 2020-11-02 -> every older coin ties at the cap).
The live bot (signals.Sleeves.ml8) rank-normalises over its trade universe (age>=180d top-150) and age_d = Binance onboardDate age.
Variants per decision bar (00/08/16 grid, ml8 trade universe B = age>=180d top150):
  RES   : rank over U30 (age>=30d top150), age_d research  -> predictions on B   (what the backtest traded)
  UNIV  : rank over B,                     age_d research  -> isolates the universe effect
  LIVE  : rank over B,                     age_d 'true'    -> universe + age effect (true age proxy = first bar in the full
          panel since 2020-01 or Bybit launch, whichever older)
Reports Spearman(pred) vs RES and the L1 turnover between the resulting ml8 weight vectors (quantile 0.2, inverse vol).
Read-only; output to stdout. Run: python3 research/audit_f2/F_ml8_skew.py > logs/agent_F2_F_ml8_skew.log"""
import os, sys, json, numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'v4'))
os.chdir(ROOT)
import lightgbm as lgb
from scipy.stats import spearmanr
from bt import Data, quantile_ls
from betautil import rolling_std
from ml_features_8h import compute_features_8h, FEATURES_8H, rank_norm, NEED_H

d = Data(start='2021-01-01', end='2026-08-31 23:00', min_turn_ever=5e6, load_hl=True, load_premium=True)
T, N = d.ret.shape; jb = int(np.flatnonzero(d.cols == 'BTCUSDT')[0])
VOL = rolling_std(d.ret, 168, 72)
mdl = lgb.Booster(model_file=os.path.join(ROOT, 'v4', 'models', 'ml_8h.txt'))
# 'true' age proxy: first valid close in the full panel (panel starts 2020-01) or the Bybit launch, whichever is older
close_full = pd.read_parquet(os.path.join(ROOT, 'data', 'panels', 'close.parquet'), columns=list(d.cols))
first = close_full.apply(lambda c: c.first_valid_index())
first_h = np.array([(pd.Timestamp(x).value // 10**9 // 3600) if x is not None else np.nan for x in first], float)
del close_full
instr = json.load(open('instruments.json'))
launch_h = np.array([instr.get(str(s), {}).get('launch', 0) / 1000 / 3600 or np.nan for s in d.cols], float)
true_birth = np.fmin(first_h, launch_h)

dec = np.flatnonzero(d.hh % 8 == 0); dec = dec[(dec >= NEED_H) & (d.idx[dec] >= pd.Timestamp('2025-09-01', tz='UTC'))][::7]
print(f'decision bars: {len(dec)} ({d.idx[dec[0]]} .. {d.idx[dec[-1]]})', flush=True)

def preds(F, idx):
    X = np.column_stack([F[f][idx] for f in FEATURES_8H]).astype(np.float32)
    return mdl.predict(rank_norm(X).astype(np.float64))

rows = []
for i in dec:
    d.prev_mask = None
    u30 = d.universe(i, min_age_h=720, min_turn=1e7, top_n=150); d.prev_mask = None
    ub = d.universe(i, min_age_h=180 * 24, min_turn=1e7, top_n=150)
    sl = slice(i - NEED_H + 1, i + 1)
    age_res = d.age[i].astype(float)
    age_true = np.where(np.isfinite(true_birth), d.hh[i] - true_birth, age_res)
    F_res = compute_features_8h(d.cff[sl], d.fund[sl], d.t24[sl], d.prem[sl], d.high[sl], d.low[sl], age_res, jb, (d.hh[i] + 1) % 24)
    F_true = dict(F_res); F_true['age_d'] = age_true / 24.0
    i30 = np.flatnonzero(u30); ib = np.flatnonzero(ub)          # column order = alphabetical (training order; bot uses alpha_order)
    p30 = np.full(N, np.nan); p30[i30] = preds(F_res, i30)
    pU = np.full(N, np.nan); pU[ib] = preds(F_res, ib)
    pL = np.full(N, np.nan); pL[ib] = preds(F_true, ib)
    pA = np.full(N, np.nan); pA[i30] = preds(F_true, i30)      # age effect only (research universe)
    res = np.where(ub, p30, np.nan)
    ok = np.isfinite(res) & np.isfinite(pU) & np.isfinite(pL)
    n_tie = int(np.sum(age_res[ib] == age_res[ib].max()))
    w_res = quantile_ls(res, 0.2, inv_vol=VOL[i]); w_u = quantile_ls(pU, 0.2, inv_vol=VOL[i]); w_l = quantile_ls(pL, 0.2, inv_vol=VOL[i])
    w_a = quantile_ls(np.where(ub, pA, np.nan), 0.2, inv_vol=VOL[i])
    rows.append(dict(t=str(d.idx[i]), n30=len(i30), nB=len(ib), n_age_tie_B=n_tie,
                     sp_univ=spearmanr(res[ok], pU[ok])[0], sp_live=spearmanr(res[ok], pL[ok])[0],
                     sp_ageonly=spearmanr(res[ok], np.where(ub, pA, np.nan)[ok])[0],
                     l1_univ=0.5 * np.abs(w_res - w_u).sum(), l1_live=0.5 * np.abs(w_res - w_l).sum(), l1_ageonly=0.5 * np.abs(w_res - w_a).sum()))
    print(rows[-1], flush=True)
df = pd.DataFrame(rows)
print('\nSUMMARY (mean / min):')
for c in ('n30', 'nB', 'n_age_tie_B', 'sp_univ', 'sp_ageonly', 'sp_live', 'l1_univ', 'l1_ageonly', 'l1_live'):
    print(f'  {c:<12} mean {df[c].mean():.3f}  min {df[c].min():.3f}  max {df[c].max():.3f}')
print('l1_* = share of the ml8 sleeve gross (per side sums 1 -> total 2; 0.5*L1 in [0, 2]) that differs from the research weights')
print('DONE')
