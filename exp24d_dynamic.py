#!/usr/bin/env python3
"""Experiment 24d — walk-forward dynamic sleeve allocation and the pre-registered ml8 kill rule (section D of exp24a,
re-run standalone after a tz bug): weights ∝ trailing-180d standalone Sharpe (clip .25–3); kill rule: ml8 weight 2 -> 1 -> 0
when its trailing-90d standalone Sharpe <= 0 / <= -0.5, restore after 90d > 0.5; also 'ml8 weight 1' and 'ml8 off' for reference."""
import time, json, math, numpy as np, pandas as pd
from bt import Data, run, quantile_ls, slip_model
from strategies import core_signal
from betautil import rolling_std
START, END, OOS = '2021-01-01', '2026-08-31 23:00', '2022-01-01'
d = Data(start=START, end=END, min_turn_ever=5e6, load_hl=False, load_premium=False); T, N = d.ret.shape
P8 = np.load('out_exp17_pred_8h_p0.npy'); Q3, Q7 = np.load('out_exp21c_pred_h3_noage.npy'), np.load('out_exp21c_pred_h7_noage.npy'); VOL = rolling_std(d.ret, 168, 72)
instr = json.load(open('instruments.json')); launch_h = np.full(N, np.nan)
for j, s in enumerate(d.cols):
    mm = instr.get(str(s))
    if mm and mm.get('launch'): launch_h[j] = mm['launch'] / 1000 / 3600
MAJ5 = [s for s in ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT'] if s in d.cols]; jm = np.array([int(np.flatnonzero(d.cols == s)[0]) for s in MAJ5])
def sl_listing(dd, i, mask):
    age = np.maximum(dd.age[i].astype(float), np.where(np.isfinite(launch_h), dd.hh[i] - launch_h, -np.inf)); t = dd.t24[i]
    cand = (age >= 24 * 7) & (age <= 24 * 90) & dd.valid[i] & np.isfinite(t) & (t >= 2e7) & (dd.age[i] >= 72); cand[jm] = False
    r24 = dd.cff[i] / dd.cff[i - 24] - 1; cand &= ~(r24 > 0.30); idx = np.flatnonzero(cand); w = np.zeros(N)
    if len(idx) == 0: return w
    idx = idx[np.argsort(-t[idx])][:30]; ws = min(1.0 / len(idx), 0.05); w[idx] = -ws; w[jm] += ws * len(idx) / len(jm); return w
def sl_core(dd, i, mask):
    m = dd.universe(i, min_age_h=720, min_turn=1e7, top_n=100); return quantile_ls(core_signal(dd, i, m, 336), 0.2)
def make_ml(P, top=0.3, min_age_h=720, iv=False):
    def f(dd, i, mask):
        m = dd.universe(i, min_age_h=min_age_h, min_turn=1e7, top_n=150)
        return quantile_ls(np.where(m & np.isfinite(P[i]), P[i], np.nan), top, inv_vol=VOL[i] if iv else None)
    return f
ML3n, ML7n, ML8 = make_ml(Q3, min_age_h=90 * 24), make_ml(Q7, min_age_h=90 * 24), make_ml(P8, 0.2, 180 * 24, True)
class Held:
    def __init__(self, fn): self.fn = fn; self.w = np.zeros(N)
    def __call__(self, dd, i, mask):
        if dd.hh[i] % 24 == 0: self.w = self.fn(dd, i, mask)
        return self.w
def F_parts(): return [Held(sl_listing), Held(sl_core), Held(ML3n), Held(ML7n), ML8]
BASE8 = dict(reb_h=8, reb_offset=0, band=0.5, smooth=0.5, fee_bps=5.5, maker_share=0.7, slip_fn=slip_model(scale=0.3), gross=1.0, pos_cap=0.02, cap_short=0.015,
             cap_exempt=MAJ5, cap_exempt_val=0.30, uni_kwargs=dict(min_age_h=0, min_turn=0, top_n=0), verbose=False, vol_target=0.006, max_lev=3.0, max_gross_x=2.0)
BASE24 = dict(BASE8); BASE24.update(reb_h=24, band=0.3)
def sh(x): x = np.asarray(x, float); return float(x.mean() / x.std(ddof=1) * math.sqrt(365)) if len(x) > 30 and x.std(ddof=1) > 0 else float('nan')
SER = {}
def go(label, fn, base=BASE8, keep=None, **kw):
    t1 = time.time(); d.start_i = int(d.idx.searchsorted(pd.Timestamp(OOS, tz='UTC'))); args = dict(base); args.update(kw); args['label'] = label
    m, eq, hp = run(d, fn, ret_diag=True, **args); de = eq.groupby(eq.index.floor('D')).last(); r = de.pct_change().dropna()
    if keep: SER[keep] = r
    print(f'{label:<62} Sh {m["sharpe"]:5.2f} | SEL {sh(r[:"2024-12-31"]):5.2f} CONF {sh(r["2025-01-01":]):5.2f} | CAGR {m["cagr"]*100:5.1f}% MDD {m["mdd"]*100:6.1f}% wd {m["worst_day"]*100:5.1f}% turn {m["turnover_x"]:4.0f}x cost {(m["fees"]+m["slip"])*100:4.1f}% | '
          + ' '.join(f'{y}:{v[1]:.1f}' for y, v in m['by_year'].items()) + f' ({time.time()-t1:.0f}s)', flush=True)
    return m
def combo(parts, weights):
    def fn(dd, i, mask):
        w = np.zeros(N)
        for f, s in zip(parts, weights): w += s * f(dd, i, mask)
        return w / sum(weights)
    return fn
print('\n=== standalone sleeves (for trailing statistics) ===', flush=True)
for name, fn, base in (('listing', sl_listing, BASE24), ('core', sl_core, BASE24), ('ml3n', ML3n, BASE24), ('ml7n', ML7n, BASE24), ('ml8', ML8, BASE8)):
    go(f'standalone {name}', fn, base, keep=name)
S = pd.DataFrame({k: SER[k] for k in ('listing', 'core', 'ml3n', 'ml7n', 'ml8')})
print('\n=== F and the pre-registered variants ===', flush=True)
go('F (weights 1,1,1,1,2)', combo(F_parts(), [1, 1, 1, 1, 2]), keep='F')
go('F ml8 weight 1', combo(F_parts(), [1, 1, 1, 1, 1]), keep='F1')
go('F ml8 off (daily age-free on 8h grid, band .5)', combo(F_parts()[:4], [1, 1, 1, 1]), keep='F0')
months = pd.period_range('2022-01', '2026-08', freq='M'); W = {}
for p in months:
    end = (p.to_timestamp(how='start') - pd.Timedelta(days=1)).tz_localize('UTC'); win = S[:end].tail(180)
    if len(win) < 120: W[str(p)] = np.array([1, 1, 1, 1, 2.0]); continue
    shs = np.array([sh(win[c].dropna()) for c in S.columns]); W[str(p)] = np.array([1, 1, 1, 1, 2.0]) * np.clip(np.nan_to_num(shs, nan=1.0), 0.25, 3.0)
print('  dynamic weights sample: ' + '; '.join(f'{k}: {np.round(v, 2).tolist()}' for k, v in list(W.items())[::12]))
def dyn(parts):
    def fn(dd, i, mask):
        w8 = W.get(str(pd.Period(dd.idx[i], freq='M')), np.array([1, 1, 1, 1, 2.0])); w = np.zeros(N)
        for f, s in zip(parts, w8): w += s * f(dd, i, mask)
        return w / w8.sum()
    return fn
go('F dynamic (weights x clip(trailing-180d Sharpe, .25, 3))', dyn(F_parts()), keep='Fdyn')
ml8r = S['ml8'].dropna(); dates = ml8r.index
sh90 = ml8r.rolling(90, min_periods=60).apply(lambda x: x.mean() / x.std(ddof=1) * math.sqrt(365) if x.std(ddof=1) > 0 else 0.0, raw=True)
state = pd.Series(2.0, index=dates); mult = 2.0; last_change = None
for t in dates:
    s = sh90.get(t, np.nan)
    if np.isfinite(s) and (last_change is None or (t - last_change).days >= 30):
        new = mult
        if mult == 2.0 and s <= 0: new = 1.0
        elif mult == 1.0 and (s <= -0.5 or s <= 0): new = 0.0 if s <= -0.5 else 1.0 if False else (0.0 if s <= 0 else 1.0)
        if mult < 2.0 and s > 0.5 and (last_change is None or (t - last_change).days >= 90): new = min(2.0, mult + 1.0)
        if new != mult: mult = new; last_change = t
    state[t] = mult
print('  kill-rule ml8 weight path: ' + ' '.join(f'{y}: {state[state.index.year == y].mean():.2f}' for y in range(2022, 2027)) + f' | share of days at 2/1/0: {(state==2).mean():.2f}/{(state==1).mean():.2f}/{(state==0).mean():.2f}')
def killed(parts):
    def fn(dd, i, mask):
        t = dd.idx[i].floor('D') - pd.Timedelta(days=1); w8 = float(state[:t].iloc[-1]) if len(state[:t]) else 2.0
        ws = [1, 1, 1, 1, w8]; w = np.zeros(N)
        for f, s in zip(parts, ws): w += s * f(dd, i, mask)
        return w / sum(ws)
    return fn
go('F with ml8 kill rule (2->1->0 on 90d Sharpe<=0; restore after 90d>0.5)', killed(F_parts()), keep='Fkill')
pd.DataFrame(SER).to_csv('out_exp24d_daily_returns.csv'); print('DONE')
