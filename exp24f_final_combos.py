#!/usr/bin/env python3
"""Experiment 24f — final candidate books after the 'other operas' round: F (baseline), ML7n->ML14n swap, funding sleeves
A1/A2, age-free 8h model (if exp24c produced it), and their combinations; SEL/CONF windows, cost x1.5/x2, maker 50%,
taker, ml8 weight 1 (pessimistic). Pre-registered choice rule: max min(SEL, CONF) subject to taker Sharpe >= 1.0 and
worst day >= -4%, preferring fewer sleeves and lower turnover within 0.15."""
import os, time, json, math, numpy as np, pandas as pd
from bt import Data, run, quantile_ls, slip_model
from strategies import core_signal
from betautil import rolling_std
START, END, OOS = '2021-01-01', '2026-08-31 23:00', '2022-01-01'
d = Data(start=START, end=END, min_turn_ever=5e6, load_hl=False, load_premium=False); T, N = d.ret.shape
P8 = np.load('out_exp17_pred_8h_p0.npy', mmap_mode='r'); Q3, Q7 = np.load('out_exp21c_pred_h3_noage.npy', mmap_mode='r'), np.load('out_exp21c_pred_h7_noage.npy', mmap_mode='r'); Q14 = np.load('out_exp24b_pred_h14_noage.npy', mmap_mode='r')
P8n = np.load('out_exp24c_pred_8h_noage.npy', mmap_mode='r') if os.path.exists('out_exp24c_pred_8h_noage.npy') else None
VOL = rolling_std(d.ret, 168, 72)
instr = json.load(open('instruments.json')); launch_h = np.full(N, np.nan)
for j, s in enumerate(d.cols):
    mm = instr.get(str(s))
    if mm and mm.get('launch'): launch_h[j] = mm['launch'] / 1000 / 3600
MAJ5 = [s for s in ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT'] if s in d.cols]; jm = np.array([int(np.flatnonzero(d.cols == s)[0]) for s in MAJ5])
FUND0 = np.where(np.isfinite(d.fund), d.fund, 0.0).astype(np.float32); CF = np.cumsum(FUND0.astype(np.float64), axis=0, dtype=np.float64)
def fsum(i, a, b): return CF[i - b] - CF[i - a]
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
def uni(dd, i): return dd.universe(i, min_age_h=720, min_turn=1e7, top_n=150)
def q(sig, m, top=0.2): return quantile_ls(np.where(m & np.isfinite(sig), sig, np.nan), top)
def fund_change(dd, i, mask): return q(-(fsum(i, 24, 0) - fsum(i, 48, 24)), uni(dd, i))
def fund_level(dd, i, mask): return q(-fsum(i, 72, 0), uni(dd, i))
ML3n, ML7n, ML14n = make_ml(Q3, min_age_h=90 * 24), make_ml(Q7, min_age_h=90 * 24), make_ml(Q14, min_age_h=90 * 24)
ML8 = make_ml(P8, 0.2, 180 * 24, True); ML8n = make_ml(P8n, 0.2, 180 * 24, True) if P8n is not None else None
class Held:
    def __init__(self, fn): self.fn = fn; self.w = np.zeros(N)
    def __call__(self, dd, i, mask):
        if dd.hh[i] % 24 == 0: self.w = self.fn(dd, i, mask)
        return self.w
def combo(parts, weights):
    def fn(dd, i, mask):
        w = np.zeros(N)
        for f, s in zip(parts, weights): w += s * f(dd, i, mask)
        return w / sum(weights)
    return fn
def book(ml_slow=ML7n, ml8=ML8, a1=False, a2=False, w8=2.0, wa=1.0):
    parts = [Held(sl_listing), Held(sl_core), Held(ML3n), Held(ml_slow), ml8]; ws = [1, 1, 1, 1, w8]
    if a1: parts.append(Held(fund_change)); ws.append(wa)
    if a2: parts.append(fund_level); ws.append(wa)
    return combo(parts, ws)
BASE8 = dict(reb_h=8, reb_offset=0, band=0.5, smooth=0.5, fee_bps=5.5, maker_share=0.7, slip_fn=slip_model(scale=0.3), gross=1.0, pos_cap=0.02, cap_short=0.015,
             cap_exempt=MAJ5, cap_exempt_val=0.30, uni_kwargs=dict(min_age_h=0, min_turn=0, top_n=0), verbose=False, vol_target=0.006, max_lev=3.0, max_gross_x=2.0)
COST = {'x1': {}, 'x1.5': dict(fee_bps=8.25, maker_fee_bps=3.0, slip_fn=slip_model(scale=0.45)), 'x2': dict(fee_bps=11.0, maker_fee_bps=4.0, slip_fn=slip_model(scale=0.6)),
        'mk50': dict(maker_share=0.5, slip_fn=slip_model(scale=0.5)), 'taker': dict(maker_share=0.0, slip_fn=slip_model(scale=1.0))}
def sh(x): x = np.asarray(x, float); return float(x.mean() / x.std(ddof=1) * math.sqrt(365)) if len(x) > 30 and x.std(ddof=1) > 0 else float('nan')
rows = []
def go(label, fn, cost='x1', **kw):
    t1 = time.time(); d.start_i = int(d.idx.searchsorted(pd.Timestamp(OOS, tz='UTC'))); args = dict(BASE8); args.update(COST[cost]); args.update(kw); args['label'] = label
    m, eq, hp = run(d, fn, ret_diag=True, **args); de = eq.groupby(eq.index.floor('D')).last(); r = de.pct_change().dropna(); sel, conf = sh(r[:'2024-12-31']), sh(r['2025-01-01':])
    rows.append(dict(label=label, cost=cost, sharpe=m['sharpe'], sel=sel, conf=conf, cagr=m['cagr'], mdd=m['mdd'], wd=m['worst_day'], turnover=m['turnover_x'], costpct=m['fees'] + m['slip'], npos=m['avg_npos']))
    print(f'{label:<50} [{cost:<5}] Sh {m["sharpe"]:5.2f} | SEL {sel:5.2f} CONF {conf:5.2f} | CAGR {m["cagr"]*100:5.1f}% MDD {m["mdd"]*100:6.1f}% wd {m["worst_day"]*100:5.1f}% turn {m["turnover_x"]:4.0f}x cost {(m["fees"]+m["slip"])*100:4.1f}% | '
          + ' '.join(f'{y}:{v[1]:.1f}' for y, v in m['by_year'].items()) + f' ({time.time()-t1:.0f}s)', flush=True)
    return m['sharpe']
CANDS = {
    'F  (listing,core,ml3n,ml7n,ml8)': book(),
    'F14 (ml7n -> ml14n)': book(ml_slow=ML14n),
    'F+A1+A2': book(a1=True, a2=True),
    'F14+A1+A2': book(ml_slow=ML14n, a1=True, a2=True),
    'F14+A1': book(ml_slow=ML14n, a1=True),
    'F14+A2': book(ml_slow=ML14n, a2=True),
    'F14+A1+A2 (w .5,.5)': book(ml_slow=ML14n, a1=True, a2=True, wa=0.5),
}
if ML8n is not None:
    CANDS['F14+A1+A2 with 8h age-free model'] = book(ml_slow=ML14n, ml8=ML8n, a1=True, a2=True)
    CANDS['F with 8h age-free model'] = book(ml8=ML8n)
print('\n=== candidates at modelled costs ===', flush=True)
for name, fn in CANDS.items(): go(name, fn)
print('\n=== cost / execution stress ===', flush=True)
for name in ('F  (listing,core,ml3n,ml7n,ml8)', 'F14 (ml7n -> ml14n)', 'F14+A1+A2', 'F+A1+A2'):
    for c in ('x1.5', 'x2', 'mk50', 'taker'): go(name, CANDS[name], c)
print('\n=== pessimistic ml8 (weight 1) and band .4 ===', flush=True)
go('F14+A1+A2 ml8 w1', book(ml_slow=ML14n, a1=True, a2=True, w8=1.0)); go('F14 ml8 w1', book(ml_slow=ML14n, w8=1.0))
go('F14+A1+A2 band .4', CANDS['F14+A1+A2'], band=0.4); go('F14+A1+A2 smooth .6', CANDS['F14+A1+A2'], smooth=0.6)
df = pd.DataFrame(rows); df.to_csv('out_exp24f.csv', index=False)
base = df[(df.cost == 'x1')].copy(); base['minwin'] = base[['sel', 'conf']].min(axis=1)
tk = df[df.cost == 'taker'].set_index('label').sharpe
base['taker'] = base.label.map(tk)
print('\n=== ranking by min(SEL, CONF) with taker Sharpe and worst day ===')
print(base.sort_values('minwin', ascending=False)[['label', 'sharpe', 'sel', 'conf', 'minwin', 'taker', 'wd', 'mdd', 'turnover', 'npos']].round(2).to_string(index=False))
print('DONE')
