#!/usr/bin/env python3
"""Experiment 24a — 'other operas': new alpha families not yet tried, each standalone and added to the final book F,
with SEL (2022-24) / CONF (2025-26) windows. A) funding-rate CHANGE (contrarian) and funding LEVEL carry with 8h holding;
B) BTC->alts lead-lag catch-up at 8h (both signs); C) residual (beta-adjusted) short-term reversal 72h/120h;
D) walk-forward dynamic sleeve allocation (weights ∝ trailing-180d standalone Sharpe) for stability across years."""
import time, json, math, numpy as np, pandas as pd
from bt import Data, run, quantile_ls, slip_model
from strategies import core_signal
from betautil import rolling_std, rolling_beta
START, END, OOS = '2021-01-01', '2026-08-31 23:00', '2022-01-01'
d = Data(start=START, end=END, min_turn_ever=5e6, load_hl=False, load_premium=False); T, N = d.ret.shape
P8 = np.load('out_exp17_pred_8h_p0.npy'); Q3, Q7 = np.load('out_exp21c_pred_h3_noage.npy'), np.load('out_exp21c_pred_h7_noage.npy')
VOL = rolling_std(d.ret, 168, 72)
instr = json.load(open('instruments.json')); launch_h = np.full(N, np.nan)
for j, s in enumerate(d.cols):
    mm = instr.get(str(s))
    if mm and mm.get('launch'): launch_h[j] = mm['launch'] / 1000 / 3600
MAJ5 = [s for s in ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT'] if s in d.cols]; jm = np.array([int(np.flatnonzero(d.cols == s)[0]) for s in MAJ5])
JB = int(np.flatnonzero(d.cols == 'BTCUSDT')[0]); BETA = rolling_beta(d.ret, JB, 720)
FUND0 = np.where(np.isfinite(d.fund), d.fund, 0.0).astype(np.float32); CF = np.cumsum(FUND0, axis=0, dtype=np.float64)
def fsum(i, a, b):                    # sum of funding over bars (i-a, i-b] i.e. last a hours excluding last b hours
    return CF[i - b] - CF[i - a]
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
ML3n, ML7n = make_ml(Q3, min_age_h=90 * 24), make_ml(Q7, min_age_h=90 * 24); ML8 = make_ml(P8, 0.2, 180 * 24, True)
def uni(dd, i): return dd.universe(i, min_age_h=720, min_turn=1e7, top_n=150)
def q(sig, m, top=0.2): return quantile_ls(np.where(m & np.isfinite(sig), sig, np.nan), top)
# ---- A: funding signals
def fund_change(dd, i, mask):      # contrarian to the CHANGE in funding (rising funding = crowding longs)
    m = uni(dd, i); return q(-(fsum(i, 24, 0) - fsum(i, 48, 24)), m)
def fund_level(dd, i, mask):       # carry: short high funding (collect), long negative funding
    m = uni(dd, i); return q(-fsum(i, 72, 0), m)
def fund_level_mom(dd, i, mask):   # funding level + 24h price momentum agreement filter (avoid shorting runaway longs)
    m = uni(dd, i); r24 = dd.cff[i] / dd.cff[i - 24] - 1.0; s = -fsum(i, 72, 0); s = np.where(np.sign(s) == -np.sign(r24), s, np.nan); return q(s, m)
# ---- B: lead-lag catch-up (8h)
def catchup(dd, i, mask):          # alts that lagged BTC's last-8h move: long laggards on BTC up, short laggards on BTC down
    m = uni(dd, i); r8 = dd.cff[i] / dd.cff[i - 8] - 1.0; b = np.where(np.isfinite(BETA[i]), BETA[i], 1.0)
    return q(b * r8[JB] - r8, m)
def anticatchup(dd, i, mask): return -catchup(dd, i, mask)
# ---- C: residual reversal
def resid_rev(h):
    def f(dd, i, mask):
        m = uni(dd, i); r = dd.cff[i] / dd.cff[i - h] - 1.0; b = np.where(np.isfinite(BETA[i]), BETA[i], 1.0)
        return q(-(r - b * r[JB]), m)
    return f
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
    print(f'{label:<60} Sh {m["sharpe"]:5.2f} | SEL {sh(r[:"2024-12-31"]):5.2f} CONF {sh(r["2025-01-01":]):5.2f} | CAGR {m["cagr"]*100:5.1f}% MDD {m["mdd"]*100:6.1f}% wd {m["worst_day"]*100:5.1f}% turn {m["turnover_x"]:4.0f}x cost {(m["fees"]+m["slip"])*100:4.1f}% | '
          + ' '.join(f'{y}:{v[1]:.1f}' for y, v in m['by_year'].items()) + f' ({time.time()-t1:.0f}s)', flush=True)
    return m
print('\n=== F reference (realised de-lever) ===', flush=True)
go('F', combo(F_parts(), [1, 1, 1, 1, 2]), keep='F')
print('\n=== A: funding signals ===', flush=True)
go('A1 funding change contrarian, daily', fund_change, BASE24); go('A1 funding change, 8h (band .5)', fund_change, BASE8)
go('A2 funding level carry, daily', fund_level, BASE24); go('A2 funding level carry, 8h', fund_level, BASE8)
go('A3 funding level + momentum filter, daily', fund_level_mom, BASE24)
go('F + A1 (w1, held daily)', combo(F_parts() + [Held(fund_change)], [1, 1, 1, 1, 2, 1]))
go('F + A2 (w1, 8h)', combo(F_parts() + [fund_level], [1, 1, 1, 1, 2, 1]))
print('\n=== B: BTC -> alts lead-lag catch-up (8h) ===', flush=True)
go('B1 catch-up (long laggards on BTC up)', catchup, BASE8); go('B2 anti catch-up', anticatchup, BASE8)
go('B1 catch-up band .3 smooth .3', catchup, BASE8, band=0.3, smooth=0.3)
print('\n=== C: residual short-term reversal ===', flush=True)
go('C1 residual reversal 72h, daily', resid_rev(72), BASE24); go('C2 residual reversal 120h, daily', resid_rev(120), BASE24)
go('C1 72h daily, smooth .7 band .4', resid_rev(72), BASE24, smooth=0.7, band=0.4)
go('F + C2 (w1, held daily)', combo(F_parts() + [Held(resid_rev(120))], [1, 1, 1, 1, 2, 1]))
print('\n=== D: walk-forward dynamic sleeve allocation ===', flush=True)
for name, fn, base in (('listing', sl_listing, BASE24), ('core', sl_core, BASE24), ('ml3n', ML3n, BASE24), ('ml7n', ML7n, BASE24), ('ml8', ML8, BASE8)):
    go(f'standalone {name}', fn, base, keep=name)
S = pd.DataFrame({k: SER[k] for k in ('listing', 'core', 'ml3n', 'ml7n', 'ml8')})
months = pd.period_range('2022-01', '2026-08', freq='M'); W = {}
for p in months:
    end = p.to_timestamp(how='start') - pd.Timedelta(days=1); win = S[:end].tail(180)
    if len(win) < 120: W[str(p)] = np.array([1, 1, 1, 1, 2.0]); continue
    shs = np.array([sh(win[c].dropna()) for c in S.columns]); base_w = np.array([1, 1, 1, 1, 2.0])
    W[str(p)] = base_w * np.clip(shs, 0.25, 3.0) / 1.0
print('  monthly weights (first/last):', {k: np.round(v, 2).tolist() for k, v in list(W.items())[:1]}, {k: np.round(v, 2).tolist() for k, v in list(W.items())[-1:]})
def dyn(parts):
    def fn(dd, i, mask):
        w8 = W.get(str(pd.Period(dd.idx[i], freq='M')), np.array([1, 1, 1, 1, 2.0])); w = np.zeros(N)
        for f, s in zip(parts, w8): w += s * f(dd, i, mask)
        return w / w8.sum()
    return fn
go('F dynamic (trailing-180d Sharpe tilt, clip .25-3)', dyn(F_parts()), keep='Fdyn')
def dyn_kill(parts):               # pre-registered kill rule: ml8 weight 2 -> 1 -> 0 when its trailing-90d Sharpe <= 0
    def fn(dd, i, mask):
        end = dd.idx[i].floor('D') - pd.Timedelta(days=1); s90 = sh(S['ml8'][:end].tail(90).dropna()) if len(S['ml8'][:end]) > 90 else 1.0
        w8 = np.array([1, 1, 1, 1, 2.0 if s90 > 0 else (1.0 if s90 > -0.5 else 0.0)]); w = np.zeros(N)
        for f, s_ in zip(parts, w8): w += s_ * f(dd, i, mask)
        return w / w8.sum()
    return fn
go('F with ml8 kill rule (90d Sharpe<=0 -> w1, <-0.5 -> w0)', dyn_kill(F_parts()), keep='Fkill')
pd.DataFrame(SER).to_csv('out_exp24a_daily_returns.csv'); print('DONE')
