#!/usr/bin/env python3
"""Experiment 22b — follow-up on the attribution results (exp21a): the listing sleeve has no alpha beyond the alt−BTC
spread and the leave-one-out book without it is better (1.87 vs 1.84); ML7 is 0.86-correlated with ML3.
Test simplified books (no listing / no ml7 / blended ML) and an alt-season regime overlay, with selection (2022-24)
and confirmation (2025-26) windows. Also the daily book without listing."""
import time, json, math, numpy as np, pandas as pd
from bt import Data, run, quantile_ls, slip_model
from strategies import core_signal
START, END, OOS = '2021-01-01', '2026-08-31 23:00', '2022-01-01'
d = Data(start=START, end=END, min_turn_ever=5e6, load_hl=True, load_premium=True); T, N = d.ret.shape
P3, P7, P8 = np.load('out_exp6_pred_h3.npy'), np.load('out_exp6_pred_h7.npy'), np.load('out_exp17_pred_8h_p0.npy')
VOL = pd.DataFrame(d.ret).rolling(168, min_periods=72).std().values.astype(np.float32)
instr = json.load(open('instruments.json')); launch_h = np.full(N, np.nan)
for j, s in enumerate(d.cols):
    mm = instr.get(str(s))
    if mm and mm.get('launch'): launch_h[j] = mm['launch'] / 1000 / 3600
MAJ5 = [s for s in ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT'] if s in d.cols]; jm = np.array([int(np.flatnonzero(d.cols == s)[0]) for s in MAJ5])
JB = int(np.flatnonzero(d.cols == 'BTCUSDT')[0])
def sl_listing(dd, i, mask):
    age = np.maximum(dd.age[i].astype(float), np.where(np.isfinite(launch_h), dd.hh[i] - launch_h, -np.inf)); t = dd.t24[i]
    cand = (age >= 24 * 7) & (age <= 24 * 90) & dd.valid[i] & np.isfinite(t) & (t >= 2e7) & (dd.age[i] >= 72); cand[jm] = False
    r24 = dd.cff[i] / dd.cff[i - 24] - 1; cand &= ~(r24 > 0.30); idx = np.flatnonzero(cand); w = np.zeros(N)
    if len(idx) == 0: return w
    idx = idx[np.argsort(-t[idx])][:30]; ws = min(1.0 / len(idx), 0.05); w[idx] = -ws; w[jm] += ws * len(idx) / len(jm); return w
def sl_core_f(lb=336, q=0.2):
    def f(dd, i, mask):
        m = dd.universe(i, min_age_h=720, min_turn=1e7, top_n=100); return quantile_ls(core_signal(dd, i, m, lb), q)
    return f
sl_core = sl_core_f()
def make_ml(P, top=0.3, min_age_h=720, iv=False):
    def f(dd, i, mask):
        m = dd.universe(i, min_age_h=min_age_h, min_turn=1e7, top_n=150)
        return quantile_ls(np.where(m & np.isfinite(P[i]), P[i], np.nan), top, inv_vol=VOL[i] if iv else None)
    return f
def ml_blend(dd, i, mask):
    m = dd.universe(i, min_age_h=720, min_turn=1e7, top_n=150)
    def rk(P):
        s = np.where(m & np.isfinite(P[i]), P[i], np.nan); ok = np.isfinite(s); r = np.full(N, np.nan)
        if ok.sum() > 10: rr = np.empty(ok.sum()); rr[np.argsort(s[ok])] = np.arange(ok.sum()); r[ok] = rr / (ok.sum() - 1)
        return r
    return quantile_ls(0.5 * (rk(P3) + rk(P7)), 0.3)
ML3, ML7, ML8 = make_ml(P3), make_ml(P7), make_ml(P8, 0.3, 180 * 24, True)
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
# ---- alt−BTC spread momentum (trailing 30d) per bar, from an equal-weight top-100 alt index (ex BTC/ETH) computed daily
RET0 = np.where(np.isfinite(d.ret), d.ret, np.nan)
day_codes = pd.factorize(pd.Series(d.idx.floor('D')))[0]; ndays = day_codes.max() + 1
alt_d = np.full(ndays, np.nan); btc_d = np.full(ndays, np.nan)
for code in range(ndays):
    rows = np.flatnonzero(day_codes == code)
    if len(rows) < 20: continue
    r = np.nanprod(1 + RET0[rows], axis=0) - 1; r = np.where(np.isfinite(RET0[rows]).sum(0) >= 20, r, np.nan)
    t = d.t24[rows[0]]; ok = np.isfinite(t) & np.isfinite(r) & d.valid[rows[0]]; ok[jm[:2]] = False
    if ok.sum() < 30: continue
    top = np.argsort(np.where(ok, t, -np.inf))[-100:]; alt_d[code] = np.nanmean(r[top]); btc_d[code] = r[JB]
spread_d = pd.Series(alt_d - btc_d).fillna(0.0); SPREAD30 = spread_d.rolling(30, min_periods=20).sum().values   # trailing 30d spread (sum of daily)
def spread_mom(i): v = SPREAD30[day_codes[i] - 1] if day_codes[i] >= 1 else np.nan; return v if np.isfinite(v) else 0.0   # use up to yesterday
print(f'spread30 quantiles: {np.nanpercentile(SPREAD30, [10, 25, 50, 75, 90]).round(3)}', flush=True)
def overlay(parts_spread, parts_neutral, weights_s, weights_n, mode='step', thr=0.0, floor=0.5):
    """scale the spread-exposed sleeves by f(spread momentum): step -> floor when spread30 > thr; linear -> clip(1 - spread30/0.3, floor, 1)."""
    def fn(dd, i, mask):
        sm = spread_mom(i)
        f = (floor if sm > thr else 1.0) if mode == 'step' else float(np.clip(1.0 - sm / 0.3, floor, 1.0))
        w = np.zeros(N); tot = 0.0
        for p, s in zip(parts_spread, weights_s): w += f * s * p(dd, i, mask); tot += f * s
        for p, s in zip(parts_neutral, weights_n): w += s * p(dd, i, mask); tot += s
        return w / tot
    return fn
BASE8 = dict(reb_h=8, reb_offset=0, band=0.3, smooth=0.5, fee_bps=5.5, maker_share=0.7, slip_fn=slip_model(scale=0.3), gross=1.0, pos_cap=0.02, cap_short=0.015,
             cap_exempt=MAJ5, cap_exempt_val=0.30, uni_kwargs=dict(min_age_h=0, min_turn=0, top_n=0), verbose=False, vol_target=0.006, max_lev=3.0, max_gross_x=2.0)
BASE24 = dict(BASE8); BASE24.update(reb_h=24)
rows = []
def sh(x): x = np.asarray(x, float); return float(x.mean() / x.std(ddof=1) * math.sqrt(365)) if len(x) > 30 and x.std(ddof=1) > 0 else float('nan')
def go(label, fn, fam, base=BASE8, **kw):
    t1 = time.time(); d.start_i = int(d.idx.searchsorted(pd.Timestamp(OOS, tz='UTC'))); args = dict(base); args.update(kw); args['label'] = label
    m, eq, hp = run(d, fn, ret_diag=True, **args)
    de = eq.groupby(eq.index.floor('D')).last(); r = de.pct_change().dropna(); sel, conf = sh(r[:'2024-12-31']), sh(r['2025-01-01':])
    rows.append(dict(family=fam, label=label, sharpe=m['sharpe'], sel=sel, conf=conf, cagr=m['cagr'], mdd=m['mdd'], wd=m['worst_day'], turnover=m['turnover_x'], cost=m['fees'] + m['slip'], npos=m['avg_npos']))
    print(f'[{fam:<8}] {label:<56} full {m["sharpe"]:5.2f} | SEL {sel:5.2f} | CONF {conf:5.2f} | CAGR {m["cagr"]*100:5.1f}% MDD {m["mdd"]*100:6.1f}% wd {m["worst_day"]*100:5.1f}% turn {m["turnover_x"]:4.0f}x cost {(m["fees"]+m["slip"])*100:4.1f}% npos {m["avg_npos"]:.0f} | '
          + ' '.join(f'{y}:{v[1]:.1f}' for y, v in m['by_year'].items()) + f' ({time.time()-t1:.0f}s)', flush=True)
    return m
def L(f): return Held(f)
print('\n=== simplified 8h books ===', flush=True)
go('R5 baseline', combo([L(sl_listing), L(sl_core), L(ML3), L(ML7), ML8], [1, 1, 1, 1, 2]), 'base')
go('R4 = core+ml3+ml7+ml8 (1,1,1,2)', combo([L(sl_core), L(ML3), L(ML7), ML8], [1, 1, 1, 2]), 'simplify')
go('R4 core 1.5', combo([L(sl_core), L(ML3), L(ML7), ML8], [1.5, 1, 1, 2]), 'simplify')
go('R4 core 2', combo([L(sl_core), L(ML3), L(ML7), ML8], [2, 1, 1, 2]), 'simplify')
go('R3 = core+ml3+ml8 (1,1,2)', combo([L(sl_core), L(ML3), ML8], [1, 1, 2]), 'simplify')
go('R3 = core+ml3+ml8 (1,1,1.5)', combo([L(sl_core), L(ML3), ML8], [1, 1, 1.5]), 'simplify')
go('R3b = core+blend+ml8 (1,2,2)', combo([L(sl_core), L(ml_blend), ML8], [1, 2, 2]), 'simplify')
go('R3b = core+blend+ml8 (1,1,2)', combo([L(sl_core), L(ml_blend), ML8], [1, 1, 2]), 'simplify')
go('R2 = core+ml8 (1,2)', combo([L(sl_core), ML8], [1, 2]), 'simplify')
go('R2 = core+ml8 (1,1)', combo([L(sl_core), ML8], [1, 1]), 'simplify')
go('R5 listing 0.5', combo([L(sl_listing), L(sl_core), L(ML3), L(ML7), ML8], [0.5, 1, 1, 1, 2]), 'simplify')
go('R4 + ml8 w3', combo([L(sl_core), L(ML3), L(ML7), ML8], [1, 1, 1, 3]), 'simplify')
print('\n=== alt-season overlay (scale spread-exposed sleeves when trailing-30d alt−BTC spread > thr) ===', flush=True)
for thr, fl in ((0.0, 0.5), (0.05, 0.5), (0.0, 0.25), (0.10, 0.5)):
    go(f'R4 step overlay thr {thr:+.2f} floor {fl}', overlay([L(ML3), L(ML7), ML8], [L(sl_core)], [1, 1, 2], [1], 'step', thr, fl), 'overlay')
go('R4 linear overlay floor .5', overlay([L(ML3), L(ML7), ML8], [L(sl_core)], [1, 1, 2], [1], 'linear', 0.0, 0.5), 'overlay')
go('R5 step overlay thr 0 floor .5 (listing incl.)', overlay([L(sl_listing), L(ML3), L(ML7), ML8], [L(sl_core)], [1, 1, 1, 2], [1], 'step', 0.0, 0.5), 'overlay')
print('\n=== daily books (daily engine) ===', flush=True)
go('DAILY baseline (listing+core+ml3+ml7)', combo([sl_listing, sl_core, ML3, ML7], [1, 1, 1, 1]), 'daily', BASE24)
go('DAILY core+ml3+ml7', combo([sl_core, ML3, ML7], [1, 1, 1]), 'daily', BASE24)
go('DAILY core+blend (1,1)', combo([sl_core, ml_blend], [1, 1]), 'daily', BASE24)
go('DAILY core+blend (1,2)', combo([sl_core, ml_blend], [1, 2]), 'daily', BASE24)
go('DAILY core+ml3 (1,1)', combo([sl_core, ML3], [1, 1]), 'daily', BASE24)
df = pd.DataFrame(rows); df.to_csv('out_exp22b.csv', index=False)
base = df[df.family == 'base'].iloc[0]
print(f'\nbaseline SEL {base.sel:.2f} CONF {base.conf:.2f} | candidates better on both windows (>+0.05):')
print(df[(df.sel > base.sel + 0.05) & (df.conf > base.conf + 0.05)][['family', 'label', 'sharpe', 'sel', 'conf', 'mdd', 'turnover', 'cost', 'npos']].round(2).to_string(index=False))
print('DONE')
