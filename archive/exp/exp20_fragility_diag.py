#!/usr/bin/env python3
"""Experiment 20: WHY is the daily+8h combination fragile? (a) adjacent configs good (smooth .5, w8 2.0) vs bad
(smooth .5, w8 2.5): monthly PnL, VT leverage path, gross, per-symbol 2025 attribution gap; (b) robust variants:
V1 8h forecast as a timing TILT on the daily book (no new names), V6 per-sleeve EMA then sum, V4 ML8 only in the
09/17 slots, V3 concentrated ML8 q10 low weight — each on a mini-grid, reporting mean/min/std of Sharpe."""
import time, json, numpy as np, pandas as pd
from bt import Data, run, quantile_ls, slip_model
from strategies import core_signal
START, END, OOS = '2021-01-01', '2026-08-31 23:00', '2022-01-01'
d = Data(start=START, end=END, min_turn_ever=5e6, load_hl=True, load_premium=True); T, N = d.ret.shape
P3, P7, P8 = np.load('out_exp6_pred_h3.npy'), np.load('out_exp6_pred_h7.npy'), np.load('out_exp17_pred_8h_p0.npy')
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
def make_ml(P, top=0.3):
    def f(dd, i, mask):
        m = dd.universe(i, min_age_h=720, min_turn=1e7, top_n=150); return quantile_ls(np.where(m & np.isfinite(P[i]), P[i], np.nan), top)
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
def daily_parts(): return [Held(sl_listing), Held(sl_core), Held(make_ml(P3)), Held(make_ml(P7))]
BASE = dict(reb_h=8, reb_offset=0, band=0.3, smooth=0.5, fee_bps=5.5, maker_share=0.7, slip_fn=slip_model(scale=0.3), gross=1.0, pos_cap=0.02, cap_short=0.015,
            cap_exempt=MAJ5, cap_exempt_val=0.30, uni_kwargs=dict(min_age_h=0, min_turn=0, top_n=0), verbose=False, vol_target=0.006, max_lev=3.0, max_gross_x=2.0)
def go(label, fn, keep=False, **kw):
    t1 = time.time(); d.start_i = int(d.idx.searchsorted(pd.Timestamp(OOS, tz='UTC'))); args = dict(BASE); args.update(kw); args['label'] = label
    if keep:
        m, eq, hp = run(d, fn, ret_diag=True, attrib=True, **args)
    else:
        m, eq = run(d, fn, **args); hp = None
    by = m['by_year']
    print(f'{label:<46} Sh {m["sharpe"]:5.2f} CAGR {m["cagr"]*100:6.1f}% MDD {m["mdd"]*100:6.1f}% wd {m["worst_day"]*100:5.1f}% turn {m["turnover_x"]:4.0f}x cost {(m["fees"]+m["slip"])*100:.1f}% npos {m["avg_npos"]:.0f} | '
          + ' '.join(f'{y}:{v[1]:.1f}' for y, v in by.items()) + f' ({time.time()-t1:.0f}s)', flush=True)
    return m, eq, hp

# ---------------- (a) diagnosis of two adjacent configs
print('\n=== (a) adjacent configs: good (w8=2.0) vs bad (w8=2.5), smooth 0.5 ===', flush=True)
mg, eqg, hpg = go('GOOD smooth0.5 w8=2.0', combo(daily_parts() + [make_ml(P8)], [1, 1, 1, 1, 2.0]), keep=True)
mb, eqb, hpb = go('BAD  smooth0.5 w8=2.5', combo(daily_parts() + [make_ml(P8)], [1, 1, 1, 1, 2.5]), keep=True)
rg = (hpg / eqg.shift(1)).dropna(); rb = (hpb / eqb.shift(1)).dropna()
mg_m = (1 + rg).groupby(rg.index.strftime('%Y-%m')).prod() - 1; mb_m = (1 + rb).groupby(rb.index.strftime('%Y-%m')).prod() - 1
cmp = pd.DataFrame({'good': mg_m, 'bad': mb_m}); cmp['diff'] = cmp.good - cmp.bad
print('\nmonthly returns (%), months where |diff| > 3%:'); print((cmp[cmp['diff'].abs() > 0.03] * 100).round(1).to_string())
print('\nby year (%):'); print((cmp.groupby(cmp.index.str[:4]).apply(lambda g: (1 + g[['good', 'bad']]).prod() - 1) * 100).round(1).to_string())
lg = pd.DataFrame(mg['lev_hist'], columns=['i', 'lev', 'gross', 'npos']); lb = pd.DataFrame(mb['lev_hist'], columns=['i', 'lev', 'gross', 'npos'])
lg['ts'] = d.idx[lg.i.values]; lb['ts'] = d.idx[lb.i.values]
print('\nVT leverage & gross by year (good | bad):')
for y in range(2022, 2027):
    a = lg[lg.ts.dt.year == y]; b = lb[lb.ts.dt.year == y]
    print(f'  {y}: lev {a.lev.mean():.2f}/{b.lev.mean():.2f}  gross {a.gross.mean():.2f}/{b.gross.mean():.2f}  npos {a.npos.mean():.0f}/{b.npos.mean():.0f}')
ag = mg['attrib']; ab = mb['attrib']
gap = (ag.price + ag.funding - ag.cost) - (ab.price + ab.funding - ab.cost)
print('\ntop symbols explaining the gap (good - bad, USD over full period):'); print(gap.sort_values(ascending=False).head(10).round(0).to_string())
print(gap.sort_values().head(10).round(0).to_string())
print(f'\nworst days good: ' + ' | '.join(f'{dte} {p:+.1f}% ({top[0][0]} {top[0][1]:+.1f})' for dte, p, top in mg['bad_days'][:5]))
print(f'worst days bad : ' + ' | '.join(f'{dte} {p:+.1f}% ({top[0][0]} {top[0][1]:+.1f})' for dte, p, top in mb['bad_days'][:5]), flush=True)

# ---------------- (b) robust variants
def grid(tag, make, values, **kw):
    sh = []
    for v in values:
        m, _, _ = go(f'{tag} {v}', make(v), **kw); sh.append(m['sharpe'])
    s = pd.Series(sh); print(f'>>> {tag}: Sharpe mean {s.mean():.2f} min {s.min():.2f} max {s.max():.2f} std {s.std():.2f}\n', flush=True)

# V1: 8h forecast as a timing tilt of the daily book: w_i *= (1 + lam * z8_i * sign(w_i)) clipped to [1-lam, 1+lam]
def tilt(lam):
    parts = daily_parts()
    def fn(dd, i, mask):
        w = np.zeros(N)
        for f in parts: w += f(dd, i, mask)
        w /= 4.0
        s = P8[i]
        if np.isfinite(s).any():
            z = (s - np.nanmean(s)) / (np.nanstd(s) + 1e-12); z = np.where(np.isfinite(z), np.clip(z, -2, 2) / 2.0, 0.0)
            w = w * (1.0 + lam * z * np.sign(w))
        return w
    return fn
print('\n=== V1: 8h forecast as timing tilt of the daily book ===', flush=True)
grid('V1 tilt lam', tilt, [0.3, 0.5, 0.7, 1.0])

# V6: per-sleeve EMA (daily sleeves smoothed once a day at 0.5; ML8 smoothed per 8h step at s8), engine smoothing off
def per_sleeve(w8, s8=0.5):
    parts = daily_parts(); st = {'d': np.zeros(N), 'm': np.zeros(N), 'day': -1}
    ml8 = make_ml(P8)
    def fn(dd, i, mask):
        wd = np.zeros(N)
        for f in parts: wd += f(dd, i, mask)
        wd /= 4.0
        if dd.hh[i] % 24 == 0: st['d'] = 0.5 * st['d'] + 0.5 * wd
        st['m'] = s8 * st['m'] + (1 - s8) * ml8(dd, i, mask)
        return (4.0 * st['d'] + w8 * st['m']) / (4.0 + w8)
    return fn
print('\n=== V6: per-sleeve EMA then sum (engine smoothing off) ===', flush=True)
grid('V6 w8', per_sleeve, [1.5, 2.0, 2.5, 3.0], smooth=0.0)

# V4: ML8 only in the 09/17 slots (hh%24 != 0), daily sleeves as before
def slots(w8):
    parts = daily_parts(); ml8 = make_ml(P8); last = {'w': np.zeros(N)}
    def fn(dd, i, mask):
        wd = np.zeros(N)
        for f in parts: wd += f(dd, i, mask)
        m8 = ml8(dd, i, mask) if dd.hh[i] % 24 != 0 else np.zeros(N)
        return (wd + w8 * m8) / (4.0 + w8)
    return fn
print('\n=== V4: ML8 only in 09/17 slots ===', flush=True)
grid('V4 w8', slots, [1.5, 2.0, 2.5])

# V3: concentrated ML8 (q10) with low weight
def conc(w8):
    return combo(daily_parts() + [make_ml(P8, 0.1)], [1, 1, 1, 1, w8])
print('\n=== V3: ML8 q10 concentrated ===', flush=True)
grid('V3 q10 w8', conc, [0.5, 1.0, 1.5])
# reference grid again for the same seeds/engine (sanity)
print('\n=== REF: standard combo, w8 grid ===', flush=True)
grid('REF w8', lambda w8: combo(daily_parts() + [make_ml(P8)], [1, 1, 1, 1, w8]), [1.5, 2.0, 2.5])
print('DONE')
