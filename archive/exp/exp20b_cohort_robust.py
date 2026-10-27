#!/usr/bin/env python3
"""Experiment 20b: (a) is the 2025-26 P&L of the 8h combination dominated by the young-listing cohort (concentration
lottery)? Attribution restricted to 2025-01-01.. for GOOD (w8 2.0), BAD (w8 2.5) and the daily-only book.
(b) robustification of the ML8 sleeve on the smooth x w8 grid: R1 min_age 180d, R2 min_turn 3e7, R3 inverse-vol
weights, R5 = R1+R3, R4 all ranking sleeves inverse-vol (smooth 0.5 only)."""
import time, json, numpy as np, pandas as pd
from bt import Data, run, quantile_ls, slip_model
from strategies import core_signal
START, END, OOS = '2021-01-01', '2026-08-31 23:00', '2022-01-01'
d = Data(start=START, end=END, min_turn_ever=5e6, load_hl=True, load_premium=True); T, N = d.ret.shape
P3, P7, P8 = np.load('out_exp6_pred_h3.npy'), np.load('out_exp6_pred_h7.npy'), np.load('out_exp17_pred_8h_p0.npy')
t0 = time.time(); VOL = pd.DataFrame(d.ret).rolling(168, min_periods=72).std().values.astype(np.float32); print(f'vol panel {time.time()-t0:.0f}s', flush=True)
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
def sl_core(iv=False):
    def f(dd, i, mask):
        m = dd.universe(i, min_age_h=720, min_turn=1e7, top_n=100); return quantile_ls(core_signal(dd, i, m, 336), 0.2, inv_vol=VOL[i] if iv else None)
    return f
def make_ml(P, top=0.3, min_age_h=720, min_turn=1e7, iv=False):
    def f(dd, i, mask):
        m = dd.universe(i, min_age_h=min_age_h, min_turn=min_turn, top_n=150)
        return quantile_ls(np.where(m & np.isfinite(P[i]), P[i], np.nan), top, inv_vol=VOL[i] if iv else None)
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
def daily_parts(iv=False): return [Held(sl_listing), Held(sl_core(iv)), Held(make_ml(P3, iv=iv)), Held(make_ml(P7, iv=iv))]
BASE = dict(reb_h=8, reb_offset=0, band=0.3, smooth=0.5, fee_bps=5.5, maker_share=0.7, slip_fn=slip_model(scale=0.3), gross=1.0, pos_cap=0.02, cap_short=0.015,
            cap_exempt=MAJ5, cap_exempt_val=0.30, uni_kwargs=dict(min_age_h=0, min_turn=0, top_n=0), verbose=False, vol_target=0.006, max_lev=3.0, max_gross_x=2.0)
def go(label, fn, keep=False, **kw):
    t1 = time.time(); d.start_i = int(d.idx.searchsorted(pd.Timestamp(OOS, tz='UTC'))); args = dict(BASE); args.update(kw); args['label'] = label
    if keep: m, eq, hp = run(d, fn, ret_diag=True, attrib=True, attrib_from='2025-01-01', **args)
    else: m, eq = run(d, fn, **args); hp = None
    by = m['by_year']
    print(f'{label:<46} Sh {m["sharpe"]:5.2f} CAGR {m["cagr"]*100:6.1f}% MDD {m["mdd"]*100:6.1f}% wd {m["worst_day"]*100:5.1f}% turn {m["turnover_x"]:4.0f}x cost {(m["fees"]+m["slip"])*100:.1f}% npos {m["avg_npos"]:.0f} | '
          + ' '.join(f'{y}:{v[1]:.1f}' for y, v in by.items()) + f' ({time.time()-t1:.0f}s)', flush=True)
    return m, eq, hp
# ---------------- (a) cohort concentration 2025-26
first_i = np.where(d.valid.any(0), d.valid.argmax(0), 0); first_ts = d.idx[first_i]
young = np.asarray(first_ts >= pd.Timestamp('2024-07-01', tz='UTC'))
print(f'\n=== (a) 2025-01..2026-08 attribution: young cohort (first bar >= 2024-07-01): {young.sum()} of {N} symbols ===', flush=True)
def cohort(m, tag):
    a = m['attrib']; pnl = (a.price + a.funding - a.cost); ab = pnl.abs()
    top10 = ab.sort_values(ascending=False).head(10)
    print(f'{tag:<28} total {pnl.sum():8.0f} | young {pnl[young].sum():8.0f} old {pnl[~young].sum():8.0f} | share of sum|pnl|: young {ab[young].sum()/ab.sum()*100:4.0f}% top10 {top10.sum()/ab.sum()*100:4.0f}% | names |pnl|>$300: {(ab>300).sum()} (young {(ab[young]>300).sum()})')
    print('   top10: ' + ', '.join(f'{s}{"*" if young[list(d.cols).index(s)] else ""} {pnl[s]:+.0f}' for s in top10.index))
mg, _, _ = go('GOOD w8=2.0', combo(daily_parts() + [make_ml(P8)], [1, 1, 1, 1, 2.0]), keep=True); cohort(mg, 'GOOD w8=2.0')
mb, _, _ = go('BAD  w8=2.5', combo(daily_parts() + [make_ml(P8)], [1, 1, 1, 1, 2.5]), keep=True); cohort(mb, 'BAD w8=2.5')
md, _, _ = go('DAILY only (4 sleeves)', combo(daily_parts(), [1, 1, 1, 1]), keep=True); cohort(md, 'DAILY only')
m8, _, _ = go('ML8 alone', make_ml(P8), keep=True); cohort(m8, 'ML8 alone')
m8a, _, _ = go('ML8 alone age180', make_ml(P8, min_age_h=180 * 24), keep=True); cohort(m8a, 'ML8 alone age180')
m8v, _, _ = go('ML8 alone invvol', make_ml(P8, iv=True), keep=True); cohort(m8v, 'ML8 alone invvol')
# ---------------- (b) robustification grids
rows = []
def grid(tag, make, smooths=(0.4, 0.5, 0.6), w8s=(1.5, 2.0, 2.5), **kw):
    sh = []
    for sm in smooths:
        for w8 in w8s:
            m, _, _ = go(f'{tag} smooth{sm} w8={w8}', make(w8), smooth=sm, **kw); sh.append(m['sharpe'])
            rows.append(dict(tag=tag, smooth=sm, w8=w8, sharpe=m['sharpe'], cagr=m['cagr'], mdd=m['mdd'], y2025=m['by_year'].get(2025, (0, 0))[1], y2026=m['by_year'].get(2026, (0, 0))[1]))
    s = pd.Series(sh); print(f'>>> {tag}: Sharpe mean {s.mean():.2f} min {s.min():.2f} max {s.max():.2f} std {s.std():.2f}\n', flush=True)
print('\n=== (b) robustification grids ===', flush=True)
grid('R1 ML8 age180', lambda w8: combo(daily_parts() + [make_ml(P8, min_age_h=180 * 24)], [1, 1, 1, 1, w8]))
grid('R3 ML8 invvol', lambda w8: combo(daily_parts() + [make_ml(P8, iv=True)], [1, 1, 1, 1, w8]))
grid('R5 ML8 age180+invvol', lambda w8: combo(daily_parts() + [make_ml(P8, min_age_h=180 * 24, iv=True)], [1, 1, 1, 1, w8]))
grid('R2 ML8 turn3e7', lambda w8: combo(daily_parts() + [make_ml(P8, min_turn=3e7)], [1, 1, 1, 1, w8]))
grid('R4 all-IV', lambda w8: combo(daily_parts(iv=True) + [make_ml(P8, iv=True)], [1, 1, 1, 1, w8]), smooths=(0.5,))
grid('R6 all-IV age180', lambda w8: combo(daily_parts(iv=True) + [make_ml(P8, min_age_h=180 * 24, iv=True)], [1, 1, 1, 1, w8]), smooths=(0.5,))
pd.DataFrame(rows).to_csv('out_exp20b_grid.csv', index=False); print('DONE')
