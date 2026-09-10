#!/usr/bin/env python3
"""Experiment 20e: mechanism check. (1) Why does the daily 4-sleeve book fall from 1.54 (daily engine) to 0.75 when
merely re-touched every 8h (exp20b)? E1 daily engine; E2 8h engine smooth .5; E3 8h smooth .794; E4 8h but no trades
in 08/16 slots (band 1e9 outside 00) -> must equal E1 if the engine is consistent; E5 8h no smoothing.
(2) path correlation: R5 w8=2 base vs band0.2 daily returns by year. (3) V8: daily sleeves executed ONLY at 00 and
left to drift, ML8 traded every 8h on top (per-sleeve smoothing) -> smooth x w8 grid."""
import time, json, numpy as np, pandas as pd
from bt import Data, run, quantile_ls, slip_model
from strategies import core_signal
START, END, OOS = '2021-01-01', '2026-08-31 23:00', '2022-01-01'
d = Data(start=START, end=END, min_turn_ever=5e6, load_hl=True, load_premium=True); T, N = d.ret.shape
P3, P7, P8 = np.load('out_exp6_pred_h3.npy'), np.load('out_exp6_pred_h7.npy'), np.load('out_exp17_pred_8h_p0.npy')
VOL = pd.DataFrame(d.ret).rolling(168, min_periods=72).std().values.astype(np.float32)
RET0 = np.where(np.isfinite(d.ret), d.ret, 0.0).astype(np.float32)
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
A180 = 180 * 24
BASE8 = dict(reb_h=8, reb_offset=0, band=0.3, smooth=0.5, fee_bps=5.5, maker_share=0.7, slip_fn=slip_model(scale=0.3), gross=1.0, pos_cap=0.02, cap_short=0.015,
             cap_exempt=MAJ5, cap_exempt_val=0.30, uni_kwargs=dict(min_age_h=0, min_turn=0, top_n=0), verbose=False, vol_target=0.006, max_lev=3.0, max_gross_x=2.0)
BASE24 = dict(BASE8); BASE24.update(reb_h=24)
rows = []
def go(label, fn, base=BASE8, keep=False, **kw):
    t1 = time.time(); d.start_i = int(d.idx.searchsorted(pd.Timestamp(OOS, tz='UTC'))); args = dict(base); args.update(kw); args['label'] = label
    if keep: m, eq, hp = run(d, fn, ret_diag=True, **args)
    else: m, eq = run(d, fn, **args); hp = None
    by = m['by_year']
    rows.append(dict(label=label, sharpe=m['sharpe'], cagr=m['cagr'], mdd=m['mdd'], turnover=m['turnover_x'], **{f'y{y}': v[1] for y, v in by.items()}))
    print(f'{label:<52} Sh {m["sharpe"]:5.2f} CAGR {m["cagr"]*100:6.1f}% MDD {m["mdd"]*100:6.1f}% wd {m["worst_day"]*100:5.1f}% turn {m["turnover_x"]:4.0f}x cost {(m["fees"]+m["slip"])*100:.1f}% npos {m["avg_npos"]:.0f} | '
          + ' '.join(f'{y}:{v[1]:.1f}' for y, v in by.items()) + f' ({time.time()-t1:.0f}s)', flush=True)
    return m, eq, hp
def summary(tag, n):
    s = pd.Series([r['sharpe'] for r in rows[-n:]]); y25 = pd.Series([r.get('y2025', np.nan) for r in rows[-n:]])
    print(f'>>> {tag}: Sharpe mean {s.mean():.2f} min {s.min():.2f} max {s.max():.2f} std {s.std():.2f} | 2025 min {y25.min():.1f}\n', flush=True)
daily24 = combo([sl_listing, sl_core, make_ml(P3), make_ml(P7)], [1, 1, 1, 1])
def daily8(): return combo([Held(sl_listing), Held(sl_core), Held(make_ml(P3)), Held(make_ml(P7))], [1, 1, 1, 1])
print('\n=== (1) re-touch mechanism ===', flush=True)
go('E1 daily book, daily engine, smooth .5', daily24, BASE24)
go('E2 daily book, 8h engine, smooth .5', daily8())
go('E3 daily book, 8h engine, smooth .794', daily8(), smooth=0.5 ** (1 / 3))
# E4: emulate "no trades at 08/16" via a signal wrapper that returns NaN... engine zeroes NaN -> full close. Instead use a
# huge band at non-00 slots: implement by a per-call band hook -> not available; emulate with a wrapper that raises band via kw? Not possible.
# So E4 uses the engine's reb_h=24 path but with VT/PNL bookkeeping identical -> equals E1 by construction; skip.
go('E5 daily book, 8h engine, no smoothing', daily8(), smooth=0.0)
go('E6 daily book, 8h engine, smooth .5, band .6', daily8(), band=0.6)
go('E7 daily book, 8h engine, smooth .794, band .6', daily8(), smooth=0.5 ** (1 / 3), band=0.6)
print('\n=== (2) path correlation R5 w8=2: base vs band0.2 ===', flush=True)
def r5(w8): return combo([Held(sl_listing), Held(sl_core), Held(make_ml(P3)), Held(make_ml(P7)), make_ml(P8, 0.3, min_age_h=A180, iv=True)], [1, 1, 1, 1, w8])
_, eqa, hpa = go('R5 w8=2 base', r5(2.0), keep=True)
_, eqb, hpb = go('R5 w8=2 band0.2', r5(2.0), keep=True, band=0.2)
ra = (hpa / eqa.shift(1)).dropna(); rb = (hpb / eqb.shift(1)).dropna()
da = (1 + ra).groupby(ra.index.floor('D')).prod() - 1; db = (1 + rb).groupby(rb.index.floor('D')).prod() - 1
for y in range(2022, 2027):
    a = da[da.index.year == y]; b = db[db.index.year == y]
    print(f'  {y}: corr(daily rets) {a.corr(b):.2f}  dvol {a.std()*100:.2f}%/{b.std()*100:.2f}%  sum {a.sum()*100:5.1f}%/{b.sum()*100:5.1f}%  | biggest daily gaps: '
          + ', '.join(f'{dt.date()} {(x - y_) * 100:+.1f}' for dt, x, y_ in sorted(zip(a.index, a.values, b.values), key=lambda t: -abs(t[1] - t[2]))[:4]), flush=True)
print('\n=== (3) V8: daily sleeves executed only at 00 (drift), ML8 on top every 8h; per-sleeve smoothing ===', flush=True)
def v8(w8, s8=0.5, ml8_kw=None):
    parts = [Held(sl_listing), Held(sl_core), Held(make_ml(P3)), Held(make_ml(P7))]; ml8 = make_ml(P8, 0.3, **(ml8_kw or dict(min_age_h=A180, iv=True)))
    st = {'d': np.zeros(N), 'm': np.zeros(N), 'last': None}
    def fn(dd, i, mask):
        if dd.hh[i] % 24 == 0:
            wd = np.zeros(N)
            for f in parts: wd += f(dd, i, mask)
            st['d'] = 0.5 * st['d'] + 0.5 * (wd / 4.0)
        elif st['last'] is not None:
            g = np.prod(1.0 + RET0[st['last'] + 1:i + 1], axis=0); st['d'] = st['d'] * g      # drift the daily part with realised returns
        st['last'] = i
        st['m'] = s8 * st['m'] + (1 - s8) * ml8(dd, i, mask)
        return (4.0 * st['d'] + w8 * st['m']) / (4.0 + w8)
    return fn
for sm in (0.4, 0.5, 0.6):
    for w8 in (1.5, 2.0, 2.5):
        go(f'V8 s8={sm} w8={w8}', v8(w8, sm), smooth=0.0)
summary('V8 grid (ML8 age180+IV)', 9)
for w8 in (1.5, 2.0, 2.5):
    go(f'V8 plain-ML8 s8=0.5 w8={w8}', v8(w8, 0.5, dict(min_age_h=720, iv=False)), smooth=0.0)
summary('V8 plain ML8', 3)
go('V8 daily-only drift (w8=0)', v8(0.0, 0.5), smooth=0.0)
pd.DataFrame(rows).to_csv('out_exp20e.csv', index=False); print('DONE')
