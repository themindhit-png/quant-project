#!/usr/bin/env python3
"""Experiment 26 — Opus round 3 research items on the F2 book:
S1 ML8 yearly attribution done right (per-year windows via end_i); S2 'funding not accrued' scenario (P&L without funding,
signals unchanged) for F and F2; S3 maker adverse-selection gradient (1/2/3 bps on maker fills); S4 cost-dependent
per-name rebalance band; S5 fcarry held daily; S6 PREREG rule-4 trigger frequency on history; S7 DSR for F2 and PBO (CSCV)
across the finalist books; S8 2026 contribution of blacklisted (RWA / tokenised stock) names to the listing sleeve."""
import time, json, math, glob, itertools, numpy as np, pandas as pd
from scipy.stats import norm, skew, kurtosis
from bt import Data, run, quantile_ls, slip_model
from strategies import core_signal
from betautil import rolling_std
START, END, OOS = '2021-01-01', '2026-08-31 23:00', '2022-01-01'
d = Data(start=START, end=END, min_turn_ever=5e6, load_hl=False, load_premium=False); T, N = d.ret.shape
P8 = np.load('out_exp17_pred_8h_p0.npy', mmap_mode='r'); Q3, Q7 = np.load('out_exp21c_pred_h3_noage.npy', mmap_mode='r'), np.load('out_exp21c_pred_h7_noage.npy', mmap_mode='r'); Q14 = np.load('out_exp24b_pred_h14_noage.npy', mmap_mode='r')
VOL = rolling_std(d.ret, 168, 72)
instr = json.load(open('instruments.json')); launch_h = np.full(N, np.nan)
for j, s in enumerate(d.cols):
    mm = instr.get(str(s))
    if mm and mm.get('launch'): launch_h[j] = mm['launch'] / 1000 / 3600
MAJ5 = [s for s in ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT'] if s in d.cols]; jm = np.array([int(np.flatnonzero(d.cols == s)[0]) for s in MAJ5]); JB = int(np.flatnonzero(d.cols == 'BTCUSDT')[0])
FUND_REAL = d.fund.copy(); FUND0 = np.where(np.isfinite(d.fund), d.fund, 0.0).astype(np.float64); CF = np.cumsum(FUND0, axis=0, dtype=np.float64)
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
ML3n, ML7n, ML14n, ML8 = make_ml(Q3, min_age_h=90 * 24), make_ml(Q7, min_age_h=90 * 24), make_ml(Q14, min_age_h=90 * 24), make_ml(P8, 0.2, 180 * 24, True)
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
def beta0(fn):
    from betautil import rolling_beta
    B = rolling_beta(d.ret, JB, 720)
    def g(dd, i, mask):
        w = fn(dd, i, mask); b = np.where(np.isfinite(B[i]), B[i], 1.0); w = w.copy(); w[JB] -= float((w * b).sum()); return w
    return g
def F(): return combo([Held(sl_listing), Held(sl_core), Held(ML3n), Held(ML7n), ML8], [1, 1, 1, 1, 2])
def F2(carry_daily=False): return combo([Held(sl_listing), Held(sl_core), Held(ML3n), Held(ML7n), ML8, Held(fund_change), Held(fund_level) if carry_daily else fund_level], [1, 1, 1, 1, 2, 1, 1])
BASE8 = dict(reb_h=8, reb_offset=0, band=0.5, smooth=0.5, fee_bps=5.5, maker_share=0.7, slip_fn=slip_model(scale=0.3), gross=1.0, pos_cap=0.02, cap_short=0.015,
             cap_exempt=MAJ5, cap_exempt_val=0.30, uni_kwargs=dict(min_age_h=0, min_turn=0, top_n=0), verbose=False, vol_target=0.006, max_lev=3.0, max_gross_x=2.0)
BASE24 = dict(BASE8); BASE24.update(reb_h=24, band=0.3)
def sh(x): x = np.asarray(x, float); return float(x.mean() / x.std(ddof=1) * math.sqrt(365)) if len(x) > 30 and x.std(ddof=1) > 0 else float('nan')
SER = {}
def go(label, fn, base=BASE8, keep=None, attrib=False, **kw):
    t1 = time.time(); d.start_i = int(d.idx.searchsorted(pd.Timestamp(OOS, tz='UTC'))); args = dict(base); args.update(kw); args['label'] = label
    m, eq, hp = run(d, fn, ret_diag=True, attrib=attrib, **args); de = eq.groupby(eq.index.floor('D')).last(); r = de.pct_change().dropna()
    if keep: SER[keep] = r
    print(f'{label:<64} Sh {m["sharpe"]:5.2f} | SEL {sh(r[:"2024-12-31"]):5.2f} CONF {sh(r["2025-01-01":]):5.2f} | CAGR {m["cagr"]*100:5.1f}% MDD {m["mdd"]*100:6.1f}% wd {m["worst_day"]*100:5.1f}% turn {m["turnover_x"]:4.0f}x cost {(m["fees"]+m["slip"])*100:4.1f}% fund {m["funding"]*100:+5.1f}% | '
          + ' '.join(f'{y}:{v[1]:.1f}' for y, v in m['by_year'].items()) + f' ({time.time()-t1:.0f}s)', flush=True)
    return m
print('\n=== S1 ML8 standalone attribution PER YEAR (proper windows) ===', flush=True)
for y in range(2022, 2027):
    d.start_i = int(d.idx.searchsorted(pd.Timestamp(f'{y}-01-01', tz='UTC'))); end_i = int(d.idx.searchsorted(pd.Timestamp(f'{y+1}-01-01', tz='UTC'))) if y < 2026 else T - 1
    args = dict(BASE8); args['label'] = f'ml8 {y}'
    m, eq, hp = run(d, ML8, ret_diag=True, attrib=True, end_i=end_i, **args); a = m['attrib']; e0 = 10000.0; yrs = (end_i - d.start_i) / 24 / 365.25
    print(f'  {y}: Sharpe {m["sharpe"]:5.2f} | gross price {a.price.sum()/e0/yrs*100:+6.1f}%/y funding {a.funding.sum()/e0/yrs*100:+5.1f}%/y costs {-a.cost.sum()/e0/yrs*100:6.1f}%/y -> net {(a.price.sum()+a.funding.sum()-a.cost.sum())/e0/yrs*100:+6.1f}%/y | turnover {m["turnover_x"]:.0f}x dvol {m["dvol"]*100:.2f}%', flush=True)
print('\n=== S2 funding NOT accrued by the account (signals unchanged, P&L without funding) ===', flush=True)
go('F  (funding accrued)', F(), keep='F'); go('F2 (funding accrued)', F2(), keep='F2')
d.fund = np.zeros_like(FUND_REAL)
go('F  (funding NOT accrued)', F()); go('F2 (funding NOT accrued)', F2(), keep='F2_nofund')
d.fund = FUND_REAL
print('\n=== S3 maker adverse selection (bps added to every maker fill) ===', flush=True)
for ab in (1.0, 2.0, 3.0): go(f'F2 maker_adverse {ab:.0f} bps', F2(), maker_adverse_bps=ab)
go('F2 maker 50% + adverse 2 bps', F2(), maker_share=0.5, slip_fn=slip_model(scale=0.5), maker_adverse_bps=2.0)
print('\n=== S4 cost-dependent per-name rebalance band (band x clip(slip_bps/2, lo, hi)) ===', flush=True)
base_slip = slip_model(scale=1.0)
def band_fn(lo, hi):
    def f(dd, i): return np.clip(base_slip(dd, i) / 2.0, lo, hi)
    return f
go('F2 band_i = .5 x clip(slip/2, .5, 2)', F2(), band_fn=band_fn(0.5, 2.0)); go('F2 band_i = .5 x clip(slip/2, .75, 1.5)', F2(), band_fn=band_fn(0.75, 1.5)); go('F2 band .4 x clip(slip/2, .5, 2)', F2(), band=0.4, band_fn=band_fn(0.5, 2.0))
print('\n=== S5 fcarry held daily ===', flush=True)
go('F2 with fcarry held daily', F2(carry_daily=True), keep='F2d')
print('\n=== S7 finalist books for PBO (daily returns) ===', flush=True)
K = {}
K['F'] = SER['F']; K['F2'] = SER['F2']
go('F14', combo([Held(sl_listing), Held(sl_core), Held(ML3n), Held(ML14n), ML8], [1, 1, 1, 1, 2]), keep='F14')
go('F14+A1+A2', combo([Held(sl_listing), Held(sl_core), Held(ML3n), Held(ML14n), ML8, Held(fund_change), fund_level], [1, 1, 1, 1, 2, 1, 1]), keep='F14A')
go('F14+A1+A2 (.5,.5)', combo([Held(sl_listing), Held(sl_core), Held(ML3n), Held(ML14n), ML8, Held(fund_change), fund_level], [1, 1, 1, 1, 2, .5, .5]), keep='F14A5')
go('F+A1', combo([Held(sl_listing), Held(sl_core), Held(ML3n), Held(ML7n), ML8, Held(fund_change)], [1, 1, 1, 1, 2, 1]), keep='FA1')
go('F+A2', combo([Held(sl_listing), Held(sl_core), Held(ML3n), Held(ML7n), ML8, fund_level], [1, 1, 1, 1, 2, 1]), keep='FA2')
go('K13 no listing beta0 q.2 band.4', beta0(combo([Held(sl_core), Held(ML3n), Held(ML7n), ML8], [1, 1, 1, 2])), band=0.4, keep='K13')
go('K14 no listing beta0 q.2 band.5', beta0(combo([Held(sl_core), Held(ML3n), Held(ML7n), ML8], [1, 1, 1, 2])), keep='K14')
go('R3b core+ml3+ml8 beta0 band.4', beta0(combo([Held(sl_core), Held(ML3n), ML8], [1, 1, 2])), band=0.4, keep='R3b')
go('D2 daily engine (F2 without ml8)', combo([sl_listing, sl_core, ML3n, ML7n, fund_change, fund_level], [1, 1, 1, 1, 1, 1]), BASE24, keep='D2')
go('DAILY age-free book', combo([sl_listing, sl_core, ML3n, ML7n], [1, 1, 1, 1]), BASE24, keep='D')
S = pd.DataFrame(SER).dropna(how='all')
# ---- S6 PREREG rule 4 frequency: 3 negative months in a row OR trailing 6-month Sharpe < 0
print('\n=== S6 PREREG rule-4 triggers on history (3 negative months in a row; 6-month Sharpe < 0) ===', flush=True)
for k in ('F2', 'F', 'D2', 'D', 'F2_nofund'):
    r = S[k].dropna(); mr = (1 + r).groupby(r.index.to_period('M')).prod() - 1
    neg3 = [str(p) for p in mr.index[2:] if (mr.loc[:p].iloc[-3:] < 0).all()]
    s6 = r.rolling(182).apply(lambda x: x.mean() / x.std(ddof=1) * math.sqrt(365) if x.std(ddof=1) > 0 else 0, raw=True)
    neg6 = s6[s6 < 0]
    print(f'  {k:<9}: negative months {int((mr < 0).sum())}/{len(mr)} | 3-in-a-row triggers at: {neg3 if neg3 else "none"} | days with 6m Sharpe<0: {int((neg6 > -99).sum())} ({neg6.index.min().date() if len(neg6) else "-"} .. {neg6.index.max().date() if len(neg6) else "-"})', flush=True)
# ---- S7 DSR for F2 and PBO (CSCV, 16 blocks) over finalists
print('\n=== S7 DSR (F2) and PBO across finalists ===', flush=True)
trials = []
for f in glob.glob('out_exp19*.csv') + glob.glob('out_exp20*.csv') + glob.glob('out_exp22*.csv') + glob.glob('out_exp24*.csv'):
    try:
        df = pd.read_csv(f); trials += list(df.sharpe.dropna().values) if 'sharpe' in df else []
    except Exception: pass
trials = np.array(trials); v = S['F2'].dropna().values; n = len(v); sd = v.mean() / v.std(ddof=1); S_ann = sd * math.sqrt(365); g3, g4 = skew(v), kurtosis(v, fisher=False)
em = 0.5772156649; Ntr = len(trials); var_tr = (trials.std(ddof=1) / math.sqrt(365)) ** 2
sr0 = math.sqrt(var_tr) * ((1 - em) * norm.ppf(1 - 1 / Ntr) + em * norm.ppf(1 - 1 / (Ntr * math.e)))
dsr = norm.cdf((sd - sr0) * math.sqrt(n - 1) / math.sqrt(1 - g3 * sd + (g4 - 1) / 4 * sd ** 2))
print(f'  F2: Sharpe {S_ann:.2f}, trials N={Ntr} (std {trials.std(ddof=1):.2f}) -> SR0 ann {sr0*math.sqrt(365):.2f}, DSR {dsr:.3f}')
M = S[['F', 'F2', 'F14', 'F14A', 'F14A5', 'FA1', 'FA2', 'K13', 'K14', 'R3b']].dropna(); X = M.values; B = 16; blocks = np.array_split(np.arange(len(X)), B); half = B // 2
logits = []
for comb in itertools.combinations(range(B), half):
    tr = np.concatenate([blocks[b] for b in comb]); te = np.concatenate([blocks[b] for b in range(B) if b not in comb])
    def shp(Z): return Z.mean(0) / (Z.std(0, ddof=1) + 1e-12)
    best = int(np.argmax(shp(X[tr]))); r_te = pd.Series(shp(X[te])).rank().iloc[best] / (X.shape[1] + 1)
    logits.append(math.log(r_te / (1 - r_te)))
logits = np.array(logits); print(f'  PBO (CSCV, {B} blocks, {len(logits)} splits, {X.shape[1]} finalists): {(logits < 0).mean():.3f} | median OOS rank of IS-best: {np.median(1/(1+np.exp(-logits))):.2f} (1 = best)')
# ---- S8 blacklisted names (tokenised stocks / RWA) in 2026 for the listing sleeve
print('\n=== S8 2026 contribution of names the bot blacklists (RWA / tokenised stocks) ===', flush=True)
import re, sys; sys.path.insert(0, 'v4'); import os; os.environ.setdefault('FIRM', 'mubite_8_dd10'); os.environ.setdefault('MODE', 'funded'); os.environ.setdefault('ACCOUNT_SIZE', '10000'); os.environ.setdefault('BYBIT_API_KEY', 'x'); os.environ.setdefault('BYBIT_API_SECRET', 'x')
import config as C
pats = list(C.BLACKLIST_PATTERNS); bl = set(C.BLACKLIST)
black = [str(s) for s in d.cols if s in bl or any(p in str(s) for p in pats)]
print(f'  patterns {pats} + explicit {len(bl)} -> {len(black)} names in data: {black[:20]}{"..." if len(black) > 20 else ""}')
d.start_i = int(d.idx.searchsorted(pd.Timestamp('2026-01-01', tz='UTC'))); args = dict(BASE8); args['label'] = 'F2 2026'
m, eq, hp = run(d, F2(), ret_diag=True, attrib=True, **args); a = m['attrib']; pnl = a.price + a.funding - a.cost
print(f'  F2 2026 (8 months, research incl. blacklisted names): net PnL {pnl.sum():.0f} of start 10000; blacklisted names {pnl[black].sum():.0f} ({pnl[black].sum()/max(abs(pnl.sum()),1)*100:.0f}% of net), Sharpe {m["sharpe"]:.2f}')
S.to_csv('out_exp26_daily_returns.csv'); print('DONE')
