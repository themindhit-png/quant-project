#!/usr/bin/env python3
"""Experiment 21a — Opus review battery on the fixed-VT books (OOS 2022-01→2026-08, maker 70%, band .3, smooth .5):
T1 standalone sleeves (Sharpe by year), correlation matrix, regressions on BTC and (alt index − BTC) with residual alpha;
T2 leave-one-out books; T3 alt/BTC regime split; T4 signal lags 1h/8h/24h; T5 drop top-20 days / top-20 names;
T6 cost stress (listing ×3/×5 slippage, age-dependent spread, fees ×2/×3); T7 Sharpe SE / PSR / Deflated Sharpe;
T8 book-beta cap (hedge only the excess beyond ±cap)."""
import time, json, glob, math, numpy as np, pandas as pd
from scipy.stats import norm, skew, kurtosis
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
def sl_core(dd, i, mask):
    m = dd.universe(i, min_age_h=720, min_turn=1e7, top_n=100); return quantile_ls(core_signal(dd, i, m, 336), 0.2)
def make_ml(P, top=0.3, min_age_h=720, iv=False):
    def f(dd, i, mask):
        m = dd.universe(i, min_age_h=min_age_h, min_turn=1e7, top_n=150)
        return quantile_ls(np.where(m & np.isfinite(P[i]), P[i], np.nan), top, inv_vol=VOL[i] if iv else None)
    return f
ML3, ML7, ML8 = make_ml(P3), make_ml(P7), make_ml(P8, 0.3, 180 * 24, True)
class Held:
    """daily sleeve on the 8h grid: recompute at 00 bars (optionally from a lagged bar), hold otherwise."""
    def __init__(self, fn, lag=0): self.fn = fn; self.lag = lag; self.w = np.zeros(N)
    def __call__(self, dd, i, mask):
        if dd.hh[i] % 24 == 0: self.w = self.fn(dd, i - self.lag, mask)
        return self.w
def lagged(fn, lag): return lambda dd, i, mask: fn(dd, i - lag, mask)
def combo(parts, weights, black=None):
    def fn(dd, i, mask):
        w = np.zeros(N)
        for f, s in zip(parts, weights): w += s * f(dd, i, mask)
        w = w / sum(weights)
        if black is not None: w[black] = 0.0
        return w
    return fn
def book5(w8=2.0, lag=0, black=None): return combo([Held(sl_listing, lag), Held(sl_core, lag), Held(ML3, lag), Held(ML7, lag), lagged(ML8, lag)], [1, 1, 1, 1, w8], black)
def book4(lag=0, black=None): return combo([lagged(sl_listing, lag), lagged(sl_core, lag), lagged(ML3, lag), lagged(ML7, lag)], [1, 1, 1, 1], black)
BASE8 = dict(reb_h=8, reb_offset=0, band=0.3, smooth=0.5, fee_bps=5.5, maker_share=0.7, slip_fn=slip_model(scale=0.3), gross=1.0, pos_cap=0.02, cap_short=0.015,
             cap_exempt=MAJ5, cap_exempt_val=0.30, uni_kwargs=dict(min_age_h=0, min_turn=0, top_n=0), verbose=False, vol_target=0.006, max_lev=3.0, max_gross_x=2.0)
BASE24 = dict(BASE8); BASE24.update(reb_h=24)
SER = {}
def go(label, fn, base=BASE8, keep=None, attrib=False, **kw):
    t1 = time.time(); d.start_i = int(d.idx.searchsorted(pd.Timestamp(OOS, tz='UTC'))); args = dict(base); args.update(kw); args['label'] = label
    if keep or attrib:
        m, eq, hp = run(d, fn, ret_diag=True, attrib=attrib, **args)
        if keep:
            de = eq.groupby(eq.index.floor('D')).last(); SER[keep] = de.pct_change().dropna()
    else: m, eq = run(d, fn, **args)
    by = m['by_year']
    print(f'{label:<48} Sh {m["sharpe"]:5.2f} CAGR {m["cagr"]*100:6.1f}% dvol {m["dvol"]*100:.2f}% MDD {m["mdd"]*100:6.1f}% wd {m["worst_day"]*100:5.1f}% turn {m["turnover_x"]:4.0f}x cost {(m["fees"]+m["slip"])*100:4.1f}% | '
          + ' '.join(f'{y}:{v[1]:.1f}' for y, v in by.items()) + f' ({time.time()-t1:.0f}s)', flush=True)
    return m
def sh(x): x = np.asarray(x, float); return float(x.mean() / x.std(ddof=1) * math.sqrt(365)) if x.std(ddof=1) > 0 else float('nan')

# ---------------- market factors: BTC daily return, equal-weight alt index (top-100 by turnover, ex BTC/ETH), spread
RET0 = np.where(np.isfinite(d.ret), d.ret, np.nan)
days = pd.Series(d.idx.floor('D')); day_codes = pd.factorize(days)[0]
i0 = int(d.idx.searchsorted(pd.Timestamp(OOS, tz='UTC')))
btc_d, alt_d, day_ts = [], [], []
for code in np.unique(day_codes[i0:]):
    rows = np.flatnonzero(day_codes == code)
    if len(rows) < 20: continue
    r = np.nanprod(1 + RET0[rows], axis=0) - 1; r = np.where(np.isfinite(RET0[rows]).sum(0) >= 20, r, np.nan)
    t = d.t24[rows[0]]; ok = np.isfinite(t) & np.isfinite(r) & d.valid[rows[0]]; ok[jm[:2]] = False
    top = np.argsort(np.where(ok, t, -np.inf))[-100:]
    alt_d.append(float(np.nanmean(r[top]))); btc_d.append(float(r[JB])); day_ts.append(d.idx[rows[0]].floor('D'))
FAC = pd.DataFrame({'btc': btc_d, 'alt': alt_d}, index=pd.DatetimeIndex(day_ts)); FAC['spread'] = FAC.alt - FAC.btc
del RET0; import gc; gc.collect()
print(f'\nfactors: {len(FAC)} days | alt idx ann ret {FAC.alt.mean()*365*100:.0f}% | spread ann {FAC.spread.mean()*365*100:+.0f}% | by year spread: '
      + ' '.join(f'{y}:{v*100:+.0f}%' for y, v in FAC.spread.groupby(FAC.index.year).sum().items()), flush=True)

# ---------------- T1 standalone sleeves + books
print('\n=== T1 standalone sleeves (direct VT) ===', flush=True)
go('listing alone (daily)', sl_listing, BASE24, keep='listing')
go('core alone (daily)', sl_core, BASE24, keep='core')
go('ml3 alone (daily)', ML3, BASE24, keep='ml3')
go('ml7 alone (daily)', ML7, BASE24, keep='ml7')
go('ml8 alone (8h, age180+IV)', ML8, BASE8, keep='ml8')
go('DAILY book (4 sleeves)', book4(), BASE24, keep='book4')
m5 = go('R5 book (5 sleeves, w8=2)', book5(), BASE8, keep='book5', attrib=True)
S = pd.DataFrame(SER).dropna(how='all')
print('\ncorrelation of daily returns:'); print(S.corr().round(2).to_string())
print('\nSharpe by year (standalone):')
tab = {k: {y: sh(v[v.index.year == y]) for y in range(2022, 2027)} for k, v in S.items()}
print(pd.DataFrame(tab).T.round(2).to_string())
print('\nregressions of daily returns on BTC and (alt − BTC) spread: alpha ann %, betas (t), R²:')
for k, v in S.items():
    df = pd.concat([v.rename('y'), FAC], axis=1, join='inner').dropna()
    X = np.column_stack([np.ones(len(df)), df.btc.values, df.spread.values]); y = df.y.values
    b, *_ = np.linalg.lstsq(X, y, rcond=None); res = y - X @ b; s2 = res @ res / (len(y) - 3); cov = s2 * np.linalg.inv(X.T @ X); se = np.sqrt(np.diag(cov))
    r2 = 1 - res.var() / y.var()
    print(f'  {k:<8} alpha {b[0]*365*100:+6.1f}%/y (t {b[0]/se[0]:+.1f}) | beta_BTC {b[1]:+.3f} (t {b[1]/se[1]:+.1f}) | beta_spread {b[2]:+.3f} (t {b[2]/se[2]:+.1f}) | R² {r2:.3f} | raw Sharpe {sh(y):.2f} residual Sharpe {sh(res + b[0]):.2f}')

# ---------------- T2 leave-one-out
print('\n=== T2 leave-one-out ===', flush=True)
for name, parts in [('R5 − listing', [Held(sl_core), Held(ML3), Held(ML7), ML8]), ('R5 − core', [Held(sl_listing), Held(ML3), Held(ML7), ML8]),
                    ('R5 − ml3', [Held(sl_listing), Held(sl_core), Held(ML7), ML8]), ('R5 − ml7', [Held(sl_listing), Held(sl_core), Held(ML3), ML8]),
                    ('R5 − ml3 − ml7', [Held(sl_listing), Held(sl_core), ML8])]:
    ws = [1] * (len(parts) - 1) + [2]; go(name, combo(parts, ws), BASE8)
go('R5 − ml8 (= daily held on 8h)', combo([Held(sl_listing), Held(sl_core), Held(ML3), Held(ML7)], [1, 1, 1, 1]), BASE8)
for name, parts in [('DAILY − listing', [sl_core, ML3, ML7]), ('DAILY − core', [sl_listing, ML3, ML7]), ('DAILY − ml3 − ml7', [sl_listing, sl_core])]:
    go(name, combo(parts, [1] * len(parts)), BASE24)

# ---------------- T3 regime split (monthly)
print('\n=== T3 alt/BTC regime split (monthly returns) ===', flush=True)
for k in ('book5', 'book4', 'listing', 'core', 'ml3', 'ml8'):
    v = S[k].dropna(); mv = (1 + v).groupby(v.index.to_period('M')).prod() - 1
    fs = FAC.spread.groupby(FAC.index.to_period('M')).sum(); fb = FAC.btc.groupby(FAC.index.to_period('M')).sum()
    dfm = pd.concat([mv.rename('r'), fs.rename('spread'), fb.rename('btc')], axis=1).dropna()
    up = dfm[dfm.spread > 0]; dn = dfm[dfm.spread <= 0]; bu = dfm[dfm.btc > 0]; bd = dfm[dfm.btc <= 0]
    print(f'  {k:<8} months {len(dfm)} | spread↑ ({len(up)}): mean {up.r.mean()*100:+.2f}% hit {(up.r>0).mean()*100:.0f}% | spread↓ ({len(dn)}): mean {dn.r.mean()*100:+.2f}% hit {(dn.r>0).mean()*100:.0f}% | '
          f'BTC↑: {bu.r.mean()*100:+.2f}% | BTC↓: {bd.r.mean()*100:+.2f}% | corr(r, spread) {dfm.r.corr(dfm.spread):+.2f} corr(r, btc) {dfm.r.corr(dfm.btc):+.2f}')

# ---------------- T4 lags
print('\n=== T4 signal lag (decide on bar i−lag, trade at bar i) ===', flush=True)
for lag in (1, 8, 24): go(f'R5 lag {lag}h', book5(lag=lag), BASE8)
for lag in (1, 24): go(f'DAILY lag {lag}h', book4(lag=lag), BASE24)

# ---------------- T5 drop top days / names
print('\n=== T5 drop best days / best names ===', flush=True)
for k in ('book5', 'book4'):
    v = S[k].dropna(); srt = v.sort_values()
    print(f'  {k}: Sharpe all {sh(v):.2f} | without 20 best days {sh(srt.iloc[:-20]):.2f} | without 20 worst days {sh(srt.iloc[20:]):.2f} | without both {sh(srt.iloc[20:-20]):.2f} | '
          f'best 20 days sum {srt.iloc[-20:].sum()*100:.0f}% of total {v.sum()*100:.0f}%')
a = m5['attrib']; pnl = a.price + a.funding - a.cost
top20 = list(pnl.sort_values(ascending=False).index[:20]); bot20 = list(pnl.sort_values().index[:20])
print(f'  R5 top-20 names by PnL: {top20[:10]}… sum {pnl[top20].sum():.0f} of total {pnl.sum():.0f} USD')
go('R5 without top-20 names', book5(black=np.array([list(d.cols).index(s) for s in top20])), BASE8)
go('R5 without worst-20 names', book5(black=np.array([list(d.cols).index(s) for s in bot20])), BASE8)
maj_black = np.array([list(d.cols).index(s) for s in top20 if s in MAJ5])
print(f'  (majors among top-20: {[s for s in top20 if s in MAJ5]})')

# ---------------- T6 cost stress
print('\n=== T6 cost stress ===', flush=True)
base_slip = slip_model(scale=0.3)
def slip_age(k=4.0, tau=20.0):
    def f(dd, i):
        s = base_slip(dd, i); age_d = dd.age[i].astype(float) / 24.0
        return s * (1.0 + k * np.exp(-np.maximum(age_d, 0) / tau))
    return f
go('listing alone, slip ×3', sl_listing, BASE24, slip_fn=slip_model(scale=0.9))
go('listing alone, slip ×5', sl_listing, BASE24, slip_fn=slip_model(scale=1.5))
go('listing alone, age-spread (5× at listing → 1× by ~60d)', sl_listing, BASE24, slip_fn=slip_age())
go('DAILY book, age-spread', book4(), BASE24, slip_fn=slip_age())
go('R5 book, age-spread', book5(), BASE8, slip_fn=slip_age())
go('R5 book, age-spread ×2 (9× at listing)', book5(), BASE8, slip_fn=slip_age(k=8.0))
go('R5 book, fees ×2 + slip ×2', book5(), BASE8, fee_bps=11.0, maker_fee_bps=4.0, slip_fn=slip_model(scale=0.6))
go('R5 book, fees ×3 + slip ×3', book5(), BASE8, fee_bps=16.5, maker_fee_bps=6.0, slip_fn=slip_model(scale=0.9))
go('DAILY book, fees ×2 + slip ×2', book4(), BASE24, fee_bps=11.0, maker_fee_bps=4.0, slip_fn=slip_model(scale=0.6))

# ---------------- T7 significance
print('\n=== T7 significance (daily returns) ===', flush=True)
trials = []
for f in glob.glob('out_exp20*.csv') + glob.glob('out_exp19*.csv'):
    try:
        df = pd.read_csv(f)
        if 'sharpe' in df: trials += list(df.sharpe.dropna().values)
    except Exception: pass
trials = np.array(trials); print(f'  trials recorded in exp19/20 csv files: N={len(trials)}, mean {trials.mean():.2f}, std {trials.std(ddof=1):.2f}')
for k in ('book5', 'book4'):
    v = S[k].dropna().values; n = len(v); Tyrs = n / 365.0
    sd = v.mean() / v.std(ddof=1); S_ann = sd * math.sqrt(365)
    se_ann = math.sqrt((1 + 0.5 * S_ann ** 2) / Tyrs)
    g3, g4 = skew(v), kurtosis(v, fisher=False)
    psr0 = norm.cdf(sd * math.sqrt(n - 1) / math.sqrt(1 - g3 * sd + (g4 - 1) / 4 * sd ** 2))
    line = f'  {k}: Sharpe {S_ann:.2f}, SE {se_ann:.2f}, 95% CI [{S_ann-1.96*se_ann:.2f}, {S_ann+1.96*se_ann:.2f}], t {S_ann/se_ann:.1f} | skew {g3:+.2f} kurt {g4:.1f} | PSR(>0) {psr0:.4f}'
    for Ntr in (len(trials), 50, 200):
        if Ntr < 2: continue
        var_tr = (trials.std(ddof=1) / math.sqrt(365)) ** 2 if len(trials) > 2 else (0.3 / math.sqrt(365)) ** 2
        em = 0.5772156649
        sr0 = math.sqrt(var_tr) * ((1 - em) * norm.ppf(1 - 1 / Ntr) + em * norm.ppf(1 - 1 / (Ntr * math.e)))
        dsr = norm.cdf((sd - sr0) * math.sqrt(n - 1) / math.sqrt(1 - g3 * sd + (g4 - 1) / 4 * sd ** 2))
        line += f' | DSR(N={Ntr}, SR0 ann {sr0*math.sqrt(365):.2f}) {dsr:.3f}'
    print(line, flush=True)

# ---------------- T8 beta cap
print('\n=== T8 book-beta cap (hedge only the excess beyond ±cap with BTC) ===', flush=True)
from betautil import rolling_beta
BETA = rolling_beta(d.ret, JB, 720)
def beta_capped(fn, cap):
    def g(dd, i, mask):
        w = fn(dd, i, mask); b = np.where(np.isfinite(BETA[i]), BETA[i], 1.0); bb = float((w * b).sum())
        if abs(bb) > cap:
            w = w.copy(); w[JB] -= (bb - math.copysign(cap, bb))      # hedge the excess only
        return w
    return g
for cap in (0.10, 0.20, 0.05, 0.0):
    go(f'R5 beta cap ±{cap:.2f}', beta_capped(book5(), cap), BASE8)
for cap in (0.10, 0.0):
    go(f'DAILY beta cap ±{cap:.2f}', beta_capped(book4(), cap), BASE24)
S.to_csv('out_exp21a_daily_returns.csv'); FAC.to_csv('out_exp21a_factors.csv')
print('DONE')
