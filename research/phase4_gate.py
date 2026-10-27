#!/usr/bin/env python3
"""Phase 4 gate (docs/PHASE4_PLAN.md): run books through the research engine on predictions extended to 2026-10-26,
slice DEV / HOLDOUT, execution lag 0/1 h, costs x1 / x1.5, taker-only; compare candidate vs base (paired block bootstrap, DSR).

  python3 research/phase4_gate.py run  <book> [--tag T]      # run one book (F2, F, D2) in all scenarios -> research/phase4_out/<tag>.csv (daily returns)
  python3 research/phase4_gate.py cmp  <base_tag> <cand_tag> --kind improve|simplify --step 4.3 --name "text"   # gate verdict + log line
  python3 research/phase4_gate.py diag <tag>                   # regression of daily PnL on the "top-50 alts - BTC" factor
Run from repo root. Env: END, PRED_SUF8, PRED_SUF37 are set here (override with env if needed)."""
import os, sys, math, glob, json, argparse, numpy as np, pandas as pd
os.environ.setdefault('END', '2026-10-26 23:00'); os.environ.setdefault('PRED_SUF8', '_oct'); os.environ.setdefault('PRED_SUF37', '_noage_oct')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); os.chdir(ROOT); sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, 'research', 'phase4_out'); LOG = os.path.join(ROOT, 'docs', 'PHASE4_LOG.md')
DEV = ('2022-01-01', '2026-02-28'); HOLD = ('2026-03-01', '2026-10-26')
SCEN = {  # name: (lag_h, cost key)
    'lag0_x1': (0, 'x1'), 'lag1_x1': (1, 'x1'), 'lag1_x1.5': (1, 'x1.5'), 'lag1_taker': (1, 'taker')}
STD = 'lag1_x1'

def sharpe(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return float(x.mean() / x.std(ddof=1) * math.sqrt(365)) if len(x) > 30 and x.std(ddof=1) > 0 else float('nan')
def stats(r):
    r = r.dropna(); eq = (1 + r).cumprod(); yrs = len(r) / 365.0
    return dict(sharpe=sharpe(r), cagr=float(eq.iloc[-1] ** (1 / yrs) - 1) if yrs > 0 else float('nan'), mdd=float((eq / eq.cummax() - 1).min()), n=len(r))
def win(r, w): return r[w[0]:w[1]]

def run_book(name):
    import exp_common as X
    from bt import run, slip_model
    from exp_common import d, F, F2, BASE8, BASE24, combo, Held, sl_listing, sl_core, ML3n, ML7n, fund_change, fund_level
    books = {'F2': lambda: F2(), 'F': lambda: F(),
             'D2': lambda: combo([sl_listing, sl_core, ML3n, ML7n, fund_change, fund_level], [1, 1, 1, 1, 1, 1])}
    COST = {'x1': {}, 'x1.5': dict(fee_bps=8.25, maker_fee_bps=3.0, slip_fn=slip_model(scale=0.45)), 'taker': dict(maker_share=0.0, slip_fn=slip_model(scale=1.0))}
    base = BASE24 if name == 'D2' else BASE8
    res = {}
    for sc, (lag, ck) in SCEN.items():
        fn = books[name]()
        if lag:
            f0 = fn
            def fn(dd, i, mask, f0=f0, lag=lag): return f0(dd, i - lag, mask)   # decide on bar i-lag information, execute at bar i
        args = dict(base); args.update(COST[ck]); args['reb_offset'] = base.get('reb_offset', 0) + lag; args['label'] = f'{name}/{sc}'
        d.start_i = int(d.idx.searchsorted(pd.Timestamp(X.OOS, tz='UTC')))
        m, eq, hp = run(d, fn, ret_diag=True, **args)
        de = eq.groupby(eq.index.floor('D')).last(); r = de.pct_change().dropna(); r.index = r.index.tz_localize(None) if r.index.tz is not None else r.index
        res[sc] = r; print(f'{name:<4} {sc:<11} Sh {m["sharpe"]:5.2f} CAGR {m["cagr"]*100:5.1f}% MDD {m["mdd"]*100:6.1f}% cost {(m["fees"]+m["slip"])*100:4.1f}%', flush=True)
    return pd.DataFrame(res)

def table(df):
    rows = []
    for sc in df.columns:
        for wn, w in (('DEV', DEV), ('HOLD', HOLD)):
            s = stats(win(df[sc], w)); rows.append(dict(scen=sc, win=wn, **s))
    return pd.DataFrame(rows)

def boot_p(a, b, block=20, nrep=2000, seed=11):
    """one-sided p that mean(b-a)<=0 ... returns p = P(boot ΔSharpe <= 0) over paired days (stationary-ish block bootstrap)"""
    x = (b - a).dropna(); ia = a.reindex(x.index).values; ib = b.reindex(x.index).values; n = len(ia); rng = np.random.default_rng(seed)
    nb = int(math.ceil(n / block)); ds = np.empty(nrep)
    for k in range(nrep):
        st = rng.integers(0, n - block + 1, nb); idx = (st[:, None] + np.arange(block)).ravel()[:n]
        ds[k] = sharpe(ib[idx]) - sharpe(ia[idx])
    return float((ds <= 0).mean()), float(np.percentile(ds, 5)), float(np.percentile(ds, 95))

def n_trials():
    n = 490
    for ln in open(LOG, encoding='utf-8'):
        c = [x.strip() for x in ln.strip().strip('|').split('|')]
        if len(c) >= 7 and c[0].startswith('20') and 'не попытка' not in c[1]: n += 1
    return n
def trials_std():
    t = []
    for f in glob.glob('out_exp19*.csv') + glob.glob('out_exp20*.csv') + glob.glob('out_exp22*.csv') + glob.glob('out_exp24*.csv'):
        try:
            df = pd.read_csv(f); t += list(df.sharpe.dropna().values) if 'sharpe' in df else []
        except Exception: pass
    return float(np.std(t, ddof=1)) if len(t) > 2 else 0.5
def dsr(v, Ntr, sd_tr):
    from scipy.stats import norm, skew, kurtosis
    v = np.asarray(pd.Series(v).dropna()); n = len(v); s = v.mean() / v.std(ddof=1); g3, g4 = skew(v), kurtosis(v, fisher=False); em = 0.5772156649
    sr0 = (sd_tr / math.sqrt(365)) * ((1 - em) * norm.ppf(1 - 1 / Ntr) + em * norm.ppf(1 - 1 / (Ntr * math.e)))
    return float(norm.cdf((s - sr0) * math.sqrt(n - 1) / math.sqrt(1 - g3 * s + (g4 - 1) / 4 * s ** 2))), sr0 * math.sqrt(365)

def compare(base, cand, kind, step, name, write=True):
    S = lambda df, sc, w: stats(win(df[sc], w))['sharpe']
    dev_b, dev_c = S(base, STD, DEV), S(cand, STD, DEV); hol_b, hol_c = S(base, STD, HOLD), S(cand, STD, HOLD)
    p, lo, hi = boot_p(win(base[STD], DEV), win(cand[STD], DEV))
    st = {sc: S(cand, sc, DEV) - S(base, sc, DEV) for sc in ('lag1_x1.5', 'lag1_taker')}
    sth = {sc: S(cand, sc, HOLD) - S(base, sc, HOLD) for sc in ('lag1_x1.5', 'lag1_taker')}
    Ntr = n_trials() + 1; D, sr0 = dsr(win(cand[STD], DEV), Ntr, trials_std())
    if kind == 'improve':
        c1 = dev_c > dev_b and p < 0.10; c2 = hol_c >= hol_b - 0.10; c3 = min(st.values()) >= -0.10 and min(sth.values()) >= -0.10; c4 = D >= 0.95
        ok = c1 and c2 and c3 and c4; det = f'DEV Δ{dev_c-dev_b:+.2f} p={p:.3f}; HOLD Δ{hol_c-hol_b:+.2f}; стресс Δ(x1.5 {st["lag1_x1.5"]:+.2f}, taker {st["lag1_taker"]:+.2f}); DSR {D:.3f} (N={Ntr}, SR0 {sr0:.2f})'
    else:
        ok = dev_c - dev_b >= -0.10 and hol_c - hol_b >= -0.10; det = f'DEV Δ{dev_c-dev_b:+.2f} (p={p:.3f}); HOLD Δ{hol_c-hol_b:+.2f}; стресс Δ(x1.5 {st["lag1_x1.5"]:+.2f}, taker {st["lag1_taker"]:+.2f})'
    verdict = 'ПРИНЯТ' if ok else 'ОТКЛОНЁН'
    line = f'| {pd.Timestamp.today().date()} | {step} | {name} ({kind}) | DEV/HOLD, lag 1 ч | {dev_b:.2f} / {hol_b:.2f} | {dev_c:.2f} / {hol_c:.2f} | {verdict}: {det} |'
    print(line)
    if write:
        s = open(LOG, encoding='utf-8').read(); mark = '\n## '
        i = s.find(mark); s = s[:i].rstrip('\n') + '\n' + line + '\n' + s[i:]; open(LOG, 'w', encoding='utf-8').write(s)
    return ok

def alt_factor():
    """daily factor 'equal-weight top-50 alts (by 24h turnover at day start) minus BTC', UTC days (sum of hourly log returns)"""
    from exp_common import d, JB
    day = d.idx.floor('D').tz_localize(None); R = np.where(np.isfinite(d.ret), d.ret, 0.0); out = {}
    for u in pd.unique(day):
        ii = np.flatnonzero(day == u)
        if len(ii) < 24 or ii[0] < 24 * 35: continue
        t = d.t24[ii[0] - 1].astype(float).copy(); t[JB] = np.nan; ok = np.isfinite(t) & d.valid[ii[0] - 1]
        top = np.flatnonzero(ok)[np.argsort(-t[ok])[:50]]
        out[u] = float(R[ii][:, top].mean(1).sum() - R[ii][:, JB].sum())
    return pd.Series(out)
def diag(tag):
    r = pd.read_csv(os.path.join(OUT, tag + '.csv'), index_col=0, parse_dates=True)[STD]; f = alt_factor().reindex(r.index)
    for lab, w in (('2022-01..2026-08 (до плохого периода)', ('2022-01-01', '2026-08-31')), ('2026-09-11..2026-10-26 (плохой период)', ('2026-09-11', '2026-10-26')), ('2026-03..2026-10', HOLD)):
        y, x = win(r, w), win(f, w); ok = y.notna() & x.notna(); y, x = y[ok], x[ok]
        b = float(np.cov(x, y)[0, 1] / x.var(ddof=1)); R2 = float(np.corrcoef(x, y)[0, 1] ** 2)
        print(f'{lab}: дней {len(y)}, beta к фактору {b:+.3f}, R2 {R2:.3f}, corr {np.corrcoef(x,y)[0,1]:+.3f}; сумма PnL книги {y.sum()*100:+.1f}%, объяснено фактором {b*x.sum()*100:+.1f}% (фактор за период {x.sum()*100:+.1f}%)')

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('cmd'); ap.add_argument('a', nargs='?'); ap.add_argument('b', nargs='?')
    ap.add_argument('--tag'); ap.add_argument('--kind', default='improve'); ap.add_argument('--step', default='?'); ap.add_argument('--name', default='?'); ap.add_argument('--nolog', action='store_true')
    a = ap.parse_args(); os.makedirs(OUT, exist_ok=True)
    if a.cmd == 'run':
        df = run_book(a.a); tag = a.tag or a.a; df.to_csv(os.path.join(OUT, tag + '.csv')); print(table(df).round(3).to_string(index=False))
    elif a.cmd == 'cmp':
        ld = lambda t: pd.read_csv(os.path.join(OUT, t + '.csv'), index_col=0, parse_dates=True)
        compare(ld(a.a), ld(a.b), a.kind, a.step, a.name, write=not a.nolog)
    elif a.cmd == 'diag':
        diag(a.a or 'F2')
if __name__ == '__main__': main()
