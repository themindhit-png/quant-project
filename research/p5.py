#!/usr/bin/env python3
"""Reg. 5 research runner and gate (docs/REG5_PLAN.md — written before the runs). Research engine, R2 lag-1 configuration (signal on bar
i-1, execution at bar i, BASE8), V2 on the combined book, exactly as research/rev_book.py / exp_r2.py.

  python3 research/p5.py run  <variant> [--tag T] [--rev] [--rec]   # variant = base | flag+flag+... (see FLAGS); --rev = reverse 2020-06..2021-12
  python3 research/p5.py cmp  <base_tag> <cand_tag> --cls S|M|A --step S --name "text" [--n N] [--nolog]
  python3 research/p5.py prop <tag>                                   # prop proxy only
Env (forward): FUND_SRC (exp_common), P5_PRED8 / P5_PRED37 (prediction suffixes, default reg. 4). Outputs research/phase4_out/P5_<tag>.csv
(daily returns per scenario; reverse -> P5_<tag>_rev.csv) and P5_<tag>.json (run metrics).
FLAGS: nofchg (drop funding-change sleeve) | fchgres (its signal = residual after 24h return and premium change) | ml37 (h3+h7 merged, weight 2)
       | softq (soft 15-25 % quantile in funding sleeves) | ml8w1 (8h sleeve weight 1) | crp (cluster risk parity) | fcm (fcarry x clip(S/med S,.5,2))
       | fcoff (fcarry off if expected carry < rebalance cost) | vt30 / vtewm (ex-ante vol target) | rk1 / rk2 (rank weights in ML sleeves) | ctrend (core = ensemble of lookbacks)"""
import os, sys, math, json, argparse, numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); os.chdir(ROOT); sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, 'research', 'phase4_out'); LOG = os.path.join(ROOT, 'docs', 'PHASE4_LOG.md')
V3 = '/home/user/edge-lab/logs/final_v3_l5tt2_daily.parquet'   # second bot's daily series (similarity gate); absent -> gate skipped
N0 = 573

def setup_env(rev):
    if rev:
        os.environ.update(START='2020-01-01', OOS='2020-06-01', EVAL_END='2022-01-01', PRED_SUF8=os.environ.get('P5_REV8', '_rev'), PRED_SUF37=os.environ.get('P5_REV37', '_rev'), END='2026-10-26 23:00')
    else:
        os.environ.update(END='2026-10-26 23:00', PRED_SUF8=os.environ.get('P5_PRED8', '_oct_noconst_r4'), PRED_SUF37=os.environ.get('P5_PRED37', '_noage_oct_r4'))

# ------------------------------------------------------------------------------------------------ books
def build(flags):
    import exp_common as X
    from exp_common import d, N, VOL, Held, sl_listing, sl_core, ML8, fund_change, fund_level, uni, fsum, FUND0, Q3, Q7, P8, qf
    from bt import quantile_ls, rank_weights, last_rate
    from scipy.stats import rankdata

    def softq(sig, m, i):            # full weight in top/bottom 15 %, linear to 0 at 25 %; ties share (average rank)
        s = np.where(m & np.isfinite(sig), np.round(sig, 9), np.nan); ok = np.flatnonzero(np.isfinite(s)); w = np.zeros(N)
        if len(ok) < 10: return w
        u = (rankdata(s[ok]) - 0.5) / len(ok)
        lw = np.clip((u - 0.75) / 0.10, 0, 1); sw = np.clip((0.25 - u) / 0.10, 0, 1)
        if lw.sum() > 0: w[ok] += lw / lw.sum()
        if sw.sum() > 0: w[ok] -= sw / sw.sum()
        return w
    sel = (lambda sig, m, i: softq(sig, m, i)) if 'softq' in flags else (lambda sig, m, i: qf(sig, m, i))

    PREM = None
    if 'fchgres' in flags:
        PREM = pd.read_parquet('data/panels/premium.parquet', columns=list(d.cols)).reindex(d.idx).values.astype(np.float32)
    def fchg(dd, i, mask):
        m = uni(dd, i); x = -(fsum(i, 24, 0) - fsum(i, 48, 24))
        if 'fchgres' in flags:
            r24 = np.log(dd.cff[i] / dd.cff[i - 24]); dp = np.nanmean(PREM[i - 23:i + 1], 0) - np.nanmean(PREM[i - 47:i - 23], 0)
            ok = m & np.isfinite(x) & np.isfinite(r24) & np.isfinite(dp); y = np.full(N, np.nan)
            if ok.sum() >= 20:
                A = np.column_stack([np.ones(ok.sum()), r24[ok], dp[ok]]); b = np.linalg.lstsq(A, x[ok], rcond=None)[0]; y[ok] = x[ok] - A @ b
            x = y
        return sel(x, m, i)
    def fcar(dd, i, mask): return sel(-fsum(i, 72, 0), uni(dd, i), i)

    def mlw(P, top, min_age_h, iv):
        def f(dd, i, mask):
            m = dd.universe(i, min_age_h=min_age_h, min_turn=1e7, top_n=150); sig = np.where(m & np.isfinite(P[i]), P[i], np.nan)
            if 'rk1' in flags or 'rk2' in flags:
                w = rank_weights(sig, None, 1.0 if 'rk1' in flags else 2.0)
                if iv:
                    w = w / np.where(np.isfinite(VOL[i]) & (VOL[i] > 0), VOL[i], np.inf); L, S = w[w > 0].sum(), -w[w < 0].sum()
                    w = np.where(w > 0, w / max(L, 1e-12), w / max(S, 1e-12))
                return w
            return quantile_ls(sig, top, inv_vol=VOL[i] if iv else None)
        return f
    ml3, ml7, ml8 = mlw(Q3, 0.3, 90 * 24, False), mlw(Q7, 0.3, 90 * 24, False), mlw(P8, 0.2, 180 * 24, True)
    def ml37(dd, i, mask):
        m = dd.universe(i, min_age_h=90 * 24, min_turn=1e7, top_n=150) & np.isfinite(Q3[i]) & np.isfinite(Q7[i]); idx = np.flatnonzero(m); s = np.full(N, np.nan)
        if len(idx) >= 10: s[idx] = (rankdata(Q3[i][idx]) + rankdata(Q7[i][idx])) / 2
        return quantile_ls(s, 0.3)

    # fcarry multiplier / switch: S = expected carry spread of the sleeve's legs (72h funding sums), known at the decision bar
    if 'fcm' in flags or 'fcoff' in flags:
        Sv = {}; Wp = {}; prev = None
        for i in range(24 * 35, d.ret.shape[0]):
            if d.hh[i] % 8: continue
            d.prev_mask = None; w = fcar(d, i, None); Sv[i] = float(-(w * fsum(i, 72, 0)).sum()); Wp[i] = float(np.abs(w - prev).sum()) if prev is not None else 1.0; prev = w
        ks = np.array(sorted(Sv)); sv = np.array([Sv[k] for k in ks]); med = pd.Series(sv).expanding().median().values; MED = dict(zip(ks, med))
        d.prev_mask = None
    def fcar2(dd, i, mask):
        w = fcar(dd, i, mask)
        if i not in Sv: return w
        if 'fcm' in flags:
            mm = MED[i]; w = w * (float(np.clip(Sv[i] / mm, 0.5, 2.0)) if mm > 0 else 1.0)
        if 'fcoff' in flags and Sv[i] * 8 / 72 < 4.5e-4 * Wp[i]: w = w * 0.0
        return w

    def core_ens(dd, i, mask):   # idea 11: equal-weight ensemble of lookbacks (ranks), same universe / quantile as the core sleeve
        from strategies import core_signal
        m = dd.universe(i, min_age_h=720, min_turn=1e7, top_n=100); base = core_signal(dd, i, m, 336); ok = np.flatnonzero(np.isfinite(base)); s = np.full(N, np.nan)
        if len(ok) >= 10:
            rk = [rankdata(np.where(np.isfinite(x[ok]), x[ok], np.nanmedian(x[ok]))) / len(ok) for x in (core_signal(dd, i, m, lb) for lb in (24, 72, 168, 336, 720))]
            s[ok] = np.mean(rk, 0)
        return quantile_ls(s, 0.2)
    parts = [Held(sl_listing), Held(core_ens if 'ctrend' in flags else sl_core)]; wts = [1, 1]; names = ['listing', 'core']
    if 'ml37' in flags: parts += [Held(ml37)]; wts += [2]; names += ['ml37']
    else: parts += [Held(ml3), Held(ml7)]; wts += [1, 1]; names += ['ml3', 'ml7']
    parts += [ml8]; wts += [1 if 'ml8w1' in flags else 2]; names += ['ml8']
    if 'nofchg' not in flags: parts += [Held(fchg)]; wts += [1]; names += ['fchg']
    parts += [fcar2 if ('fcm' in flags or 'fcoff' in flags) else fcar]; wts += [1]; names += ['fcarry']

    if 'crp' in flags:   # cluster risk parity on causal paper returns of the sleeves
        CL = {'listing': 'L', 'core': 'C', 'ml3': 'M', 'ml7': 'M', 'ml37': 'M', 'ml8': 'M', 'fchg': 'F', 'fcarry': 'F'}
        raw = {'listing': sl_listing, 'core': sl_core, 'ml3': ml3, 'ml7': ml7, 'ml37': ml37, 'ml8': ml8, 'fchg': fchg, 'fcarry': fcar}
        daily8 = {'ml8', 'fcarry'}; T = d.ret.shape[0]; R0 = np.nan_to_num(d.ret); H = np.zeros((T, len(names))); cur = {n: np.zeros(N) for n in names}
        for i in range(24 * 35, T - 1):
            if d.hh[i] % 8 == 0:
                for n in names:
                    if n in daily8 or d.hh[i] % 24 == 0: d.prev_mask = None; cur[n] = raw[n](d, i, None)
            H[i + 1] = [cur[n] @ R0[i + 1] for n in names]
        d.prev_mask = None
        day = d.idx.floor('D'); DR = pd.DataFrame(H, index=day).groupby(level=0).sum(); qkey = DR.index.to_period('Q')
        WQ = {}
        for qq in pd.unique(qkey):
            hist = DR[DR.index < qq.start_time.tz_localize('UTC')]
            hist = hist[(hist != 0).all(1)]
            if len(hist) < 90: WQ[str(qq)] = np.ones(len(names)); continue
            C = hist.cov().values; C = 0.5 * C + 0.5 * np.diag(np.diag(C)); cl = [CL[n] for n in names]; u = sorted(set(cl)); wv = np.zeros(len(names))
            for c in u:
                a = np.array([1.0 / cl.count(c) if x == c else 0.0 for x in cl]); s = math.sqrt(a @ C @ a); wv += a / s
            WQ[str(qq)] = wv / wv.sum() * len(names)
        print('crp quarterly weights (' + ','.join(names) + '):', {k: np.round(v, 2).tolist() for k, v in list(WQ.items())[::4]}, flush=True)
        def book(dd, i, mask):
            wv = WQ[str(d.idx[i].to_period('Q'))]; w = np.zeros(N)
            for f, s in zip(parts, wv): w += s * f(dd, i, mask)
            return w / wv.sum()
    else:
        book = X.combo(parts, wts)
    print('sleeves:', dict(zip(names, wts)), flush=True)

    def v2(fn, mult=0.5):   # verbatim research/exp_r2.py
        def g(dd, i, mask):
            w = fn(dd, i, mask).copy(); nz = np.flatnonzero((w != 0) & np.isfinite(VOL[i]))
            if len(nz) < 10: return w
            L, S = w[w > 0].sum(), -w[w < 0].sum(); w[nz[np.argsort(VOL[i][nz])[int(len(nz) * 2 / 3):]]] *= mult
            L2, S2 = w[w > 0].sum(), -w[w < 0].sum()
            if L2 > 0: w[w > 0] *= L / L2
            if S2 > 0: w[w < 0] *= S / S2
            return w
        return g
    return v2(book)

def run_variant(var, rev, rec):
    flags = set() if var == 'base' else set(var.split('+'))
    import exp_common as X
    from exp_common import d, BASE8
    from bt import run, slip_model
    COST = {'x1': {}, 'x1.5': dict(fee_bps=8.25, maker_fee_bps=3.0, slip_fn=slip_model(scale=0.45)), 'taker': dict(maker_share=0.0, slip_fn=slip_model(scale=1.0))}
    scen = ['x1'] if rev else ['x1', 'x1.5', 'taker']
    EEND = os.environ.get('EVAL_END'); end_i = int(d.idx.searchsorted(pd.Timestamp(EEND, tz='UTC'))) if EEND else None
    res, meta = {}, {}
    for sc in scen:
        f0 = build(flags)
        def fn(dd, i, mask, f0=f0): return f0(dd, i - 1, mask)
        args = dict(BASE8); args.update(COST[sc]); args['reb_offset'] = 1; args['label'] = f'{var}/{sc}'
        if 'vt30' in flags: args['vt_src'] = 'exante30'
        if 'vtewm' in flags: args['vt_src'] = 'exante_ewm10'
        R = [] if (rec and sc == 'x1') else None
        d.start_i = int(d.idx.searchsorted(pd.Timestamp(X.OOS, tz='UTC')))
        m, eq, hp = run(d, fn, ret_diag=True, end_i=end_i, rec=R, **args)
        de = eq.groupby(eq.index.floor('D')).last(); r = de.pct_change().dropna(); r.index = r.index.tz_localize(None)
        res[f'lag1_{sc}'] = r
        meta[sc] = {k: float(m[k]) for k in ('sharpe', 'cagr', 'mdd', 'turnover_x', 'fees', 'slip', 'funding', 'avg_gross') if k in m}
        print(f'{var} {"REV" if rev else "FWD"} {sc}: Sh {m["sharpe"]:.2f} CAGR {m["cagr"]*100:.1f}% MDD {m["mdd"]*100:.1f}% turn {m["turnover_x"]:.0f}x '
              f'cost {(m["fees"]+m["slip"])*100:.1f}% fund {m["funding"]*100:+.1f}% gross {m["avg_gross"]:.2f}', flush=True)
        if R is not None:
            rows = [(d.idx[i], d.cols[j], float(v[j])) for i, v in R for j in np.flatnonzero(v)]
            os.makedirs('logs', exist_ok=True); tg = os.environ.get('REC_TAG', var)
            pd.DataFrame(rows, columns=['time', 'symbol', 'w']).to_parquet(f'logs/f2_weights_{tg}.parquet', index=False)
            hr = eq.pct_change().dropna(); hr.index = hr.index.tz_localize(None); hr.rename('ret').to_frame().to_parquet(f'logs/f2_hourly_{tg}.parquet')
            print(f'weights: {len(rows)} rows ({len(R)} rebalances) -> logs/f2_weights_{tg}.parquet; hourly book returns -> logs/f2_hourly_{tg}.parquet', flush=True)
    return pd.DataFrame(res), meta

# ------------------------------------------------------------------------------------------------ gates
def sharpe(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return float(x.mean() / x.std(ddof=1) * math.sqrt(365)) if len(x) > 30 and x.std(ddof=1) > 0 else float('nan')
def mdd(r): e = (1 + r.fillna(0)).cumprod(); return float((e / e.cummax() - 1).min())
def sboot(a, b, nrep=1000, mean_block=20, seed=7):
    """paired stationary bootstrap (geometric blocks): distribution of Sharpe(b) - Sharpe(a)"""
    j = a.index.intersection(b.index); A, B = a[j].values, b[j].values; n = len(A); rng = np.random.default_rng(seed); out = np.empty(nrep); p = 1.0 / mean_block
    for k in range(nrep):
        idx = np.empty(n, dtype=np.int64); idx[0] = rng.integers(n); jump = rng.random(n) < p; st = rng.integers(0, n, n)
        for t in range(1, n): idx[t] = st[t] if jump[t] else (idx[t - 1] + 1) % n
        out[k] = sharpe(B[idx]) - sharpe(A[idx])
    return out
def prop_proxy(r, starts=None):
    """123 starts every 14 days; phase targets 8 % / 5 %, min days 10 / 3, static total DD 10 %, daily loss 5 % of the phase start balance"""
    starts = starts or pd.date_range('2022-01-03', '2026-09-07', freq='14D'); r = r.dropna(); out = {'pass': 0, 'fail': 0, 'open': 0}; days = []
    for s in starts:
        x = r[s:]; k = 0; st = 'open'; n = 0
        for tgt, mind in ((0.08, 10), (0.05, 3)):
            e = 1.0; nd = 0; st = 'open'
            while k < len(x):
                v = x.iloc[k]; k += 1; nd += 1; n += 1
                if v * e <= -0.05: st = 'fail'; break
                e *= 1 + v
                if e <= 0.90: st = 'fail'; break
                if e >= 1 + tgt and nd >= mind: st = 'pass'; break
            if st != 'pass': break
        out[st] += 1
        if st == 'pass': days.append(n)
    out['median_days'] = float(np.median(days)) if days else float('nan'); return out
def v3_corr(r):
    if not os.path.exists(V3): return float('nan')
    v = pd.read_parquet(V3)['fin']; v.index = pd.to_datetime(v.index).tz_localize(None); j = r.dropna().index.intersection(v.index)
    return float(np.corrcoef(r[j], v[j])[0, 1])
def load(tag, rev=False): return pd.read_csv(os.path.join(OUT, f'P5_{tag}{"_rev" if rev else ""}.csv'), index_col=0, parse_dates=True)
def halves(r):
    return {f'{y}H{h}': r[f'{y}-{"01" if h == 1 else "07"}-01':f'{y}-{"06-30" if h == 1 else "12-31"}'] for y in range(2022, 2027) for h in (1, 2)}

TH = {'S': dict(d1=-0.03, d15=-0.03, yr=-0.05, yrmean=None, hb=-0.05, tk=-0.05, rev=-0.2), 'M': dict(d1=0.05, d15=0.03, yr=0.0, yrmean=0.0, hb=0.0, tk=0.0, rev=-0.2),
      'A': dict(d1=0.10, d15=0.05, yr=0.0, yrmean=0.0, hb=0.0, tk=0.0, rev=-0.2)}
def compare(bt, ct, cls, step, name, nacc, write=True, rev=True):
    B, C = load(bt), load(ct); t = TH[cls]; g = {}
    s = lambda df, c: sharpe(df[c])
    d1 = s(C, 'lag1_x1') - s(B, 'lag1_x1'); d15 = s(C, 'lag1_x1.5') - s(B, 'lag1_x1.5'); g['dSh x1'] = (d1 >= t['d1'], f'{d1:+.3f}'); g['dSh x1.5'] = (d15 >= t['d15'], f'{d15:+.3f}')
    yrs = {y: sharpe(C['lag1_x1'][str(y)]) - sharpe(B['lag1_x1'][str(y)]) for y in range(2022, 2027)}; ny = sum(v >= t['yr'] for v in yrs.values())
    g['годы'] = (ny >= 4 and (t['yrmean'] is None or np.mean(list(yrs.values())) > t['yrmean']), f'{ny}/5 ' + ' '.join(f'{v:+.2f}' for v in yrs.values()))
    hb, hc = halves(B['lag1_x1']), halves(C['lag1_x1']); hd = {k: sharpe(hc[k]) - sharpe(hb[k]) for k in hb}; nh = sum(v >= t['hb'] for v in hd.values())
    g['полугодия'] = (nh >= 8, f'{nh}/10')
    dt = s(C, 'lag1_taker') - s(B, 'lag1_taker'); g['тейкер'] = (dt >= t['tk'], f'{dt:+.3f}')
    if rev and os.path.exists(os.path.join(OUT, f'P5_{ct}_rev.csv')) and os.path.exists(os.path.join(OUT, f'P5_{bt}_rev.csv')):
        dr = sharpe(load(ct, True)['lag1_x1']) - sharpe(load(bt, True)['lag1_x1']); g['реверс'] = (dr >= t['rev'], f'{dr:+.3f}')
    else: g['реверс'] = (None, 'нет')
    bs = sboot(B['lag1_x1'], C['lag1_x1'])
    if cls == 'S': pb = float((bs < -0.05).mean()); g['бутстрап'] = (pb <= 0.5, f'P(Δ<-0.05)={pb:.2f}')
    else: pb = float((bs > 0).mean()); g['бутстрап'] = (pb >= (0.75 if cls == 'M' else 0.80), f'P(Δ>0)={pb:.2f}')
    dm = (mdd(C['lag1_x1']) - mdd(B['lag1_x1'])) * 100; dw = (C['lag1_x1'].min() - B['lag1_x1'].min()) * 100
    g['MDD/худший день'] = (dm >= -1.0 and dw >= -0.5, f'MDD {dm:+.2f} п.п., день {dw:+.2f} п.п.')
    pB, pC = prop_proxy(B['lag1_x1']), prop_proxy(C['lag1_x1']); g['проп-прокси'] = (pC['fail'] <= pB['fail'], f'провалы {pB["fail"]}->{pC["fail"]}, проход {pB["pass"]}->{pC["pass"]}, медиана дн {pB["median_days"]:.0f}->{pC["median_days"]:.0f}')
    cb, cc = v3_corr(B['lag1_x1']), v3_corr(C['lag1_x1']); g['похожесть'] = ((cc - cb) <= 0.05 if np.isfinite(cb) else None, f'корр {cb:.3f}->{cc:.3f}')
    g = {k: ((None if p is None else bool(p)), txt) for k, (p, txt) in g.items()}   # numpy bools -> bool (bug 2026-11-01: `is False` missed np.False_)
    ok = all(v[0] is not False for v in g.values())
    for k, (p, txt) in g.items(): print(f'  {k:<16} {"OK " if p else ("-- " if p is None else "НЕТ")} {txt}')
    verdict = 'ПРОШЁЛ' if ok else 'НЕ ПРОШЁЛ'
    print(f'{name}: {verdict}')
    fails = ', '.join(k for k, v in g.items() if v[0] is False) or '—'
    line = (f'| {pd.Timestamp.today().date()} | {step} | {name} | {cls} | {s(B,"lag1_x1"):.2f} / {s(B,"lag1_x1.5"):.2f} | {s(C,"lag1_x1"):.2f} / {s(C,"lag1_x1.5"):.2f} | '
            + '; '.join(f'{k} {v[1]}' for k, v in g.items()) + f' | {verdict} (не прошли: {fails}) | {nacc} |')
    if write:
        txt = open(LOG, encoding='utf-8').read(); hdr = '## Рег. 5'
        if hdr not in txt:
            txt = txt.rstrip('\n') + f'\n\n{hdr} — попытки по docs/REG5_PLAN.md (N накопл. стартует с {N0})\n\n| дата | шаг | вариант | класс | база ×1 / ×1.5 | вариант ×1 / ×1.5 | ворота | итог | N накопл. |\n|---|---|---|---|---|---|---|---|---|\n'
        open(LOG, 'w', encoding='utf-8').write(txt.rstrip('\n') + '\n' + line + '\n')
    return ok, g

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('cmd'); ap.add_argument('a', nargs='?'); ap.add_argument('b', nargs='?')
    ap.add_argument('--tag'); ap.add_argument('--rev', action='store_true'); ap.add_argument('--rec', action='store_true')
    ap.add_argument('--cls', default='A'); ap.add_argument('--step', default='?'); ap.add_argument('--name', default='?'); ap.add_argument('--n', default='?'); ap.add_argument('--nolog', action='store_true')
    a = ap.parse_args()
    if a.cmd == 'run':
        setup_env(a.rev); df, meta = run_variant(a.a, a.rev, a.rec); tag = a.tag or a.a
        df.to_csv(os.path.join(OUT, f'P5_{tag}{"_rev" if a.rev else ""}.csv')); json.dump(meta, open(os.path.join(OUT, f'P5_{tag}{"_rev" if a.rev else ""}.json'), 'w'), indent=1)
        print('DONE', tag)
    elif a.cmd == 'cmp':
        compare(a.a, a.b, a.cls, a.step, a.name, a.n, write=not a.nolog)
    elif a.cmd == 'prop':
        print(prop_proxy(load(a.a)['lag1_x1']))
if __name__ == '__main__': main()
