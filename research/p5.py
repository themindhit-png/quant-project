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
       | fcoff (fcarry off if expected carry < rebalance cost) | vt30 / vtewm (ex-ante vol target) | rk1 / rk2 (rank weights in ML sleeves) | ctrend (core = ensemble of lookbacks)
       Reg. 5b (docs/REG5_PLAN.md, extra sleeves at weight 1): prem24 / prem72 (basis: long low / short high mean premium) | liq24 / liq72
       (continuation in the most liquid 20 %, reversal in the least liquid 50 %) | clus21 / clus7 (momentum of causal residual-correlation
       clusters) | brk (sleeve brake on the full paper return: price + funding - 3.7 bp x own turnover, k = 10). With any of these the book goes
       through a paper tracker (per-sleeve daily paper price / funding / cost, stored in P5_<tag>.json 'diag'); weights unchanged unless brk.
Env: P5_V3 (similarity series; default as before), P5_BAL=1 (cmp adds the balance gate: dSh 2023 > 0, dSh 2026 > 0, no year < -0.3)."""
import os, sys, math, json, argparse, numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); os.chdir(ROOT); sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, 'research', 'phase4_out'); LOG = os.path.join(ROOT, 'docs', 'PHASE4_LOG.md')
V3 = os.environ.get('P5_V3', '/home/user/edge-lab/logs/final_v3_l5tt2_daily.parquet')   # second bot's daily series (similarity gate); absent -> gate skipped
N0 = 573
DIAG = {}   # reg. 5b paper tracker output of the last build (x1 scenario is stored)
NEWF = {'prem24', 'prem72', 'liq24', 'liq72', 'clus21', 'clus7', 'brk', 'trk'}   # trk = tracker only (code check: must equal base)

def setup_env(rev):
    if rev:
        os.environ.update(START='2020-01-01', OOS='2020-06-01', EVAL_END='2022-01-01', PRED_SUF8=os.environ.get('P5_REV8', '_rev'), PRED_SUF37=os.environ.get('P5_REV37', '_rev'), END='2026-10-26 23:00')
    else:
        os.environ.update(END='2026-10-26 23:00', PRED_SUF8=os.environ.get('P5_PRED8', '_oct_noconst_r4'), PRED_SUF37=os.environ.get('P5_PRED37', '_noage_oct_r4'))

# ------------------------------------------------------------------------------------------------ books
def combo_track(parts, weights, names, EXTRA, brake_k=0.0, cost=3.7e-4):
    """combo with a causal per-sleeve paper tracker: between consecutive calls each sleeve's previous target weights earn the coin price
    returns and pay funding (FUND_REAL, long pays when rate > 0); each call charges cost x |dw|_1 of the sleeve's own weights. Booked per
    UTC day. brake_k > 0: sleeve mix weight x0.5 while its FULL paper equity (price + funding - cost) is below its running max by more than
    brake_k x expanding daily std (>= 60 d), state of the previous completed day, no renormalisation. Same arithmetic as exp_common.combo."""
    import exp_common as X
    d, N = X.d, X.N; R = d.ret; FR = np.where(np.isfinite(X.FUND_REAL), X.FUND_REAL, 0.0); K = len(parts)
    st = {'i': None, 'w': None, 'x': {}, 'day': None}; acc = {}; hist = [[] for _ in range(K)]; eq = np.zeros(K); peak = np.zeros(K); on = np.zeros(K, bool)
    rows = []; DIAG.clear(); DIAG.update(rows=rows, names=list(names))
    def roll(day):
        for k, n in enumerate(names):
            v = acc.get(n + '|p', 0.0) + acc.get(n + '|f', 0.0) + acc.get(n + '|c', 0.0); hist[k].append(v); eq[k] += v; peak[k] = max(peak[k], eq[k])
            sd = float(np.std(hist[k], ddof=1)) if len(hist[k]) >= 60 else np.nan
            on[k] = bool(brake_k > 0 and np.isfinite(sd) and sd > 0 and eq[k] - peak[k] < -brake_k * sd)
        rec = dict(acc); rec.update({'on|' + n: float(on[k]) for k, n in enumerate(names)}); rows.append((day, rec)); acc.clear()
    def fn(dd, i, mask):
        outs = [f(dd, i, mask) for f in parts]
        if st['i'] is not None and i > st['i']:
            rr = np.nan_to_num(R[st['i'] + 1:i + 1]).sum(0); ff = FR[st['i'] + 1:i + 1].sum(0)
            for k, n in enumerate(names):
                w0 = np.nan_to_num(st['w'][k]); acc[n + '|p'] = acc.get(n + '|p', 0.0) + float(w0 @ rr); acc[n + '|f'] = acc.get(n + '|f', 0.0) - float(w0 @ ff)
            for n, w0 in st['x'].items():
                acc['x:' + n + '|p'] = acc.get('x:' + n + '|p', 0.0) + float(w0 @ rr); acc['x:' + n + '|f'] = acc.get('x:' + n + '|f', 0.0) - float(w0 @ ff)
        day = d.idx[i].floor('D')
        if st['day'] is None: st['day'] = day
        elif day != st['day']: roll(st['day']); st['day'] = day
        for k, n in enumerate(names):
            pw = 0.0 if st['w'] is None else np.nan_to_num(st['w'][k]); acc[n + '|c'] = acc.get(n + '|c', 0.0) - cost * float(np.abs(np.nan_to_num(outs[k]) - pw).sum())
        st['i'] = i; st['w'] = [o.copy() for o in outs]; st['x'] = {k: v.copy() for k, v in EXTRA.items()}
        w = np.zeros(N)
        for o, s, k in zip(outs, weights, range(K)): w += s * (0.5 if on[k] else 1.0) * o
        return w / sum(weights)
    return fn

def diag_summary(rows, names):
    """per-year paper stats of the tracker (sleeves: price / funding / cost sums and full-return Sharpe; brake share; extra baskets)"""
    if not rows: return {}
    df = pd.DataFrame([r for _, r in rows], index=pd.to_datetime([t for t, _ in rows]).tz_localize(None)).fillna(0.0); df = df['2022-01-01':]
    out = {}
    for y, g in list(df.groupby(df.index.year)) + [('2022-26', df)]:
        o = {}
        for n in names:
            full = g.get(n + '|p', 0) + g.get(n + '|f', 0) + g.get(n + '|c', 0)
            o[n] = dict(p=round(float(g.get(n + '|p', pd.Series(0)).sum()), 4), f=round(float(g.get(n + '|f', pd.Series(0)).sum()), 4),
                        c=round(float(g.get(n + '|c', pd.Series(0)).sum()), 4), sh_full=round(sharpe(full), 2), brake=round(float(g.get('on|' + n, pd.Series(0)).mean()), 2))
        xs = sorted({c[2:].split('|')[0] for c in g.columns if c.startswith('x:')})
        for n in xs: o['x:' + n] = dict(p=round(float(g['x:' + n + '|p'].sum()), 4), f=round(float(g.get('x:' + n + '|f', pd.Series(0)).sum()), 4))
        if y == '2022-26':
            P = pd.DataFrame({n: g.get(n + '|p', 0) + g.get(n + '|f', 0) + g.get(n + '|c', 0) for n in names})
            o['corr_full'] = P.corr().round(2).to_dict()
        out[str(y)] = o
    return out

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
    # ---------------- reg. 5b: new sleeves (docs/REG5_PLAN.md "Рег. 5б"); EXTRA = diagnostic paper baskets (no effect on weights)
    EXTRA = {}
    if flags & {'prem24', 'prem72'}:
        PREMP = pd.read_parquet('data/panels/premium.parquet', columns=list(d.cols)).reindex(d.idx).values.astype(np.float32)
        LP = 24 if 'prem24' in flags else 72
        def sl_prem(dd, i, mask):     # basis factor, sign fixed a priori: long low premium, short high premium
            m = uni(dd, i)
            with np.errstate(all='ignore'): p = np.nanmean(PREMP[i - LP + 1:i + 1], 0)
            ok = np.flatnonzero(m & np.isfinite(p)); w = np.zeros(N)
            if len(ok) < 10: return w
            o = ok[np.argsort(p[ok], kind='stable')]; qn = np.floor(np.arange(len(o)) * 5 / len(o)).astype(int)
            for k in range(5):
                x = np.zeros(N); mem = o[qn == k]; x[mem] = 1.0 / len(mem); EXTRA[f'Q{k + 1}'] = x      # Q1 = lowest premium (long-only baskets)
            return quantile_ls(np.where(m & np.isfinite(p), -p, np.nan), 0.2)
    if flags & {'liq24', 'liq72'}:
        LR = 24 if 'liq24' in flags else 72
        def sl_liq(dd, i, mask):      # liquidity-conditioned short-term reversal / continuation
            m = uni(dd, i); w = np.zeros(N)
            with np.errstate(all='ignore'):
                lq = np.nanmean(dd.t24[i - 720:i + 1:24], 0); r = np.log(dd.cff[i] / dd.cff[i - LR])
            ok = np.flatnonzero(m & np.isfinite(lq) & np.isfinite(r))
            if len(ok) < 20: return w
            o = ok[np.argsort(-lq[ok], kind='stable')]; n = len(o); top = o[:int(round(n * 0.2))]; bot = o[n - int(round(n * 0.5)):]
            s1 = np.full(N, np.nan); s1[top] = r[top]; s2 = np.full(N, np.nan); s2[bot] = -r[bot]
            wm, wr = quantile_ls(s1, 0.3), quantile_ls(s2, 0.2); EXTRA['mom_liquid'] = wm; EXTRA['rev_illiquid'] = wr
            return 0.5 * wm + 0.5 * wr
    if flags & {'clus21', 'clus7'}:
        from scipy.cluster.hierarchy import linkage, fcluster
        from scipy.spatial.distance import squareform
        LC = 21 if 'clus21' in flags else 7; CLM = {}; CST = {'w': np.zeros(N)}
        def labels(dd, i):            # causal: first call of each calendar month, trailing 90 daily (00:00) returns
            key = (dd.idx[i].year, dd.idx[i].month)
            if key in CLM: return CLM[key]
            m = uni(dd, i); ii = i - 24 * np.arange(90, -1, -1); lab = np.full(N, -1)
            with np.errstate(all='ignore'):
                Rr = np.diff(np.log(dd.cff[ii]), axis=0)
            Rr[~(dd.valid[ii][1:] & dd.valid[ii][:-1])] = np.nan
            cand = np.flatnonzero(m & (np.isfinite(Rr).sum(0) >= 60))
            if len(cand) >= 20:
                Xc = Rr[:, cand]; mk = np.nanmean(Xc, 1); res = np.zeros_like(Xc)
                for c in range(Xc.shape[1]):
                    x = Xc[:, c]; o = np.isfinite(x) & np.isfinite(mk); xm, mm = x[o] - x[o].mean(), mk[o] - mk[o].mean()
                    b = float(xm @ mm / max(mm @ mm, 1e-18)); res[o, c] = xm - b * mm
                with np.errstate(all='ignore'): C = np.nan_to_num(np.corrcoef(res.T))
                D = np.sqrt(np.clip(2 * (1 - np.clip(C, -1, 1)), 0, None)); np.fill_diagonal(D, 0.0)
                lab[cand] = fcluster(linkage(squareform(D, checks=False), method='ward'), 10, criterion='maxclust')
            CLM[key] = lab; return lab
        def sl_clus(dd, i, mask):     # momentum of the cluster (leave-one-out), weekly
            lab = labels(dd, i)
            if dd.hh[i] % 168 == 0:
                m = uni(dd, i)
                with np.errstate(all='ignore'): r = np.log(dd.cff[i] / dd.cff[i - 24 * LC])
                ok = m & (lab >= 0) & np.isfinite(r); s = np.full(N, np.nan)
                for c in np.unique(lab[ok]):
                    mem = np.flatnonzero(ok & (lab == c))
                    if len(mem) >= 3: s[mem] = (r[mem].sum() - r[mem]) / (len(mem) - 1)
                CST['w'] = quantile_ls(s, 0.2)
            return CST['w']

    parts = [Held(sl_listing), Held(core_ens if 'ctrend' in flags else sl_core)]; wts = [1, 1]; names = ['listing', 'core']
    if 'ml37' in flags: parts += [Held(ml37)]; wts += [2]; names += ['ml37']
    else: parts += [Held(ml3), Held(ml7)]; wts += [1, 1]; names += ['ml3', 'ml7']
    parts += [ml8]; wts += [1 if 'ml8w1' in flags else 2]; names += ['ml8']
    if 'nofchg' not in flags: parts += [Held(fchg)]; wts += [1]; names += ['fchg']
    parts += [fcar2 if ('fcm' in flags or 'fcoff' in flags) else fcar]; wts += [1]; names += ['fcarry']
    if flags & {'prem24', 'prem72'}: parts += [Held(sl_prem)]; wts += [1]; names += ['prem']
    if flags & {'liq24', 'liq72'}: parts += [Held(sl_liq)]; wts += [1]; names += ['liq']
    if flags & {'clus21', 'clus7'}: parts += [sl_clus]; wts += [1]; names += ['clus']

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
    elif flags & NEWF:
        book = combo_track(parts, wts, names, EXTRA, 10.0 if 'brk' in flags else 0.0)
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
        if sc == 'x1' and flags & NEWF and DIAG.get('rows'): meta['diag'] = diag_summary(DIAG['rows'], DIAG['names'])
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
    if os.environ.get('P5_BAL') == '1':   # reg. 5b balance gate
        g['баланс'] = (yrs[2023] > 0 and yrs[2026] > 0 and min(yrs.values()) >= -0.3, f'2023 {yrs[2023]:+.2f}, 2026 {yrs[2026]:+.2f}, мин {min(yrs.values()):+.2f}')
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
