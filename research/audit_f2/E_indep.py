#!/usr/bin/env python3
"""Independent re-implementation (audit E) of two F2 sleeves (core, fcarry) and the portfolio accounting.
Does NOT import bt.py / exp_common.py / strategies.py. Written from the spec read in those files:
  * panel index = bar open time; decision on bar i (closed at i+1h) executes at close[i], earns ret[i+1]
  * funding stored at the hour it is paid; a position held at that hour pays pos*rate (long pays rate>0)
  * rebalance every 8h on bars with hour%8==0 (exec 01/09/17 UTC); daily sleeves refresh on hour%24==0
  * EMA 0.5 of target weights, vol target 0.6%/day on the unlevered book (30d of 24h sums), caps, neutrality,
    gross<=2x, dust 15$, band 0.5, fees 5.5/2bp maker share 0.7, slip max(1, 8*(t24/1e6)^-0.35)*0.3 bp.
Usage: python3 E_indep.py <run_name> [...]   run names: core_raw fcarry_raw core_full fcarry_full combo_full
Outputs logs/E_indep_<run>.pkl (hourly equity, components, per-rebalance diagnostics)."""
import os, sys, time, pickle, numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PAN = os.path.join(ROOT, 'data', 'panels'); LOGS = os.path.join(ROOT, 'logs')
START, END, OOS = pd.Timestamp('2021-01-01', tz='UTC'), pd.Timestamp(os.environ.get('END', '2026-08-31 23:00'), tz='UTC'), pd.Timestamp('2022-01-01', tz='UTC')


class Panel:
    def __init__(self):
        t0 = time.time(); a = START - pd.Timedelta(days=60)
        qv = pd.read_parquet(os.path.join(PAN, 'qvol.parquet')).loc[a:END]
        t24 = qv.rolling(24, min_periods=6).sum()
        keep = [c for c in t24.columns if str(c).isascii() and np.nanmax(t24[c].values) >= 5e6]
        self.sym = np.array(keep); self.t24 = t24[keep].values.astype(np.float32); del t24
        px = pd.read_parquet(os.path.join(PAN, 'close.parquet'), columns=keep).loc[a:END]
        self.time = px.index; P = px.values.astype(np.float64); del px
        F = pd.read_parquet(os.path.join(PAN, 'funding.parquet'), columns=keep).loc[a:END].values.astype(np.float64)
        self.fund = np.nan_to_num(F, nan=0.0).astype(np.float32); del F
        T, N = P.shape; self.T, self.N = T, N
        self.ok = np.isfinite(P) & (P > 0)
        if os.environ.get('FREEZE') == '1':          # variant: >=24h runs of unchanged close with zero volume = no price (delisted)
            Qv = np.nan_to_num(qv[keep].values); fz = np.zeros((T, N), bool)
            fz[1:] = (P[1:] == P[:-1]) & (Qv[1:] == 0) & self.ok[1:]
            for j in range(N):
                d_ = np.diff(np.r_[0, fz[:, j].astype(np.int8), 0]); st = np.flatnonzero(d_ == 1); en = np.flatnonzero(d_ == -1)
                for s_, e_ in zip(st, en):
                    if e_ - s_ >= 24: self.ok[s_:e_, j] = False
            print('FREEZE: frozen bars set invalid:', int((np.isfinite(P) & (P > 0) & ~self.ok).sum()), flush=True)
        # listing segments: a run of valid bars broken by >24h of missing bars
        self.age = np.full((T, N), -1, np.int32); self.last_bar = np.zeros((T, N), bool)
        for j in range(N):
            v = np.flatnonzero(self.ok[:, j])
            if not len(v): continue
            cut = np.flatnonzero(np.diff(v) > 24)
            st = np.r_[v[0], v[cut + 1]]; en = np.r_[v[cut], v[-1]]
            for s, e in zip(st, en):
                self.age[s:e + 1, j] = np.arange(e - s + 1)
                if e < T - 1: self.last_bar[e, j] = True
        Pf = pd.DataFrame(np.where(self.ok, P, np.nan)).ffill().values
        R = np.full((T, N), np.nan); R[1:] = Pf[1:] / Pf[:-1] - 1
        R[(self.age == 0) | ~self.ok] = np.nan
        self.R0 = np.nan_to_num(R, nan=0.0); del Pf, P
        self.hour = np.asarray(self.time.hour); self.hcount = ((self.time - pd.Timestamp('1970-01-01', tz='UTC')) // pd.Timedelta(hours=1)).values
        # cumulative sums for rolling stats
        okr = np.isfinite(R); X = np.where(okr, R, 0.0)
        z = np.zeros((1, N))
        self.cS = np.vstack([z, np.cumsum(X, 0)]); self.cSS = np.vstack([z, np.cumsum(X * X, 0)]); self.cN = np.vstack([z, np.cumsum(okr, 0)]).astype(np.float32)
        self.cF = np.vstack([z, np.cumsum(self.fund, 0, dtype=np.float64)]); del R, X, okr, qv
        self.majors = np.isin(self.sym, ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT'])
        print(f'Panel {T}x{N} {self.time[0]} -> {self.time[-1]} ({time.time()-t0:.0f}s)', flush=True)

    def eligible(self, i, min_age, min_t24, top):
        t = self.t24[i]
        m = (self.age[i] >= min_age) & self.ok[i] & np.isfinite(t) & (t >= min_t24)
        if top and m.sum() > top:
            idx = np.flatnonzero(m); best = idx[np.argsort(-t[idx], kind='stable')[:top]]
            m = np.zeros(self.N, bool); m[best] = True
        return m


TIE = os.environ.get('TIE', 'stable'); RNG = np.random.default_rng(int(os.environ.get('SEED', '0')))


def ls_quantile(score, frac=0.2, min_n=10):
    w = np.zeros(len(score)); fin = np.flatnonzero(np.isfinite(score))
    if len(fin) < min_n: return w
    k = max(2, int(round(len(fin) * frac)))
    if TIE == 'npdefault':                         # replicate numpy's default (unstable) sort over the full vector, as bt does
        o = np.argsort(np.where(np.isfinite(score), score, -np.inf))[-len(fin):]
    elif TIE == 'random':
        o = fin[np.lexsort((RNG.random(len(fin)), score[fin]))]
    else:
        o = fin[np.argsort(score[fin], kind='stable')]
    w[o[-k:]] = 1.0 / k; w[o[:k]] = -1.0 / k
    return w


def sig_core(p, i):
    m = p.eligible(i, 720, 1e7, 100); L = 336
    S = p.cS[i + 1] - p.cS[i + 1 - L]; SS = p.cSS[i + 1] - p.cSS[i + 1 - L]; n = p.cN[i + 1] - p.cN[i + 1 - L]
    with np.errstate(all='ignore'):
        mu = S / n; sd = np.sqrt(np.maximum((SS - S * S / n) / (n - 1), 0))
        sc = np.where(m & (n >= 0.6 * L) & (sd > 0), mu / sd, np.nan)
    return ls_quantile(sc, 0.2)


def sig_fcarry(p, i):
    m = p.eligible(i, 720, 1e7, 150)
    f72 = p.cF[i + 1] - p.cF[i + 1 - 72]          # funding paid at hours i-71..i (all known at bar-i close)
    return ls_quantile(np.where(m, -f72, np.nan), 0.2)


class Daily:                                       # refresh only on the 00:00 bar, otherwise keep previous weights
    def __init__(self, f): self.f = f; self.w = None
    def __call__(self, p, i):
        if p.hour[i] == 0 or self.w is None: self.w = self.f(p, i) if p.hour[i] == 0 else np.zeros(p.N)
        return self.w


def simulate(p, sleeves, full=True, remask=False, fee_t=5.5, fee_m=2.0, maker=0.7, slip_scale=0.3, start_eq=1e4, rec_pos=False):
    T, N = p.T, p.N
    i0 = max(int(p.time.searchsorted(OOS)), 840); i_last = T - 1
    unit_cost = (fee_t * (1 - maker) + fee_m * maker) / 1e4
    eq = start_eq; pos = np.zeros(N); prev_w = None; lev = 1.0; lev_real = 1.0
    E = np.full(T, np.nan); comp = {k: np.zeros(T) for k in ('price', 'fund', 'fee', 'slip', 'shock', 'traded')}
    unlev = []; diag = []; held_frozen = []; w_cur = None; VTPAPER = os.environ.get('VTPAPER') == '1'
    for i in range(i0, i_last):
        if i > i0:                                   # funding paid at hour i on position held at hour i
            fpay = -float(pos @ p.fund[i]); eq += fpay; comp['fund'][i] = fpay
        dl = (pos != 0) & p.last_bar[i]              # forced exit at the last close of a listing segment
        if dl.any():
            sh = 0.05 * np.abs(pos[dl]).sum(); eq -= sh; comp['shock'][i] = -sh; pos[dl] = 0
        if p.hcount[i] % 8 == 0:
            outer = p.ok[i] & np.isfinite(p.t24[i])
            w = sum(s * f(p, i) for f, s in sleeves) / sum(s for _, s in sleeves)
            w = np.where(outer, w, 0.0)
            if full:
                if prev_w is not None: w = 0.5 * w + 0.5 * prev_w
                if remask: w = np.where(outer, w, 0.0)    # variant: names without a price cannot keep a smoothed target
                prev_w = w.copy()
                if len(unlev) >= 240:
                    h = np.asarray(unlev[-720:]); h = h[len(h) % 24:]
                    if len(h) >= 48:
                        dv = h.reshape(-1, 24).sum(1).std(ddof=1)
                        if dv > 1e-6: lev = min(max(0.006 / dv, 0.05), 3.0)
                tgt = w * lev * eq
                cap_l = np.where(p.majors, 0.30, 0.02) * eq; cap_s = np.where(p.majors, 0.30, 0.015) * eq
                tgt = np.clip(tgt, -cap_s, cap_l)
                L, S = tgt[tgt > 0].sum(), -tgt[tgt < 0].sum()
                if L > 0 and S > 0:
                    tgt[tgt > 0] *= min(L, S) / L; tgt[tgt < 0] *= min(L, S) / S
                g = np.abs(tgt).sum()
                if g > 2.0 * eq: tgt *= 2.0 * eq / g
                tgt[np.abs(tgt) < 15.0] = 0.0
                gw = np.abs(w).sum(); lev_real = np.abs(tgt).sum() / (eq * gw) if gw > 1e-12 else lev
                d = tgt - pos; close_all = (tgt == 0) & (pos != 0)
                go = close_all | (np.abs(d) >= np.maximum(15.0, 0.5 * np.abs(tgt)))
                d = np.where(go, d, 0.0)
            else:
                tgt = w * eq; d = tgt - pos; lev_real = 1.0
            q = np.abs(d)
            if q.sum() > 0:
                t = p.t24[i]; t = np.where(np.isfinite(t) & (t > 0), t, 1e5)
                slip = q * np.maximum(1.0, 8.0 * (t / 1e6) ** -0.35) * slip_scale / 1e4
                fee = q * unit_cost
                eq -= fee.sum() + slip.sum(); comp['fee'][i] = -fee.sum(); comp['slip'][i] = -slip.sum(); comp['traded'][i] = q.sum()
                pos = pos + d
            w_cur = w.copy()
            diag.append((i, lev, lev_real, np.abs(pos).sum() / eq, np.abs(w).sum(), int((pos != 0).sum())))
        E[i] = eq
        r = p.R0[i + 1]; pnl = float(pos @ r)
        if rec_pos:                                  # positions in names whose next bar has no trading at all
            fr = (pos != 0) & ~p.ok[i + 1]              # held into a bar without a price -> zero return booked
            fz = (pos != 0) & p.ok[i + 1] & (np.nan_to_num(p.t24[i + 1]) <= 0)   # price present but no volume in 24h
            if fr.any() or fz.any(): held_frozen.append((i + 1, float(np.abs(pos[fr]).sum() / eq), float(np.abs(pos[fz]).sum() / eq), list(p.sym[fr | fz])))
        if VTPAPER: unlev.append(float(w_cur @ r) if w_cur is not None else 0.0)   # variant: exogenous paper return of current weights
        else: unlev.append(pnl / max(eq * lev_real, 1e-9))
        comp['price'][i + 1] = pnl; eq += pnl; pos = pos * (1 + r)
    E[i_last] = eq
    sl = slice(i0, i_last + 1)
    out = dict(eq=pd.Series(E[sl], p.time[sl]), comp={k: pd.Series(v[sl], p.time[sl]) for k, v in comp.items()},
               diag=pd.DataFrame(diag, columns=['i', 'lev', 'lev_real', 'gross', 'gw', 'npos']), held_frozen=held_frozen)
    return out


RUNS = {'core_raw': ([('core', 1)], False), 'fcarry_raw': ([('fcarry', 1)], False),
        'core_full': ([('core', 1)], True), 'fcarry_full': ([('fcarry', 1)], True),
        'combo_full': ([('core', 1), ('fcarry', 1)], True), 'combo_full_remask': ([('core', 1), ('fcarry', 1)], True)}

if __name__ == '__main__':
    p = Panel()
    for name in sys.argv[1:]:
        t0 = time.time(); spec, full = RUNS[name]
        sleeves = [(Daily(sig_core) if k == 'core' else sig_fcarry, s) for k, s in spec]
        out = simulate(p, sleeves, full=full, remask=name.endswith('remask'), rec_pos=True)
        e = out['eq']; dd = e.groupby(e.index.floor('D')).last().pct_change().dropna()
        print(f'{name}: Sharpe {dd.mean()/dd.std()*np.sqrt(365):.2f} end {e.iloc[-1]:.0f} ({time.time()-t0:.0f}s)', flush=True)
        pickle.dump(out, open(os.path.join(LOGS, f'E_indep_{name}{os.environ.get("TAG", "")}.pkl'), 'wb'))
