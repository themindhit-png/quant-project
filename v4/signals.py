#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sleeves for PROP-SLEEVES v4. Each returns a weight vector over the panel's symbols where each side
sums to ~1 (dollar-neutral, gross ~1). Math mirrors research engine bt.py / strategies.py exactly."""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
from ml_features import compute_features, rank_norm, FEATURES, NEED_H as ML_NEED_H
from ml_features_8h import compute_features_8h, FEATURES_8H

try:
    import lightgbm as lgb
except Exception:       # pragma: no cover
    lgb = None


def quantile_ls(sig, top_frac=0.2, min_n=10, inv_vol=None):
    """Long top / short bottom quantile; each side sums to 1.0. inv_vol: per-name vol -> weights within a side are
    proportional to 1/vol (names without a vol estimate get weight 0). Mirrors bt.quantile_ls exactly."""
    s = np.where(np.isfinite(sig), sig, np.nan)
    n = int(np.isfinite(s).sum())
    w = np.zeros_like(s)
    if n < min_n:
        return w
    # degenerate-signal guard (Opus P1): a near-constant signal (dead funding feed, constant model output) would split into
    # quantiles by array order -> refuse (empty sleeve -> the cycle is aborted by main's empty-sleeve fail-safe)
    if np.unique(s[np.isfinite(s)]).size < min(C.SIG_MIN_DISTINCT, max(2, n // 2)):
        return w
    k = max(2, int(round(n * top_frac)))
    order = np.argsort(np.where(np.isfinite(s), s, -np.inf))
    valid_idx = order[-n:]
    shorts, longs = valid_idx[:k], valid_idx[-k:]
    if inv_vol is None:
        w[longs] = 1.0 / k; w[shorts] = -1.0 / k
    else:
        iv = 1.0 / np.where(np.isfinite(inv_vol) & (inv_vol > 0), inv_vol, np.nan)
        wl = np.nan_to_num(iv[longs]); ws = np.nan_to_num(iv[shorts])
        if wl.sum() > 0: w[longs] = wl / wl.sum()
        if ws.sum() > 0: w[shorts] = -ws / ws.sum()
    return w


class Universe:
    """Point-in-time eligibility with hysteresis (held names stay while ranked < top_n + extra)."""
    def __init__(self):
        self.prev = {}     # name -> set of symbols

    def select(self, name, symbols, t24_now, age_h, valid_now, stale, min_age_h, min_turn, top_n, extra, blacklist, delist):
        N = len(symbols)
        m = np.array([(s not in blacklist) and (s not in delist) for s in symbols])
        m &= (age_h >= min_age_h) & valid_now & (~stale) & np.isfinite(t24_now) & (t24_now >= min_turn)
        if top_n and m.sum() > top_n:
            score = np.where(m, t24_now, -np.inf)
            order = np.argsort(-score)
            rank = np.empty(N, dtype=np.int64); rank[order] = np.arange(N)
            keep = rank < top_n
            prev = self.prev.get(name, set())
            if prev:
                held = np.array([s in prev for s in symbols])
                keep |= held & (rank < top_n + extra)
            m &= keep
        self.prev[name] = {symbols[j] for j in np.flatnonzero(m)}
        return m


class Sleeves:
    def __init__(self, log=print):
        self.log = log
        self.uni = Universe()
        self.models = {}
        self.held = {}          # daily sleeve name -> {symbol: weight} held between daily updates
        if lgb is not None:
            for hz in sorted({int(s[2:]) for s in C.SLEEVES if s.startswith('ml') and s[2:].isdigit() and s != 'ml8'} | {3, 7}):
                p = os.path.join(C.MODEL_DIR, f'ml_h{hz}{C.ML_MODEL_SUFFIX}.txt')
                if os.path.exists(p):
                    self.models[hz] = lgb.Booster(model_file=p)
                    feats_cfg = [f for f in FEATURES if f not in C.ML_DROP_FEATS]
                    self._check_spec(p, self.models[hz], feats_cfg, os.path.join(C.MODEL_DIR, f'spec{C.ML_MODEL_SUFFIX}.json'))
            p8 = os.path.join(C.MODEL_DIR, 'ml_8h.txt')
            if os.path.exists(p8):
                self.models[8] = lgb.Booster(model_file=p8)
                self._check_spec(p8, self.models[8], list(FEATURES_8H), os.path.join(C.MODEL_DIR, 'spec_8h.json'))
            self.log(f'ML models loaded: {sorted(self.models)}')
        else:
            self.log('WARN: lightgbm not available — ML sleeves disabled')

    # ------------------------------------------------------------ helpers
    @staticmethod
    def _check_spec(model_path, mdl, feats_cfg, spec_path):
        """Refuse to run when the model's training spec (feature list, in order) differs from what the bot will feed it.
        Models were trained on numpy arrays, so LightGBM's own feature names are generic — the spec json is the contract."""
        import json, hashlib
        n_feat = int(mdl.num_feature())
        if n_feat != len(feats_cfg):
            raise SystemExit(f'FATAL: model {model_path} expects {n_feat} features, bot provides {len(feats_cfg)}')
        if not os.path.exists(spec_path):
            raise SystemExit(f'FATAL: spec file {spec_path} missing — cannot verify feature order for {model_path}')
        spec = json.load(open(spec_path)); feats_spec = list(spec.get('features') or [])
        if feats_spec != list(feats_cfg):
            diff = [(a, b) for a, b in zip(feats_spec, feats_cfg) if a != b][:5]
            raise SystemExit(f'FATAL: feature spec mismatch for {model_path}: spec {len(feats_spec)} vs bot {len(feats_cfg)}; first diffs {diff}')
        with open(spec_path, 'rb') as f:
            spec_hash = hashlib.sha256(f.read()).hexdigest()[:12]
        print(f'model {os.path.basename(model_path)}: {n_feat} features verified against {os.path.basename(spec_path)} (sha256 {spec_hash})')

    @staticmethod
    def _last(panel):
        return panel['cff'].shape[0] - 1

    def core_universe(self, panel, age_h, held):
        i = self._last(panel)
        return self.uni.select('core', panel['symbols'], panel['t24'][i], age_h, panel['valid'][i], panel['stale'],
                               C.UNI_MIN_AGE_D * 24, C.UNI_MIN_TURN, C.UNI_TOP_CORE, C.UNI_EXIT_EXTRA, C.BLACKLIST, panel.get('delist', set()))

    def ml_universe(self, panel, age_h):
        i = self._last(panel)
        return self.uni.select('ml', panel['symbols'], panel['t24'][i], age_h, panel['valid'][i], panel['stale'],
                               C.ML_MIN_AGE_D * 24, C.UNI_MIN_TURN, C.UNI_TOP_ML, C.UNI_EXIT_EXTRA, C.BLACKLIST, panel.get('delist', set()))

    # ------------------------------------------------------------ sleeves
    def core(self, panel, mask):
        """Trend quality: mean/std of hourly returns over CORE_LB_H closed bars; quantile L/S."""
        cff = panel['cff']; i = self._last(panel)
        sub = cff[i - C.CORE_LB_H:i + 1]
        r = sub[1:] / sub[:-1] - 1.0
        # returns over bars without an actual close (data gaps) are unknown, not zero (as in research)
        r = np.where(panel['valid'][i - C.CORE_LB_H + 1:i + 1], r, np.nan)
        cnt = np.isfinite(r).sum(0)
        mu = np.nanmean(r, 0); sd = np.nanstd(r, 0, ddof=1)
        sig = np.where((cnt >= C.CORE_LB_H * 0.6) & (sd > 0) & mask, mu / sd, np.nan)
        return quantile_ls(sig, C.CORE_TOP_FRAC)

    def ml(self, panel, mask, age_h, hz):
        mdl = self.models.get(hz)
        if mdl is None:
            return np.zeros(len(panel['symbols']))
        if panel['cff'].shape[0] < ML_NEED_H:
            self.log('ML: insufficient history'); return np.zeros(len(panel['symbols']))
        syms = panel['symbols']
        if 'BTCUSDT' not in syms:
            self.log('ML: BTCUSDT not in panel'); return np.zeros(len(syms))
        jb = syms.index('BTCUSDT')
        F = compute_features(panel['cff'][-ML_NEED_H:], panel['fund'][-ML_NEED_H:], panel['t24'][-ML_NEED_H:],
                             panel['prem'][-ML_NEED_H:], panel['high'][-ML_NEED_H:], panel['low'][-ML_NEED_H:], age_h, jb)
        idx = np.flatnonzero(mask)
        if len(idx) < 20:
            return np.zeros(len(syms))
        feats = [f for f in FEATURES if f not in C.ML_DROP_FEATS]
        X = np.column_stack([F[f][idx] for f in feats]).astype(np.float32)
        Xr = rank_norm(X)
        pred = mdl.predict(Xr.astype(np.float64))      # LightGBM handles NaN natively (as in training)
        sig = np.full(len(syms), np.nan); sig[idx] = pred
        return quantile_ls(sig, C.ML_TOP_FRAC)

    def fund_universe(self, panel, age_h):
        i = self._last(panel)
        return self.uni.select('fund', panel['symbols'], panel['t24'][i], age_h, panel['valid'][i], panel['stale'],
                               C.UNI_MIN_AGE_D * 24, C.UNI_MIN_TURN, C.UNI_TOP_ML, C.UNI_EXIT_EXTRA, C.BLACKLIST, panel.get('delist', set()))

    def fund_change(self, panel, mask):
        """Contrarian to the 24h change in funding: rising funding = crowding longs -> short; quantile L/S (research fund_change)."""
        f = np.where(np.isfinite(panel['fund']), panel['fund'], 0.0); i = self._last(panel)
        if i < 48:
            return np.zeros(len(panel['symbols']))
        sig = -(f[i - 23:i + 1].sum(0) - f[i - 47:i - 23].sum(0))
        return quantile_ls(np.where(mask, sig, np.nan), C.FUND_TOP_FRAC)

    def fund_carry(self, panel, mask):
        """Funding-level carry: short high (positive) 72h funding, long negative; quantile L/S (research fund_level)."""
        f = np.where(np.isfinite(panel['fund']), panel['fund'], 0.0); i = self._last(panel)
        if i < 72:
            return np.zeros(len(panel['symbols']))
        return quantile_ls(np.where(mask, -f[i - 71:i + 1].sum(0), np.nan), C.FUND_TOP_FRAC)

    def ml8_universe(self, panel, age_h):
        """ml8 trades only names >= ML8_MIN_AGE_D old (young names are a jump lottery on an 8h grid; exp20b/20h)."""
        i = self._last(panel)
        return self.uni.select('ml8', panel['symbols'], panel['t24'][i], age_h, panel['valid'][i], panel['stale'],
                               C.ML8_MIN_AGE_D * 24, C.UNI_MIN_TURN, C.UNI_TOP_ML, C.UNI_EXIT_EXTRA, C.BLACKLIST, panel.get('delist', set()))

    def ml8_vol(self, panel):
        """Std of hourly returns over the last ML8_VOL_H closed bars (gap bars -> unknown), >= 72 obs; else NaN.
        Mirrors the research VOL panel (rolling 168h std, min_periods 72)."""
        cff = panel['cff']; i = self._last(panel); H = C.ML8_VOL_H
        if i < H:
            return np.full(cff.shape[1], np.nan)
        sub = cff[i - H:i + 1]
        r = sub[1:] / sub[:-1] - 1.0
        r = np.where(panel['valid'][i - H + 1:i + 1], r, np.nan)
        cnt = np.isfinite(r).sum(0)
        with np.errstate(all='ignore'):
            sd = np.nanstd(r, 0, ddof=1)
        return np.where(cnt >= 72, sd, np.nan)

    def beta_btc(self, panel):
        """Beta of every symbol to BTC over the last BETA_WIN_H hourly returns (gap bars -> 0, as in research betautil)."""
        syms = panel['symbols']; cff = panel['cff']; i = self._last(panel); W = C.BETA_WIN_H
        if 'BTCUSDT' not in syms or i < W:
            return np.full(len(syms), np.nan)
        sub = cff[i - W:i + 1]
        with np.errstate(all='ignore'):
            r = sub[1:] / sub[:-1] - 1.0
        r = np.where(panel['valid'][i - W + 1:i + 1] & np.isfinite(r), r, 0.0)
        y = r[:, syms.index('BTCUSDT')]; vy = y.var()
        if vy <= 0:
            return np.full(len(syms), np.nan)
        return ((r - r.mean(0)) * (y - y.mean())[:, None]).mean(0) / vy

    def apply_beta_cap(self, w, panel, info):
        """Hedge the book beta beyond ±BETA_CAP with BTC (constraint, not an alpha bet)."""
        syms = panel['symbols']
        if C.BETA_CAP < 0 or 'BTCUSDT' not in syms:
            return w
        b = self.beta_btc(panel); bb = float(np.nansum(w * np.where(np.isfinite(b), b, 1.0)))
        info['beta_book'] = round(bb, 3)
        if abs(bb) > C.BETA_CAP:
            w = w.copy(); w[syms.index('BTCUSDT')] -= (bb - (C.BETA_CAP if bb > 0 else -C.BETA_CAP))
            info['beta_hedge_btc_w'] = round(float(w[syms.index('BTCUSDT')]), 4)
        return w

    def ml8_weights(self, sig, panel):
        return quantile_ls(sig, C.ML8_TOP_FRAC, inv_vol=self.ml8_vol(panel) if C.ML8_INV_VOL else None)

    def ml8(self, panel, mask, age_h, hour_utc):
        """Intraday sleeve: 8h-horizon model on the v3 feature set + intraday extras; inverse-vol weights."""
        mdl = self.models.get(8)
        if mdl is None or panel['cff'].shape[0] < ML_NEED_H:
            return np.zeros(len(panel['symbols']))
        syms = panel['symbols']
        if 'BTCUSDT' not in syms:
            return np.zeros(len(syms))
        jb = syms.index('BTCUSDT')
        F = compute_features_8h(panel['cff'][-ML_NEED_H:], panel['fund'][-ML_NEED_H:], panel['t24'][-ML_NEED_H:], panel['prem'][-ML_NEED_H:],
                                panel['high'][-ML_NEED_H:], panel['low'][-ML_NEED_H:], age_h, jb, hour_utc)
        idx = np.flatnonzero(mask)
        if len(idx) < 20:
            return np.zeros(len(syms))
        X = np.column_stack([F[f][idx] for f in FEATURES_8H]).astype(np.float32)
        pred = mdl.predict(rank_norm(X).astype(np.float64))
        sig = np.full(len(syms), np.nan); sig[idx] = pred
        return self.ml8_weights(sig, panel)

    def listing(self, panel, age_h, majors_idx):
        """Short young liquid perps (age window, two-clock age), long majors equally; pump filter."""
        i = self._last(panel); syms = panel['symbols']; N = len(syms)
        t = panel['t24'][i]
        cand = (age_h >= C.LIST_AGE_MIN_D * 24) & (age_h <= C.LIST_AGE_MAX_D * 24) & panel['valid'][i] & (~panel['stale'])
        cand &= np.isfinite(t) & (t >= C.LIST_MIN_TURN) & (panel['data_age'] >= 72)
        cand &= np.array([(s not in C.BLACKLIST) and (s not in panel.get('delist', set())) for s in syms])
        cand[majors_idx] = False
        r24 = panel['cff'][i] / panel['cff'][i - 24] - 1.0
        cand &= ~(r24 > C.LIST_PUMP_EXCL)
        idx = np.flatnonzero(cand)
        w = np.zeros(N)
        if len(idx) == 0 or len(majors_idx) == 0:
            return w
        idx = idx[np.argsort(-t[idx])][:C.LIST_MAX_NAMES]
        ws = min(1.0 / len(idx), C.LIST_W_NAME)
        w[idx] = -ws
        w[majors_idx] += ws * len(idx) / len(majors_idx)
        return w

    # ------------------------------------------------------------ combination
    def combined(self, panel, age_h, held, age_ml=None, daily_update=True, hour_utc=0, panel_daily=None):
        """age_h: hours since earliest listing (listing sleeve, eligibility). age_ml: Binance-clock age for
        the ML feature (falls back to age_h). daily_update: recompute the daily sleeves (listing/core/ml3/ml7/fchg);
        otherwise their last weights are held (stored per symbol in self.held). ml8/fcarry are recomputed at every call.
        panel_daily: optional panel ending on the 00:00 UTC bar (same symbols) — used for the daily sleeves when the
        daily update is a late catch-up, so they are computed on the bar the research used, not on the current bar."""
        syms = panel['symbols']; N = len(syms)
        age_ml = age_h if age_ml is None else np.where(age_ml >= 0, age_ml, age_h)
        majors_idx = np.array([syms.index(s) for s in C.MAJORS if s in syms], dtype=np.int64)
        parts, info = {}, {}
        daily = [s for s in C.SLEEVES if s in ('listing', 'core', 'fchg') or (s.startswith('ml') and s[2:].isdigit() and s != 'ml8')]
        pdl = panel if panel_daily is None or list(panel_daily.get('symbols', [])) != list(syms) else panel_daily
        if daily_update or any(k not in self.held for k in daily):
            if 'core' in daily:
                mc = self.core_universe(pdl, age_ml, held)
                parts['core'] = self.core(pdl, mc); info['core_uni'] = int(mc.sum())
            mls = [s for s in daily if s.startswith('ml')]
            if mls:
                mm = self.ml_universe(pdl, age_ml); info['ml_uni'] = int(mm.sum())
                for s in mls:
                    parts[s] = self.ml(pdl, mm, age_ml, int(s[2:]))
            if 'fchg' in daily:                    # funding-change contrarian (exp24a/e), daily held
                parts['fchg'] = self.fund_change(pdl, self.fund_universe(pdl, age_ml))
            if 'listing' in daily:
                parts['listing'] = self.listing(pdl, age_h, majors_idx)
                info['listing_n'] = int((parts['listing'] < 0).sum())
            info['daily_on_00_bar'] = pdl is panel_daily
            for k, v in parts.items():
                self.held[k] = {syms[j]: float(v[j]) for j in np.flatnonzero(v != 0)}
            info['daily_update'] = True
        else:
            for k in daily:
                parts[k] = np.array([self.held[k].get(s, 0.0) for s in syms])
            info['daily_update'] = False
        if 'ml8' in C.SLEEVES:
            mm8 = self.ml8_universe(panel, age_ml); info['ml8_uni'] = int(mm8.sum())
            parts['ml8'] = self.ml8(panel, mm8, age_ml, hour_utc); info['ml8_n'] = int((parts['ml8'] != 0).sum())
        if 'fcarry' in C.SLEEVES:                  # funding-level carry (exp24a/e), recomputed every rebalance (8h holding)
            parts['fcarry'] = self.fund_carry(panel, self.fund_universe(panel, age_ml)); info['fcarry_n'] = int((parts['fcarry'] != 0).sum())
        w = np.zeros(N); tot = 0.0
        for k, v in parts.items():
            s = C.SLEEVE_SCALES.get(k, 2.0 if k == 'ml8' else 1.0)
            w += s * v; tot += s
        if tot > 0:
            w /= tot
        w = self.apply_beta_cap(w, panel, info)
        info['gross_w'] = float(np.abs(w).sum())
        return w, parts, info
