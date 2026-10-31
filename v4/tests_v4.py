#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit/consistency tests for PROP-SLEEVES v4 against the research engine (small data window)."""
import os, sys, json, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
os.environ.update(dict(FIRM='hyrotrader_2step', MODE='challenge', ACCOUNT_SIZE='10000', BYBIT_API_KEY='x', BYBIT_API_SECRET='x',
                       USE_BINANCE_CLOCK='0', TG_BOT_TOKEN='', MODEL_DIR=os.path.join(HERE, 'models'),
                       ML_MODEL_SUFFIX='_noage', ML_DROP_FEATS='age_d', ML_MIN_AGE_D='90'))          # the production ML setup (evns/base.env)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)
# Real Binance panels (data/panels, not in git) when present; otherwise (CI, fresh clone, or V4_TESTS_SYNTH=1) synthetic ones
import bt, tempfile
REAL_PANELS = os.path.exists(os.path.join(bt.PAN, 'close.parquet')) and os.environ.get('V4_TESTS_SYNTH', '0') != '1'
if not REAL_PANELS:
    from tests_synth import make_panels
    bt.PAN = make_panels(tempfile.mkdtemp(prefix='v4_synth_panels_'))
print(f'panels: {"REAL " + bt.PAN if REAL_PANELS else "SYNTHETIC (tests_synth.py, seed 7)"}')
import config as C
from signals import Sleeves, quantile_ls as q_bot
from risk import RiskEngine
from execution import Executor
from state import Memory
from bt import Data, quantile_ls as q_res
from strategies import core_signal

res = []
def check(name, cond, detail=''):
    res.append((name, bool(cond))); print(f'{"PASS" if cond else "FAIL"}  {name} {detail}')

d = Data(start='2025-06-01', end='2025-09-30 23:00', min_turn_ever=5e6, load_hl=True, load_premium=True)
T, N = d.ret.shape
syms = [str(s) for s in d.cols]
i = T - 1 - 24 * 10
sl = slice(i - 2199, i + 1)
js = np.arange(N)
panel = dict(grid=d.hh[sl], cff=d.cff[sl], high=d.high[sl].astype(np.float64), low=d.low[sl].astype(np.float64),
             t24=d.t24[sl].astype(np.float64), fund=d.fund[sl].astype(np.float64), prem=d.prem[sl].astype(np.float64),
             valid=d.valid[sl], stale=~d.valid[i], data_age=d.age[i].astype(float), symbols=syms, delist=set())
age_h = d.age[i].astype(float)
S = Sleeves(log=lambda *a: None)

# T1: quantile_ls identical
sig = np.random.default_rng(0).standard_normal(N); sig[::7] = np.nan
check('T1 quantile_ls bot == research', np.allclose(q_bot(sig, 0.2), q_res(sig, 0.2)))

# T2: core weights == research core (same universe)
d.prev_mask = None
m_res = d.universe(i, min_age_h=720, min_turn=1e7, top_n=100)
w_res = q_res(core_signal(d, i, m_res, 336), 0.2)
S.uni.prev = {}
m_bot = S.core_universe(panel, age_h, set())
check('T2a core universe == research universe', (m_bot == m_res).all(), f'bot {m_bot.sum()} res {m_res.sum()} diff {int((m_bot != m_res).sum())}')
w_bot = S.core(panel, m_bot)
check('T2b core weights == research', np.allclose(w_bot, w_res, atol=1e-9), f'max diff {np.abs(w_bot - w_res).max():.2e}')

# T3: listing sleeve == exp13b sl_listing (two-clock) — bot uses age_hours from md; here same age array
instr = json.load(open(os.path.join(ROOT, 'instruments.json')))
launch_h = np.array([instr[s]['launch'] / 3.6e6 if s in instr and instr[s].get('launch') else np.nan for s in syms])
age2 = np.maximum(age_h, np.where(np.isfinite(launch_h), d.hh[i] - launch_h, -np.inf))
jm = np.array([syms.index(s) for s in C.MAJORS if s in syms])
w_l = S.listing(panel, age2, jm)
t = d.t24[i]
cand = (age2 >= 24 * 7) & (age2 <= 24 * 90) & d.valid[i] & np.isfinite(t) & (t >= 2e7) & (d.age[i] >= 72)
cand[jm] = False; r24 = d.cff[i] / d.cff[i - 24] - 1; cand &= ~(r24 > 0.30)
idx = np.flatnonzero(cand); idx = idx[np.argsort(-t[idx])][:30]
w_ref = np.zeros(N)
if len(idx):
    ws = min(1.0 / len(idx), 0.05); w_ref[idx] = -ws; w_ref[jm] += ws * len(idx) / len(jm)
check('T3 listing sleeve == research sl_listing', np.allclose(w_l, w_ref, atol=1e-12), f'n short {int((w_l < 0).sum())}')

# T4: ML sleeve runs and is dollar neutral (models optional)
mm = S.ml_universe(panel, age_h)
if S.models:
    w3 = S.ml(panel, mm, age_h, 3)
    check('T4 ML weights neutral & sized', abs(w3.sum()) < 1e-9 and abs(np.abs(w3).sum() - 2.0) < 1e-6 and (w3 != 0).sum() > 40, f'n {int((w3 != 0).sum())}')
else:
    print('SKIP  T4 (no models yet)')

# T5: risk engine floors and K
mem = Memory(); mem.save = lambda: None
R = RiskEngine(mem, log=lambda *a: None)
from datetime import datetime, timezone
now = datetime(2025, 9, 1, 0, 10, tzinfo=timezone.utc)
R.observe_equity(10000.0, now)
k0, why0 = R.multiplier(10000.0, now)
check('T5a K at start == BASE', abs(k0 - C.BASE_SCALE) < 1e-9, f'k {k0:.3f} ({why0})')
R.observe_equity(10600.0, now); k1, _ = R.multiplier(10600.0, now)
tf = R.total_floor()
check('T5b trailing floor follows peak', abs(tf - (10600 - 1000)) < 1e-6, f'floor {tf:.0f}')
from datetime import timedelta
now2 = now + timedelta(days=1)                # new day so the daily rule does not interfere with the CPPI check
R.observe_equity(9800.0, now2)                # buffer 200 of allowance 1000 -> 200/(0.5*1000)=0.4 -> K = 0.4*BASE
k2, why2 = R.multiplier(9800.0, now2)
check('T5c CPPI scales K', abs(k2 - C.BASE_SCALE * 0.4) < 1e-9, f'k {k2:.3f} ({why2})')
now3 = now2 + timedelta(days=1)
R.observe_equity(10000.0, now3); R.observe_equity(9700.0, now3)   # day pnl -300 -> -3% of initial >= halt 2.5%
k3, why3 = R.multiplier(9700.0, now3)
check('T5d daily halt triggers K=0', k3 == 0.0, why3)
k4, _ = R.multiplier(9640.0, now)
check('T5e floor guard K=0 near floor', k4 == 0.0)

# T6: build_targets caps, neutrality, gross, dust
mem2 = Memory(); mem2.save = lambda: None; R2 = RiskEngine(mem2, log=lambda *a: None); mem2['lev'] = 2.0
w = np.zeros(N); w[:40] = 0.05; w[40:60] = -0.02; w[60:80] = -0.03          # long 2.0, short 1.0 -> caps & neutrality
tgt, info = R2.build_targets(w, syms, 10000.0, np.full(N, 5e7), 1.0, set(C.MAJORS))
check('T6a neutrality after caps', abs(tgt.sum()) < 1e-6, f'net {tgt.sum():.4f}')
check('T6b long cap 2% / short cap 1.5%', tgt.max() <= 200 + 1e-6 and -tgt.min() <= 150 + 1e-6, f'max {tgt.max():.0f} min {tgt.min():.0f}')
check('T6c gross <= min(2x eq, notional_x*init*safety, margin)', np.abs(tgt).sum() <= info['g_max'] + 1e-6, f'gross {np.abs(tgt).sum():.0f} gmax {info["g_max"]:.0f}')
check('T6d dust rule', (np.abs(tgt[tgt != 0]) >= C.MIN_TRADE_USDT).all())
low = np.full(N, 5e7); low[0] = 1e6
tgt2, _ = R2.build_targets(w, syms, 10000.0, low, 1.0, set(C.MAJORS))
cap_low = C.RULES['lowcap_pct'] / 100.0 * R2.notional_base(10000.0) * C.SAFETY_MARGIN
check('T6e low-cap gross capped at lowcap_pct of base (Hyro rule)', abs(tgt2[0]) <= cap_low + 1e-6 and (tgt2[0] != 0.0 or tgt[0] == 0.0), f'|tgt| {abs(tgt2[0]):.0f} cap {cap_low:.0f}')

# T7: executor plan closes positions without price and outside targets
class DummyApi: pass
ex = Executor(DummyApi(), log=lambda *a: None, tg=lambda *a: None, dry_run=True)
positions = {'AAAUSDT': dict(qty=10.0, mark=5.0, entry=5.0, upnl=0, im=1, legs=[]), 'BBBUSDT': dict(qty=-100.0, mark=1.0, entry=1.0, upnl=0, im=1, legs=[])}
tickers = {'AAAUSDT': dict(mark=5.0, last=5.0, bid=5.0, ask=5.0), 'CCCUSDT': dict(mark=2.0, last=2.0, bid=2.0, ask=2.0)}
instr2 = {s: dict(qtyStep=0.1, minQty=0.1, maxQty=1e9, minNotional=5.0, tickSize=0.001) for s in ('AAAUSDT', 'BBBUSDT', 'CCCUSDT')}
legs = ex.plan({'CCCUSDT': 100.0}, positions, tickers, instr2, 0.3, 15.0)
kinds = {(l[0], l[1], l[3], l[4]) for l in legs}
check('T7a zombie AAA (not in targets) closed', ('AAAUSDT', 'Sell', True, True) in kinds)
check('T7b BBB without price closed at mark', ('BBBUSDT', 'Buy', True, True) in kinds)
check('T7c CCC opened', ('CCCUSDT', 'Buy', False, False) in kinds)
check('T7d reductions first', all(l[3] for l in legs[:2]))

# T8: inverse-vol quantile_ls identical to research; T9: ml8 vol == research VOL panel; ml8 universe age >= 180d
vol_res = pd.DataFrame(d.ret[sl]).rolling(168, min_periods=72).std().values[-1]
vol_bot = S.ml8_vol(panel)
both = np.isfinite(vol_res) & np.isfinite(vol_bot)
check('T9a ml8 vol == research rolling std', (np.isfinite(vol_res) == np.isfinite(vol_bot)).all() and np.allclose(vol_res[both], vol_bot[both], rtol=1e-6, atol=1e-12),
      f'n {int(both.sum())} max rel diff {np.nanmax(np.abs(vol_res[both] / vol_bot[both] - 1)) if both.any() else 0:.2e}')
check('T8 quantile_ls inverse-vol bot == research', np.allclose(q_bot(sig, 0.3, inv_vol=vol_bot), q_res(sig, 0.3, inv_vol=vol_res)))
S.uni.prev = {}
m8_bot = S.ml8_universe(panel, age_h); d.prev_mask = None; m8_res = d.universe(i, min_age_h=180 * 24, min_turn=1e7, top_n=150)
check('T9b ml8 universe (age>=180d) == research', (m8_bot == m8_res).all(), f'bot {m8_bot.sum()} res {m8_res.sum()}')
w8_bot = S.ml8_weights(np.where(m8_bot, sig, np.nan), panel); w8_res = q_res(np.where(m8_res, sig, np.nan), C.ML8_TOP_FRAC, inv_vol=vol_res)
check('T9c ml8 inverse-vol weights == research', np.allclose(w8_bot, w8_res, atol=1e-12) and abs(w8_bot.sum()) < 1e-9)

# T10: direct vol target — lev = target / std(unlevered daily returns); idempotent (no recursion)
mem3 = Memory(); mem3.save = lambda: None; R3 = RiskEngine(mem3, log=lambda *a: None)
rng = np.random.default_rng(1); r = rng.standard_normal(40) * 0.01
mem3['eq_daily'] = list(10000.0 * np.cumprod(1 + r)); mem3['mult_daily'] = [2.0] * 40
lev1, why = R3.update_lev(); lev2, _ = R3.update_lev()
rr = np.diff(np.array(mem3['eq_daily'][-(C.VT_WIN_D + 1):])) / np.array(mem3['eq_daily'][-(C.VT_WIN_D + 1):])[:-1]
exp_lev = float(np.clip(C.VT_TARGET_DVOL / (np.std(rr / 2.0, ddof=1)), 0.05, C.VT_MAX_LEV))
check('T10a direct VT lev = target / unlevered vol', abs(lev1 - exp_lev) < 1e-9, f'lev {lev1:.3f} expected {exp_lev:.3f} ({why})')
check('T10b VT idempotent (no multiplicative recursion)', abs(lev1 - lev2) < 1e-12)
mem3['mult_daily'] = [1.0] * 40; lev3, _ = R3.update_lev()
check('T10c halving the exposure multiplier halves lev', abs(lev3 - min(exp_lev / 2.0, C.VT_MAX_LEV)) < 1e-9 or lev3 >= C.VT_MAX_LEV - 1e-9, f'lev {lev3:.3f}')

# T11: beta to BTC == research betautil at the same bar; beta cap neutralises the book
from betautil import rolling_beta
jb = syms.index('BTCUSDT'); B_res = rolling_beta(d.ret[sl], jb, 720)[-1]; b_bot = S.beta_btc(panel)
ok = np.isfinite(B_res) & np.isfinite(b_bot)
check('T11a beta_btc == research rolling_beta', ok.sum() > 100 and np.allclose(B_res[ok], b_bot[ok], atol=1e-4), f'n {int(ok.sum())} max diff {np.abs(B_res[ok] - b_bot[ok]).max():.2e} beta(BTC) {b_bot[jb]:.3f}')
C.BETA_CAP = 0.0; infoB = {}
wtest = np.zeros(N); wtest[:60] = 1.0 / 60; wtest[60:120] = -1.0 / 60
wcap = S.apply_beta_cap(wtest, panel, infoB); bcap = float(np.nansum(wcap * np.where(np.isfinite(b_bot), b_bot, 1.0)))
check('T11b beta cap 0 neutralises book beta via BTC', abs(bcap) < 1e-9 and infoB.get('beta_book') is not None, f'before {infoB.get("beta_book")} after {bcap:.2e}')
C.BETA_CAP = -1.0

# T12: funding sleeves == research definitions (exp24a/e: fund_change / fund_level on the top-150 >=30d universe)
FUND0 = np.where(np.isfinite(d.fund), d.fund, 0.0)
mu = S.fund_universe(panel, age_h); d.prev_mask = None; mu_res = d.universe(i, min_age_h=720, min_turn=1e7, top_n=150)
check('T12a fund universe == research', (mu == mu_res).all(), f'bot {mu.sum()} res {mu_res.sum()}')
# reg. 4: research exp_common.qf = bt.quantile_ls_tb(sig, -last_rate(72 h), t24) (FUND_TIEBREAK=1, the default of both)
from bt import quantile_ls_tb as qtb_res, last_rate as lr_res
sec12 = -lr_res(FUND0[i - 71:i + 1])
sig_chg = -(FUND0[i - 23:i + 1].sum(0) - FUND0[i - 47:i - 23].sum(0)); w_chg_res = qtb_res(np.where(mu_res & np.isfinite(sig_chg), sig_chg, np.nan), sec12, d.t24[i], 0.2)
check('T12b fund_change weights == research', np.allclose(S.fund_change(panel, mu), w_chg_res, atol=1e-12), f'n {int((w_chg_res != 0).sum())}')
sig_lvl = -FUND0[i - 71:i + 1].sum(0); w_lvl_res = qtb_res(np.where(mu_res & np.isfinite(sig_lvl), sig_lvl, np.nan), sec12, d.t24[i], 0.2)
check('T12c fund_carry weights == research', np.allclose(S.fund_carry(panel, mu), w_lvl_res, atol=1e-12), f'n {int((w_lvl_res != 0).sum())}')



# ---------------------------------------------------------------- Opus round 3 (T13-T17)
# T13: payouts are cash flows, not losses: VT NAV returns and day anchor unchanged by a withdrawal
mem4 = Memory(); R4 = RiskEngine(mem4, lambda *a: None)
from datetime import datetime, timezone, timedelta
t0 = datetime(2026, 1, 1, 1, 0, tzinfo=timezone.utc)
eq = 10000.0
for k_ in range(40):
    eq *= 1.0 + 0.004 * ((-1) ** k_)
    R4.observe_equity(eq, t0 + timedelta(days=k_))
mem4['mult_daily'] = [1.0] * 40; lev_a, _ = R4.update_lev()
R4.note_cash_flow(-500.0); eq -= 500.0                       # payout of 500
R4.observe_equity(eq, t0 + timedelta(days=40)); R4.observe_equity(eq, t0 + timedelta(days=41))
mem4['mult_daily'] = [1.0] * 42; lev_b, _ = R4.update_lev()
check('T13a VT unchanged by a payout (NAV return excludes cash flows)', abs(lev_a - lev_b) / lev_a < 0.05, f'lev before {lev_a:.3f} after {lev_b:.3f}')
mem5 = Memory(); R5 = RiskEngine(mem5, lambda *a: None); R5.observe_equity(10000.0, t0); R5.note_cash_flow(-300.0)
met = R5.observe_equity(9700.0, t0 + timedelta(hours=1))
check('T13b day drawdown 0 after a payout', abs(met['day_dd']) < 1e-9, f'day_dd {met["day_dd"]:.2f} anchor {mem5["day_anchor"]:.0f}')

# T14: due-slot scheduling (missed slot is caught up; a start at hh:05 does not skip hh:10)
from main import Bot
class _API:
    def equity(self): return 10000.0
bot = Bot.__new__(Bot); bot.reb_every_base = 8; C.REBAL_EVERY_H = 8; C.REBAL_HOUR_UTC = 1; C.REBAL_MINUTE = 10
def _slot(dt): return bot.due_slot(dt)
check('T14a 01:15 -> slot 01', _slot(datetime(2026, 5, 3, 1, 15, tzinfo=timezone.utc)).hour == 1)
check('T14b 03:00 -> slot 01 (missed 01:10 is still due)', _slot(datetime(2026, 5, 3, 3, 0, tzinfo=timezone.utc)) == datetime(2026, 5, 3, 1, 10, tzinfo=timezone.utc))
check('T14c 00:30 -> previous day 17', _slot(datetime(2026, 5, 3, 0, 30, tzinfo=timezone.utc)) == datetime(2026, 5, 2, 17, 10, tzinfo=timezone.utc))
C.REBAL_EVERY_H = 24
check('T14d daily cadence: 09:00 -> today 01:10', _slot(datetime(2026, 5, 3, 9, 0, tzinfo=timezone.utc)) == datetime(2026, 5, 3, 1, 10, tzinfo=timezone.utc))
C.REBAL_EVERY_H = 8

# T15: effective config hash: deterministic, excludes secrets, changes with a risk parameter
cfg1, h1 = C.effective_config(); cfg2, h2 = C.effective_config()
check('T15a effective_config deterministic', h1 == h2 and len(h1) == 16)
check('T15b secrets / connection / paths excluded (real global names)', not any(k in cfg1 for k in (
    'API_KEY', 'API_SECRET', 'TG_TOKEN', 'TG_CHAT', 'STATUS_TOKEN', 'PORT', 'STATE_FILE', 'STATE_BACKUP_FILE', 'BASE_URL', 'BLACKLIST', 'MODEL_DIR')))
old = C.VT_TARGET_DVOL; C.VT_TARGET_DVOL = old * 1.5; _, h3 = C.effective_config(); C.VT_TARGET_DVOL = old
check('T15c hash changes with a risk parameter', h3 != h1)

# T16: batched PostOnly quotes with per-order codes (fake API) + per-cycle stats reset
class FakeAPI:
    def __init__(self): self.batches = []; self.singles = []
    def create_batch(self, orders): self.batches.append(orders); return [(0, '') if o['symbol'] != 'BADUSDT' else (110017, 'would cross') for o in orders]
    def post_only(self, *a, **k): self.singles.append(a)
    def market(self, *a, **k): self.singles.append(a)
    def cancel_all(self, *a, **k): return None
fa = FakeAPI(); ex = Executor(fa, lambda *a: None, None, False)
meta = dict(minQty=0.001, qtyStep=0.001, maxQty=1e9, tickSize=0.1, minNotional=5.0)
quotes = [('AAAUSDT', 'Buy', 1.0, False, '100.0', meta), ('BADUSDT', 'Sell', 2.0, False, '50.0', meta), ('CCCUSDT', 'Buy', 3.0, True, '10.0', meta)]
res_q = ex._send_quotes(quotes)
check('T16a quotes go out as ONE batch of 3', len(fa.batches) == 1 and len(fa.batches[0]) == 3 and not fa.singles, f'batches {len(fa.batches)} singles {len(fa.singles)}')
check('T16b per-order codes returned', res_q['AAAUSDT'][0] == 0 and res_q['BADUSDT'][0] == 110017 and fa.batches[0][2].get('reduceOnly') is True)
ex._reset_stats(); check('T16c stats reset per cycle, cumulative kept', ex.stats['maker_sent'] == 0 and ex.stats_total['maker_sent'] == 3)

# T17: kill rule measures the shadow (full-weight) ml8 series when present
mem6 = Memory(); bot6 = Bot.__new__(Bot); bot6.mem = mem6; bot6.reb_every_base = 8
hist = [{'date': f'2026-01-{1 + (k_ % 28):02d}', 'eq': 10000.0, 'ml8': 0.0, 'ml8_shadow': (-40.0 if k_ % 2 else 30.0)} for k_ in range(C.ML8_KILL_WIN_D)]
mem6['sleeve_pnl_hist'] = hist; mem6['ml8_mult'] = 0.0; mem6['ml8_mult_date'] = '2025-01-01'
C.SLEEVE_SCALES['ml8'] = 0.0; C.ML8_KILL_RULE = 1
bot6.apply_ml8_kill_rule(10000.0, datetime(2026, 6, 1, tzinfo=timezone.utc))
check('T17 shadow ml8 negative -> multiplier stays 0 and cadence 24h', mem6['ml8_mult'] == 0.0 and C.REBAL_EVERY_H == 24, f'mult {mem6["ml8_mult"]} every {C.REBAL_EVERY_H}')
C.REBAL_EVERY_H = 8; C.SLEEVE_SCALES['ml8'] = C.ML8_BASE_SCALE

# ---------------------------------------------------------------- Phase 3a (T15d, T18-T23)
import subprocess, inspect, types
from datetime import datetime, timezone, timedelta
def _cfg_hash(**env):
    e = dict(os.environ); e.update(env)
    return subprocess.run([sys.executable, os.path.join(HERE, 'config.py')], env=e, capture_output=True, text=True, timeout=60).stdout.strip()
ha = _cfg_hash(); hb = _cfg_hash(BYBIT_BASE_URL='https://api.bybit.com', PORT='12345', BLACKLIST_SYMBOLS='FOOUSDT,BARUSDT'); hc = _cfg_hash(VT_TARGET_DVOL='0.0123')
check('T15d hash ignores BASE_URL / PORT / BLACKLIST_SYMBOLS, reacts to a risk parameter', len(ha) == 16 and ha == hb and ha != hc, f'{ha} {hb} {hc}')

# T18: start() retried on network errors only; watchdog thread is started after start()
import requests
from bybit import ApiError
class _Eq:
    def __init__(self): self.n = 0
    def equity(self):
        self.n += 1
        if self.n <= 2: raise requests.ConnectionError('boom')
        return 10000.0
b18 = Bot.__new__(Bot); eq18 = _Eq(); b18.start = lambda: eq18.equity()
b18.start_with_retry(sleeps=(0, 0))
check('T18a start_with_retry survives 2 ConnectionErrors, 3rd attempt ok', eq18.n == 3)
b18.start = lambda: (_ for _ in ()).throw(SystemExit('FATAL hash'))
try:
    b18.start_with_retry(sleeps=(0, 0)); ex18 = False
except SystemExit:
    ex18 = True
b18.start = lambda: (_ for _ in ()).throw(ApiError(10002, 'x'))
try:
    b18.start_with_retry(sleeps=(0, 0)); ex18b = False
except ApiError:
    ex18b = True
check('T18b SystemExit is not retried; persistent ApiError is raised after the last attempt', ex18 and ex18b)
eq18c = _Eq()
def _st18c():                                       # what the real client raises after its own retries
    try:
        eq18c.equity()
    except requests.ConnectionError as e:
        raise RuntimeError(f'/v5/account/wallet-balance: {e}')
b18.start = _st18c
b18.start_with_retry(sleeps=(0, 0))
check('T18d RuntimeError from Bybit._req (wrapped network error) is retried', eq18c.n == 3)
import main as _main
src = inspect.getsource(_main.main)
check('T18c main(): start_with_retry before the watchdog thread; trading loop in the main thread', 0 <= src.index('start_with_retry') < src.index('watchdog_loop') < src.index('bot.loop_body'))

# T19: quantity rounding is exact (never above the position) and formatted without exponents
from execution import round_qty
from bybit import fmt_qty
cases = [(0.5, 1.5, 1.5), (0.5, 2.6, 2.5), (0.05, 0.15, 0.15), (0.05, 0.35, 0.35), (0.001, 0.0029, 0.002), (5, 12, 10), (1, 12, 12), (0.1, 0.3, 0.3), (0.01, 1.005, 1.0)]
okr = all(abs(round_qty(q, dict(qtyStep=st)) - want) < 1e-12 and round_qty(q, dict(qtyStep=st)) <= q + 1e-12 for st, q, want in cases)
check('T19a round_qty multiples of the step, never above the quantity', okr, str([(st, q, round_qty(q, dict(qtyStep=st))) for st, q, _ in cases]))
check('T19b fmt_qty plain decimals', fmt_qty(0.00001, 1e-5) == '0.00001' and fmt_qty(1.5, 0.5) == '1.5' and fmt_qty(12.0, 1) == '12' and fmt_qty(1e-05) == '0.00001' and fmt_qty(100.0) == '100')

# T20: scale pass sends reducing legs even when the cut is inside the daily band; k_ref moves only after clean execution
class _Api20:
    def equity(self): return 10000.0
    def tickers(self): return {'AAAUSDT': dict(mark=5.0, last=5.0, bid=5.0, ask=5.0, turnover24h=1e8)}
    def positions(self): return {'AAAUSDT': dict(qty=100.0, mark=5.0, entry=5.0, upnl=0, im=1, legs=[])}
def _mk20(errors):
    b = Bot.__new__(Bot); b.api = _Api20(); b.mem = Memory(); b.mem.save = lambda: None; b.risk = RiskEngine(b.mem, lambda *a: None)
    b.lock = threading.Lock(); b.abort_event = threading.Event(); b.instr = {'AAAUSDT': dict(qtyStep=0.1, minQty=0.1, maxQty=1e9, minNotional=5.0, tickSize=0.001)}
    b.mem['last_targets'] = {'AAAUSDT': 500.0}; b.mem['k_ref'] = 2.0
    b.risk.multiplier = lambda eq, now: (1.6, 'test'); b.guards = lambda eq: True
    b.sent = []
    b.ex = types.SimpleNamespace(plan=Executor(None, lambda *a: None, None, True).plan, execute=lambda legs, *a, **k: (b.sent.append(legs) or ({}, errors)))
    return b
import threading
C.REBAL_BAND = 0.30
b20 = _mk20([]); b20.scale_pass()
check('T20a ratio 0.8 (inside the daily band): reduce-only leg sent, k_ref -> 1.6', len(b20.sent) == 1 and b20.sent[0] and all(l[3] for l in b20.sent[0]) and abs(b20.mem['k_ref'] - 1.6) < 1e-9 and abs(b20.mem['last_targets']['AAAUSDT'] - 400.0) < 1e-6, f'{b20.sent} k_ref {b20.mem["k_ref"]}')
b20e = _mk20([('AAAUSDT', 'Sell', 'err')]); b20e.scale_pass()
check('T20b order error: k_ref and last_targets unchanged', b20e.mem['k_ref'] == 2.0 and b20e.mem['last_targets']['AAAUSDT'] == 500.0)

# T21: a held name with a candle-load error is kept, closed on the 3rd consecutive cycle; counter resets on success
b21 = Bot.__new__(Bot); b21.mem = Memory(); pos21 = {'XUSDT': dict(qty=1.0, mark=1.0)}
r1 = b21.bad_data_split(['XUSDT'], pos21, set()); r2 = b21.bad_data_split(['XUSDT'], pos21, set()); r3 = b21.bad_data_split(['XUSDT'], pos21, set())
check('T21a held bad name kept on cycles 1-2, closed (blocked) on the 3rd', r1 == (set(), {'XUSDT'}) and r2 == (set(), {'XUSDT'}) and r3 == ({'XUSDT'}, set()), f'{r1} {r2} {r3}')
b21.bad_data_split([], pos21, set()); r4 = b21.bad_data_split(['XUSDT'], pos21, set())
check('T21b counter resets after a good cycle; not-held and stale names are blocked at once', r4[1] == {'XUSDT'} and b21.bad_data_split(['YUSDT'], pos21, set()) == ({'YUSDT'}, set()) and b21.bad_data_split(['XUSDT'], pos21, {'XUSDT'})[0] == {'XUSDT'})
legs21 = Executor(None, lambda *a: None, None, True).plan({'XUSDT': 1.0 * 1.0 * 1000.0}, {'XUSDT': dict(qty=1000.0, mark=1.0, entry=1, upnl=0, im=1, legs=[])},
        {'XUSDT': dict(mark=1.0, last=1.0, bid=1.0, ask=1.0)}, {'XUSDT': dict(qtyStep=0.1, minQty=0.1, maxQty=1e9, minNotional=5.0, tickSize=0.001)}, 0.3, 15.0)
check('T21c target = current position -> no leg', legs21 == [])

# T22: RESET_* is a one-time key; day roll in guards triggers save()
b22 = Bot.__new__(Bot); b22.mem = Memory(); b22.mem.saves = 0
def _sv(): b22.mem.saves += 1
b22.mem.save = _sv
os.environ['RESET_TOTAL_STOP'] = '1'
b22.mem['halt_total'] = True; b22.apply_resets(); c1 = b22.mem['halt_total']
b22.mem['halt_total'] = True; b22.apply_resets(); c2 = b22.mem['halt_total']
os.environ['RESET_TOTAL_STOP'] = '2'; b22.apply_resets(); c3 = b22.mem['halt_total']
del os.environ['RESET_TOTAL_STOP']
check('T22a RESET_TOTAL_STOP: first value clears, same value again does not, a new value clears', c1 is False and c2 is True and c3 is False)
b22b = Bot.__new__(Bot); b22b.mem = Memory(); b22b.risk = RiskEngine(b22b.mem, lambda *a: None); b22b.mem.saves = 0
def _sv2(): b22b.mem.saves += 1
b22b.mem.save = _sv2
b22b.ex = types.SimpleNamespace(flatten_all=lambda *a, **k: 0); b22b.instr = {}
tday = datetime(2026, 5, 3, 0, 5, tzinfo=timezone.utc); b22b.now = lambda: tday
b22b.mem['day_date'] = '2026-05-02'; b22b.mem['day_anchor'] = 10000.0; b22b.mem['day_peak'] = 10000.0
b22b.guards(10000.0)
check('T22b guards saves the state when the UTC day rolls over', b22b.mem.saves >= 1 and b22b.mem['day_date'] == '2026-05-03', f'saves {b22b.mem.saves}')
import tempfile
_old_sf = C.STATE_FILE; C.STATE_FILE = os.path.join(tempfile.mkdtemp(), 'sub', 'dir', 'state.json')
Memory().check_writable(); okdir = os.path.isdir(os.path.dirname(C.STATE_FILE))
C.STATE_FILE = '/proc/nonexistent/state.json'
try:
    Memory().check_writable(); bad_dir = False
except SystemExit:
    bad_dir = True
C.STATE_FILE = _old_sf
check('T22c state dir is created; an unwritable one aborts the start', okdir and bad_dir)
# T22d: restart round-trip. An old state with nulls for container keys (save() wrote absent PERSIST keys as null) must not
# break the cycle: bad_n = None crashed every rebalance on 2026-10-28 ('NoneType' object is not iterable in bad_data_split)
_old_sf, _old_bf = C.STATE_FILE, C.STATE_BACKUP_FILE; C.STATE_FILE = os.path.join(tempfile.mkdtemp(), 'state.json'); C.STATE_BACKUP_FILE = ''
with open(C.STATE_FILE, 'w') as f:
    json.dump({'created': 1.0, 'bad_n': None, 'ml8_audit': None, 'sleeve_pnl_day': None, 'halted_day': None, 'k_ref': 2.0}, f)
b22d = Bot.__new__(Bot); b22d.mem = Memory(); b22d.mem.load()
r22d = b22d.bad_data_split(['XUSDT'], {'XUSDT': dict(qty=1.0)}, set())
b22d.mem.setdefault('ml8_audit', []).append(1)
m22e = Memory(); m22e.save(); saved22 = json.load(open(C.STATE_FILE))
C.STATE_FILE, C.STATE_BACKUP_FILE = _old_sf, _old_bf
check('T22d state with null containers loads and rebalance helpers work; save() writes no nulls for absent keys',
      r22d == (set(), {'XUSDT'}) and b22d.mem['ml8_audit'] == [1] and 'halted_day' in b22d.mem and b22d.mem['k_ref'] == 2.0
      and 'bad_n' not in saved22 and 'sleeve_pnl_hist' not in saved22 and saved22.get('k_ref', 0) is None, f'{r22d} keys {sorted(saved22)}')

# T23: a name with long Binance age but <30d of bars on the trading exchange is outside the ML universes
S.uni.prev = {}; mm_a = S.ml_universe(panel, age_h); j23 = int(np.flatnonzero(mm_a)[0])
panel_s = dict(panel); da = np.array(panel['data_age'], float); da[j23] = 500.0; panel_s['data_age'] = da
S.uni.prev = {}; mm_b = S.ml_universe(panel_s, age_h); S.uni.prev = {}; m8_b = S.ml8_universe(panel_s, age_h)
check('T23 short Bybit history (500h, age 200d) excluded from ml and ml8 universes', mm_a[j23] and not mm_b[j23] and not m8_b[j23] and S.short_hist_n >= 1, f'short_hist_n {S.short_hist_n}')

# ---------------------------------------------------------------- Phase 3b (T24)
import time
# T24: the watchdog does not wait for a rebalance; a hard limit hit mid-rebalance stops its execution between quote rounds
class _Api24:
    def __init__(self, ev=None): self.sent = []; self.ev = ev
    def cancel_all(self, *a, **k): return None
    def tickers(self): return {'AAAUSDT': dict(mark=10.0, last=10.0, bid=9.99, ask=10.01)}
    def post_only(self, *a, **k):
        self.sent.append(('PO',) + a)
        if self.ev is not None: self.ev.set()        # the limit is hit while the first quote rests
    def market(self, *a, **k): self.sent.append(('MKT',) + a)
meta24 = {'AAAUSDT': dict(minQty=0.1, qtyStep=0.1, maxQty=1e9, tickSize=0.01, minNotional=5.0)}
legs24 = [('AAAUSDT', 'Buy', 500.0, False, False)]
a24 = _Api24(); ex24 = Executor(a24, lambda *a: None, None, False); ex24.abort = threading.Event()
st24, er24 = ex24.execute(legs24, meta24, a24.tickers(), lambda: {}, maker_attempts=2, wait_s=0)
ev24 = threading.Event(); a24b = _Api24(ev24); ex24b = Executor(a24b, lambda *a: None, None, False); ex24b.abort = ev24
st24b, er24b = ex24b.execute(legs24, meta24, a24b.tickers(), lambda: {}, maker_attempts=2, wait_s=0)
check('T24a no abort: 2 quote rounds + market fallback', [x[0] for x in a24.sent] == ['PO', 'PO', 'MKT'] and not st24['aborted'], f'{a24.sent}')
check('T24b abort set after round 1: no further orders, aborted flagged as an error', [x[0] for x in a24b.sent] == ['PO'] and st24b['aborted'] and er24b and er24b[0][0] == '*', f'{a24b.sent} {er24b}')
b24 = Bot.__new__(Bot); b24.mem = Memory(); b24.risk = RiskEngine(b24.mem, lambda *a: None); b24.lock = threading.Lock(); b24.abort_event = threading.Event()
b24.mem['initial'] = 10000.0; b24.mem['eq_peak'] = 10000.0; b24.mem['halt_total'] = False; b24.mem['target_hit'] = False
b24.now = lambda: datetime(2026, 5, 3, 12, 0, tzinfo=timezone.utc)
b24.mem['day_date'] = '2026-05-03'; b24.mem['day_anchor'] = 10000.0; b24.mem['day_peak'] = 10000.0; b24.mem['halted_day'] = None
b24.guards = lambda eq: (_ for _ in ()).throw(AssertionError('guards must not run while the lock is held'))
floor24 = b24.risk.total_floor(); dd24 = b24.risk.dd_amount()
eq_ok = 10000.0; eq_floor = floor24 + 0.5 * C.FLOOR_GUARD_FRAC * dd24
b24.equity = lambda: eq_ok
b24.lock.acquire(); b24.watchdog(); ab_ok = b24.abort_event.is_set()
b24.equity = lambda: eq_floor
t24 = time.time(); b24.watchdog(); dt24 = time.time() - t24; ab_hit = b24.abort_event.is_set(); b24.lock.release()
check('T24c lock held: watchdog returns after ~5 s, sets abort only on a breach', not ab_ok and ab_hit and dt24 < 8 and b24.mem['halt_total'] is False,
      f'ok {ab_ok} hit {ab_hit} {dt24:.1f}s {b24.limit_breach(eq_floor)}')
from state import get_status as _gs32
st32 = _gs32(True)
check('T32 /status has equity and floors between rebalances (watchdog, even while a rebalance holds the lock)',
      st32.get('equity') == round(eq_floor, 2) and st32.get('total_floor') == round(floor24, 2) and 'trading_days' in st32, f'{st32}')
# T33: trading-day marker (Mubite counts a day only with a close >= 5 % of the starting balance)
class _Api33:
    def __init__(self): self.pos = {'AAAUSDT': dict(qty=10.0, mark=100.0, entry=100.0, upnl=0.0)}; self.orders = []
    def positions(self): return {k: dict(v) for k, v in self.pos.items()}
    def tickers(self): return {'BTCUSDT': dict(last=100000.0), 'ETHUSDT': dict(last=4000.0)}
    def market(self, sym, qty, side, reduce_only=False):
        self.orders.append((sym, qty, side, reduce_only))
        if reduce_only: self.pos.pop(sym, None)
        else: px = self.tickers()[sym]['last']; self.pos[sym] = dict(qty=qty if side == 'Buy' else -qty, mark=px, entry=px, upnl=0.0)
_fr = C.DAY_MARKER_FRAC; C.DAY_MARKER_FRAC = 0.06
b33 = Bot.__new__(Bot); b33.mem = Memory(); b33.risk = RiskEngine(b33.mem, lambda *a: None); b33.api = _Api33()
b33.mem['initial'] = 10000.0; b33.mem['eq_peak'] = 10000.0; b33.mem['halt_total'] = False; b33.mem['target_hit'] = False
b33.mem['last_targets'] = {'BTCUSDT': 500.0}; b33.instr = {'BTCUSDT': dict(qtyStep=0.001, minQty=0.001, maxLeverage=100), 'ETHUSDT': dict(qtyStep=0.01, minQty=0.01, maxLeverage=100)}
_t = [datetime(2026, 5, 3, 2, 0, tzinfo=timezone.utc)]; b33.now = lambda: _t[0]
b33.day_marker_step()
o1 = list(b33.api.orders)
_t[0] = datetime(2026, 5, 3, 2, 30, tzinfo=timezone.utc); b33.day_marker_step(); still = 'ETHUSDT' in b33.api.pos     # below the threshold: hold
mg = abs(b33.api.pos['ETHUSDT']['qty']) * 4000.0 / C.SET_LEVERAGE; b33.api.pos['ETHUSDT']['upnl'] = -(C.DAY_MARKER_PNL_MARGIN * mg + 0.01)
_t[0] = datetime(2026, 5, 3, 2, 40, tzinfo=timezone.utc); b33.day_marker_step()
n_before = len(b33.api.orders); b33.day_marker_step()                                                                   # marked: nothing more today
check('T33a marker: symbol outside the book (BTC is a target -> ETH), size >= 6 % of initial, against the book net',
      o1 and o1[0][0] == 'ETHUSDT' and o1[0][1] * 4000.0 >= 600.0 and o1[0][2] == 'Sell', f'{o1}')
check('T33b marker held below the P/L threshold, closed reduce-only above it, day marked once',
      still and b33.api.orders[-1] == ('ETHUSDT', o1[0][1], 'Buy', True) and b33.mem['mk_ok_date'] == '2026-05-03' and len(b33.api.orders) == n_before == 2,
      f'{b33.api.orders} {b33.mem.get("mk_ok_date")}')
_t[0] = datetime(2026, 5, 4, 2, 0, tzinfo=timezone.utc); b33.api.pos.clear(); b33.api.pos['AAAUSDT'] = dict(qty=10.0, mark=100.0, entry=100.0, upnl=0.0)
for _ in range(2):
    b33.day_marker_step(); _t[0] += timedelta(minutes=C.DAY_MARKER_MAX_MIN + 1); b33.day_marker_step()                # time limit, no threshold
check('T33c time limit: closed without counting, retried', b33.mem.get('mk_ok_date') == '2026-05-03' and b33.mem['mk']['n'] == 2 and len(b33.api.orders) == 6,
      f'{b33.api.orders[2:]} {b33.mem["mk"]}')
_t[0] = datetime(2026, 5, 3, 16, 0, tzinfo=timezone.utc); b33.mem['mk'] = {}; b33.mem['mk_ok_date'] = None; n0 = len(b33.api.orders)
b33.day_marker_step()
check('T33d no marker within MAX_MIN+15 min of the next rebalance (17:10)', len(b33.api.orders) == n0)
r33 = RiskEngine(Memory(), lambda *a: None); r33.mem['initial'] = 10000.0; r33.mem['eq_peak'] = 10000.0
r33.mem['day_date'] = '2026-05-03'; r33.mem['day_anchor'] = 10000.0; r33.mem['day_peak'] = 10000.0; r33.mem['mk_ok_date'] = '2026-05-03'
r33.observe_equity(10001.0, datetime(2026, 5, 4, 0, 5, tzinfo=timezone.utc)); td1 = r33.mem['trading_days']
r33.mem['mk_ok_date'] = None; r33.observe_equity(10400.0, datetime(2026, 5, 5, 0, 5, tzinfo=timezone.utc)); td2 = r33.mem['trading_days']
check('T33e marker on: trading day = marked day (tiny P/L counts, a big move without a marker does not)', td1 == 1 and td2 == 1, f'{td1} {td2}')
_t[0] = datetime(2026, 5, 5, 2, 0, tzinfo=timezone.utc); b33.mem['mk'] = {}; n0 = len(b33.api.orders)
b33.mem['trading_days'] = C.RULES['min_days'][0] + C.DAY_MARKER_EXTRA; b33.day_marker_step(); stop_ok = len(b33.api.orders) == n0
b33.mem['trading_days'] -= 1; b33.day_marker_step(); go_ok = len(b33.api.orders) == n0 + 1
check('T33f markers stop once trading days >= firm minimum + DAY_MARKER_EXTRA', stop_ok and go_ok, f'{stop_ok} {go_ok}')
r33.mem['mk_ok_date'] = None; r33.observe_equity(10400.0 + 0.006 * 10000, datetime(2026, 5, 6, 0, 5, tzinfo=timezone.utc))
check('T33g marker on: a profit day counts without a marker', r33.mem['profit_days'] >= 1 and r33.mem['trading_days'] == 1, f'{r33.mem["profit_days"]} {r33.mem["trading_days"]}')
a33 = _Api33(); a33.tickers = lambda: {'BTCUSDT': dict(last=1e5, turnover24h=9e9), 'ETHUSDT': dict(last=4e3, turnover24h=8e9),
                                     'DOGEUSDT': dict(last=0.2, turnover24h=5e8), 'TINYUSDT': dict(last=1.0, turnover24h=1e6), 'XAUTUSDT': dict(last=4e3, turnover24h=9e8)}
b33.api = a33; b33.mem['trading_days'] = 0; b33.mem['mk'] = {}; b33.mem['last_targets'] = {'BTCUSDT': 1.0, 'ETHUSDT': 1.0}
b33.instr.update({'DOGEUSDT': dict(qtyStep=1.0, minQty=1.0, maxLeverage=50), 'TINYUSDT': dict(qtyStep=1.0, minQty=1.0, maxLeverage=50),
                  'XAUTUSDT': dict(qtyStep=0.01, minQty=0.01, maxLeverage=50)})
_t[0] = datetime(2026, 5, 7, 2, 0, tzinfo=timezone.utc); b33.day_marker_step()
check('T33h preferred symbols all in the book -> most liquid free symbol (not blacklisted, turnover >= 20M)',
      a33.orders and a33.orders[0][0] == 'DOGEUSDT' and a33.orders[0][1] * 0.2 >= 600.0, f'{a33.orders}')
C.DAY_MARKER_FRAC = _fr
dl24 = -1.01 * C.DAILY_HALT_FRAC * C.RULES['daily_dd'] / 100.0 * 10000.0
check('T24d limit_breach: a day loss past the daily halt is detected read-only', b24.limit_breach(10000.0 + dl24) is not None, str(b24.limit_breach(10000.0 + dl24)))
check('T24e limit_breach healthy equity -> None', b24.limit_breach(10000.0) is None)


# T25: ML feature parity — the training path (exp21c_ml_noage.py formulas on the research Data arrays, one decision bar)
# vs the live compute_features on the bot panel at the same bar; then the production h3/h7 models on both inputs.
# premz is excluded: the training std window is fixed at the first decision day (known, Phase 4: one premz formula).
from ml_features import compute_features, rank_norm, FEATURES
from scipy.stats import spearmanr
def _research_feats(d, i, jb):
    dec = np.array([i]); cff = d.cff; t24 = d.t24; fund = d.fund; prem = d.prem; high = d.high; low = d.low
    rh = lambda k: np.log(cff[dec] / cff[dec - k])[0]
    win = lambda a, k, f: f(a[i - k + 1:i + 1], 0)
    F = {f'r{k}': rh(k) for k in (24, 72, 168, 336, 720, 1440, 2160)}
    F['r720_skip24'] = np.log(cff[i - 24] / cff[i - 720]); F['r168_skip24'] = np.log(cff[i - 24] / cff[i - 168])
    F['rel168'] = F['r168'] - np.log(cff[i, jb] / cff[i - 168, jb]); F['rel720'] = F['r720'] - np.log(cff[i, jb] / cff[i - 720, jb])
    F['vol168'] = win(d.ret, 168, np.nanstd); F['vol720'] = win(d.ret, 720, np.nanstd); F['volratio'] = F['vol168'] / F['vol720']
    for k in (24, 72, 168, 720): F[f'fund{k}'] = np.nansum(fund[i - k + 1:i + 1], 0, dtype=np.float64)
    for k in (8, 24, 168): F[f'prem{k}'] = np.nanmean(prem[i - k + 1:i + 1], 0, dtype=np.float64)
    F['logturn'] = np.log(np.where(t24[i] > 0, t24[i], np.nan)); F['turnratio'] = np.log(t24[i] / np.nanmean(t24[i - 719:i + 1], 0, dtype=np.float64))
    F['hl168'] = np.log(win(high, 168, np.nanmax) / win(low, 168, np.nanmin))
    F['dist_hi720'] = np.log(cff[i] / win(high, 720, np.nanmax)); F['dist_lo720'] = np.log(cff[i] / win(low, 720, np.nanmin))
    R = d.ret[i - 719:i + 1]; rb = R[:, jb]; ok = np.isfinite(rb); Rm = np.where(np.isfinite(R[ok]), R[ok], 0.0); rbb = rb[ok]
    F['beta720'] = ((Rm - Rm.mean(0)) * (rbb - rbb.mean())[:, None]).mean(0) / rbb.var()
    return F
jb25 = syms.index('BTCUSDT')
Fr = _research_feats(d, i, jb25)
Fl = compute_features(panel['cff'][-2161:], np.where(np.isfinite(panel['fund']), panel['fund'], 0.0)[-2161:], panel['t24'][-2161:], panel['prem'][-2161:],
                      panel['high'][-2161:], panel['low'][-2161:], age_h, jb25)
S.uni.prev = {}; mu25 = S.ml_universe(panel, age_h); ix25 = np.flatnonzero(mu25)
sp25 = {}
for f in Fr:
    a, b = Fr[f][ix25], Fl[f][ix25]; ok = np.isfinite(a) & np.isfinite(b)
    sp25[f] = spearmanr(a[ok], b[ok]).correlation if ok.sum() > 10 else np.nan
bad25 = {f: round(v, 4) for f, v in sp25.items() if not v >= 0.99}
check('T25a ML features: training formulas == live compute_features (Spearman >= 0.99 on the ML universe, premz excluded)', not bad25 and len(ix25) >= 30,
      f'n {len(ix25)} min {min(sp25.values()):.4f} worse: {bad25}')
if S.models.get(3) is not None:
    feats25 = [f for f in FEATURES if f not in C.ML_DROP_FEATS]
    Fr2 = dict(Fr, premz=Fl['premz'], age_d=Fl['age_d'])
    Xr = rank_norm(np.column_stack([Fr2[f][ix25] for f in feats25]).astype(np.float32)).astype(np.float64)
    Xl = rank_norm(np.column_stack([Fl[f][ix25] for f in feats25]).astype(np.float32)).astype(np.float64)
    sp_pred = {hz: spearmanr(S.models[hz].predict(Xr), S.models[hz].predict(Xl)).correlation for hz in (3, 7) if S.models.get(hz) is not None}
    check('T25b h3/h7 predictions: training-path inputs vs live inputs, Spearman >= 0.99', all(v >= 0.99 for v in sp_pred.values()),
          ' '.join(f'h{k} {v:.4f}' for k, v in sp_pred.items()))
else:
    print('SKIP  T25b (no ML models)')

# T25c: no ml8 feature may be constant across symbols (Phase 4.3(ii)): after rank_norm a constant column becomes the
# symbol's position in the column order (alphabetical in training, by turnover live). The order-permutation Spearman of
# the predictions is printed for information: ties (e.g. equal funding) still make rank_norm order-dependent (Phase 4.4).
if S.models.get(8) is not None:
    from ml_features_8h import compute_features_8h, FEATURES_8H, CONST_8H
    F8 = compute_features_8h(panel['cff'][-2161:], panel['fund'][-2161:], panel['t24'][-2161:], panel['prem'][-2161:],
                             panel['high'][-2161:], panel['low'][-2161:], age_h, jb25, int((d.hh[i] + 1) % 24))
    ix8 = ix25   # ML universe (the ml8 one needs age >= 180d, longer than this test's data window)
    X8 = np.column_stack([F8[f][ix8] for f in FEATURES_8H]).astype(np.float32)
    # age_d skipped: the test's age is counted from the first loaded bar, so on a short panel every symbol can share it;
    # live passes the real (Binance-clock) listing age
    const8 = [f for k, f in enumerate(FEATURES_8H) if f != 'age_d' and len(np.unique(X8[np.isfinite(X8[:, k]), k])) < 3]
    perm = np.random.default_rng(25).permutation(len(ix8)); inv = np.argsort(perm)
    p8a = S.models[8].predict(rank_norm(X8).astype(np.float64)); p8b = S.models[8].predict(rank_norm(X8[perm]).astype(np.float64))[inv]
    check('T25c ml8 features: none constant across symbols (CONST_8H excluded)', not const8 and not set(CONST_8H) & set(FEATURES_8H) and len(ix8) >= 30,
          f'n {len(ix8)} constant {const8} | info: symbol-order permutation Spearman {spearmanr(p8a, p8b).correlation:.4f}')
else:
    print('SKIP  T25c (no ml8 model)')

# T26: ML sleeves do not depend on the panel's symbol order (Phase 4.4): live orders symbols by turnover, training
# alphabetically; rank_norm breaks ties by row order, so the bot sorts the universe by name before ranking.
if S.models:
    perm26 = np.random.default_rng(26).permutation(len(syms)); inv26 = np.argsort(perm26)
    pp = dict(panel, symbols=[syms[j] for j in perm26], **{k: v[:, perm26] for k, v in panel.items() if isinstance(v, np.ndarray) and v.ndim == 2})
    hr26 = int((d.hh[i] + 1) % 24); d26 = {}
    for hz in sorted(S.models):
        S.uni.prev = {}
        wa = S.ml8(panel, mu25, age_h, hr26) if hz == 8 else S.ml(panel, mu25, age_h, hz)
        wb = S.ml8(pp, mu25[perm26], age_h[perm26], hr26) if hz == 8 else S.ml(pp, mu25[perm26], age_h[perm26], hz)
        d26[hz] = (float(np.abs(wa - wb[inv26]).max()), int((wa != 0).sum()))
    check('T26 ML weights invariant to symbol order', all(v[0] < 1e-12 and v[1] > 0 for v in d26.values()),
          ' '.join(f'h{k} maxdiff {v[0]:.1e} n {v[1]}' for k, v in d26.items()))
else:
    print('SKIP  T26 (no ML models)')

# T27: V2 (HIVOL_MULT, Phase R2): off = identity and registered hash unchanged; on = research v2() (top-vol third x mult, side sums kept)
rng27 = np.random.default_rng(27); vol27 = S.ml8_vol(panel); w27 = np.zeros(len(syms)); ok27 = np.flatnonzero(np.isfinite(vol27))
pick27 = rng27.choice(ok27, size=min(60, len(ok27)), replace=False); half = len(pick27) // 2
w27[pick27[:half]] = rng27.random(half); w27[pick27[half:]] = -rng27.random(len(pick27) - half); w27 /= np.abs(w27).sum() / 2
C0 = C.HIVOL_MULT
try:
    C.HIVOL_MULT = 1.0; off27 = S.damp_hivol(w27, panel, {})
    C.HIVOL_MULT = 0.5; inf27 = {}; on27 = S.damp_hivol(w27, panel, inf27)
finally:
    C.HIVOL_MULT = C0
nz27 = np.flatnonzero(w27 != 0); hv27 = nz27[np.argsort(vol27[nz27])[int(len(nz27) * 2 / 3):]]; ref27 = w27.copy(); ref27[hv27] *= 0.5
ref27[ref27 > 0] *= w27[w27 > 0].sum() / ref27[ref27 > 0].sum(); ref27[ref27 < 0] *= w27[w27 < 0].sum() / ref27[ref27 < 0].sum()
h_off = _cfg_hash(HIVOL_MULT='1.0'); h_on = _cfg_hash(HIVOL_MULT='0.5')
check('T27a HIVOL_MULT=1 is identity; =0.5 equals research v2 (side sums kept, top-vol third down-weighted)',
      np.array_equal(off27, w27) and np.allclose(on27, ref27) and abs(on27[on27 > 0].sum() - w27[w27 > 0].sum()) < 1e-12 and abs(on27[hv27]).sum() < abs(w27[hv27]).sum() and inf27.get('hivol_n') == len(hv27),
      f'hv {len(hv27)}')
check('T27b hash: HIVOL_MULT off = baseline hash, on = new hash', h_off == ha and h_on != ha and len(h_on) == 16, f'{ha} {h_off} {h_on}')

# T28: R3-bis P2 (K_MIN_FRAC: K floor above the floor guard) and P3 (drawdown mode: top-vol third x DD_HIVOL_MULT when the buffer is thin)
mem28 = Memory(); mem28.save = lambda: None; R28 = RiskEngine(mem28, log=lambda *a: None)
t28 = datetime(2025, 9, 1, 0, 10, tzinfo=timezone.utc); R28.observe_equity(10000.0, t28); t28b = t28 + timedelta(days=1)
dd28 = R28.dd_amount(); eq_thin = R28.total_floor() + 0.10 * dd28; eq_wide = R28.total_floor() + 0.80 * dd28
R28.observe_equity(eq_thin, t28b)
K0, B0, G0 = C.K_MIN_FRAC, C.DD_HIVOL_BUF, C.DD_HIVOL_MULT
try:
    C.K_MIN_FRAC = 0.0; k_off28, _ = R28.multiplier(eq_thin, t28b)
    C.K_MIN_FRAC = 0.3; k_on28, _ = R28.multiplier(eq_thin, t28b); k_guard28, _ = R28.multiplier(R28.total_floor() + 0.5 * C.FLOOR_GUARD_FRAC * dd28, t28b)
    C.DD_HIVOL_BUF = 0.0; dm_off28 = R28.dd_mode(eq_thin)
    C.DD_HIVOL_BUF = 0.5; dm_thin28, dm_wide28 = R28.dd_mode(eq_thin), R28.dd_mode(eq_wide)
    C.DD_HIVOL_MULT = 0.0; S.hivol_mult = C.DD_HIVOL_MULT; inf28 = {}; on28 = S.damp_hivol(w27, panel, inf28)
finally:
    C.K_MIN_FRAC, C.DD_HIVOL_BUF, C.DD_HIVOL_MULT = K0, B0, G0; S.hivol_mult = None
ref28 = w27.copy(); ref28[hv27] = 0.0
ref28[ref28 > 0] *= w27[w27 > 0].sum() / ref28[ref28 > 0].sum(); ref28[ref28 < 0] *= w27[w27 < 0].sum() / ref28[ref28 < 0].sum()
check('T28a K_MIN_FRAC: off = CPPI, 0.3 = K >= 0.3 x base above the guard, guard still K=0',
      abs(k_off28 - C.BASE_SCALE * 0.10 / C.CPPI_FRAC) < 1e-9 and abs(k_on28 - 0.3 * C.BASE_SCALE) < 1e-9 and k_guard28 == 0.0, f'{k_off28:.3f} {k_on28:.3f} {k_guard28}')
check('T28b drawdown mode: off by default, on below DD_HIVOL_BUF x allowance, override zeroes the top-vol third (side sums kept)',
      not dm_off28 and dm_thin28 and not dm_wide28 and np.allclose(on28, ref28) and inf28.get('hivol_mult') == 0.0 and S.hivol_mult is None)
h28 = [_cfg_hash(K_MIN_FRAC='0'), _cfg_hash(DD_HIVOL_MULT='0.3'), _cfg_hash(K_MIN_FRAC='0.3'), _cfg_hash(DD_HIVOL_BUF='0.5')]
check('T28c hash: K_MIN_FRAC=0 and DD_HIVOL_* with DD_HIVOL_BUF=0 = baseline; turning either on changes it',
      h28[0] == ha and h28[1] == ha and h28[2] != ha and h28[3] != ha and h28[2] != h28[3], ' '.join(h28))

# T29: Mubite profitable days (3 + 2): counted at the day roll when the day's PnL >= PROFIT_DAY_PCT % of initial; preset values
mem29 = Memory(); mem29.save = lambda: None; R29 = RiskEngine(mem29, log=lambda *a: None)
t29 = datetime(2025, 9, 1, 0, 10, tzinfo=timezone.utc); R29.observe_equity(10000.0, t29)
for k_, e_ in enumerate((10060.0, 10070.0, 10200.0, 10180.0)):          # day pnl +60 (+0.6 %), +10, +130 (+1.3 %), -20
    R29.observe_equity(e_, t29 + timedelta(hours=12 + 24 * k_)); R29.observe_equity(e_, t29 + timedelta(days=k_ + 1))
check('T29 profitable days: +0.6 % and +1.3 % days counted, +0.1 % / loss days not; Mubite presets require 3 + 2',
      mem29['profit_days'] == 2 and C.FIRM_PRESETS['mubite_8_dd10']['profit_days'] == [3, 2] and 'profit_days' in Memory.PERSIST,
      f"profit_days {mem29['profit_days']} trading_days {mem29['trading_days']}")

# T30: payout pause (PAUSE_FILE): flat + any equity change booked as a cash flow (daily anchor moves, no DAILY HALT); transfers while trading
import tempfile as _tf30
_pf = os.path.join(_tf30.mkdtemp(), 'state.json.pause'); _old_pf = C.PAUSE_FILE; C.PAUSE_FILE = _pf
class _Api30:
    def __init__(s): s.eq = 10300.0; s.pos = {'BTCUSDT': {'qty': 1}}; s.tr = (0.0, 0)
    def equity(s): return s.eq
    def positions(s): return s.pos
    def transfers_since(s, ms): return s.tr
b30 = Bot.__new__(Bot); b30.mem = Memory(); b30.mem.save = lambda: None; b30.risk = RiskEngine(b30.mem, lambda *a: None); b30.api = _Api30()
b30.instr = {}; b30.ex = types.SimpleNamespace(flatten_all=lambda *a, **k: b30.api.pos.clear())
t30 = datetime(2026, 5, 3, 10, 0, tzinfo=timezone.utc); b30.now = lambda: t30
b30.mem['day_date'] = '2026-05-03'; b30.mem['day_anchor'] = 10250.0; b30.mem['day_peak'] = 10300.0
open(_pf, 'w').close()
r1 = b30.guards(10300.0)                       # paused: closes positions
r2 = b30.guards(10300.0)                       # flat: remembers equity 10300
b30.api.eq = 10050.0                           # payout of 250 processed by the firm
r3 = b30.guards(10050.0)                       # booked as a cash flow
os.remove(_pf); r4 = b30.guards(10050.0)       # resumed: trades again, no daily halt
anchor_ok = abs(b30.mem['day_anchor'] - 10000.0) < 1e-6 and b30.mem.get('halted_day') is None
b30.mem['transfer_chk_ms'] = 0; b30.api.tr = (-500.0, int(time.time() * 1000)); b30.api.eq = 9550.0
r5 = b30.guards(9550.0)                        # own-account style withdrawal while trading, from the transfer log
C.PAUSE_FILE = _old_pf
check('T30 payout pause: flatten -> flat equity remembered -> payout booked as cash flow (anchor -250, no halt) -> resume; transfer-log withdrawal booked',
      (r1, r2, r3, r4) == (False, False, False, True) and not b30.api.pos and anchor_ok and r5 is True and abs(b30.mem['day_anchor'] - 9500.0) < 1e-6
      and abs(b30.mem['flow_today'] + 750.0) < 1e-6, f"anchor {b30.mem['day_anchor']} flow {b30.mem['flow_today']} halted {b30.mem.get('halted_day')}")

# T31: own account (trailing_pct): the drawdown allowance is 25 % of the equity PEAK (grows with the account), withdrawals move the base
_rules31 = dict(C.RULES); _firm31 = C.FIRM
C.RULES = dict(C.RULES, total_dd=25.0, total_dd_mode='trailing_pct'); C.FIRM = 'personal'
try:
    m31 = Memory(); m31.save = lambda: None; R31 = RiskEngine(m31, lambda *a: None); m31['initial'] = 5000.0; m31['eq_peak'] = 5000.0
    R31.observe_equity(15000.0, datetime(2026, 1, 1, 1, tzinfo=timezone.utc)); f1 = R31.total_floor(); d1 = R31.dd_amount()
    R31.note_cash_flow(-3000.0); f2 = R31.total_floor()
finally:
    C.RULES = _rules31; C.FIRM = _firm31
check('T31 own account: floor 25 % below the peak (15000 -> 11250), a 3000 withdrawal lowers peak and floor (12000 -> 9000)',
      abs(f1 - 11250.0) < 1e-6 and abs(d1 - 3750.0) < 1e-6 and abs(f2 - 9000.0) < 1e-6 and abs(m31['initial'] - 2000.0) < 1e-6, f'{f1} {d1} {f2}')

# ---------------------------------------------------------------- PREREG reg. 4 (audit 2026-10-31): T34-T37
# T34: funding-sleeve ties. (a) bot quantile_ls_tb == research copy; (b) last_rate; (c) inside a tie the higher latest rate is
# shorted first, then the more liquid name; (d) weights do not depend on the panel's symbol order; (e) FUND_TIEBREAK=0 = old order
import signals as _sig
rng34 = np.random.default_rng(34); n34 = 300
sig34 = np.round(rng34.integers(-3, 4, n34) * 1e-4 + rng34.standard_normal(n34) * (rng34.random(n34) < 0.3) * 1e-4, 8); sig34[::11] = np.nan
sec34 = rng34.integers(-2, 3, n34) * 5e-5; liq34 = rng34.random(n34) * 1e8
check('T34a quantile_ls_tb bot == research', np.array_equal(_sig.quantile_ls_tb(sig34, sec34, liq34, 0.2), qtb_res(sig34, sec34, liq34, 0.2)))
fw34 = np.zeros((72, 4)); fw34[7, 0] = 1e-4; fw34[63, 0] = 3e-4; fw34[15, 1] = -2e-4; fw34[71, 3] = 5e-5
check('T34b last_rate = latest non-zero settlement (0 if none); bot == research',
      np.allclose(_sig.last_rate(fw34), [3e-4, -2e-4, 0.0, 5e-5]) and np.array_equal(_sig.last_rate(fw34), lr_res(fw34)))
s34 = np.zeros(20); s34[:4] = [-9e-4, -8e-4, 9e-4, 8e-4]                    # 16 names tie at 0; k = 4 per leg
sec_c = np.zeros(20); sec_c[4] = -1e-4                                   # name 4: highest latest rate (sec = -rate is lowest)
liq_c = np.arange(20, dtype=float) * 1e6                                 # liquidity rises with the index
w34 = qtb_res(s34 + 1e-17 * rng34.standard_normal(20), sec_c, liq_c, 0.2)  # float noise below 1e-9 must not matter
check('T34c tie order: signal, then latest rate, then turnover (more liquid first in both legs)',
      set(np.flatnonzero(w34 < 0)) == {0, 1, 4, 19} and set(np.flatnonzero(w34 > 0)) == {2, 3, 18, 17} and abs(w34.sum()) < 1e-12,
      f'shorts {sorted(np.flatnonzero(w34 < 0))} longs {sorted(np.flatnonzero(w34 > 0))}')
perm34 = np.random.default_rng(341).permutation(len(syms)); inv34 = np.argsort(perm34)
pp34 = dict(panel, symbols=[syms[j] for j in perm34], **{k: v[:, perm34] for k, v in panel.items() if isinstance(v, np.ndarray) and v.ndim == 2})
wa34 = S.fund_carry(panel, mu); wb34 = S.fund_carry(pp34, mu[perm34]); wc34 = S.fund_change(panel, mu); wd34 = S.fund_change(pp34, mu[perm34])
check('T34d funding sleeves invariant to symbol order', np.abs(wa34 - wb34[inv34]).max() < 1e-12 and np.abs(wc34 - wd34[inv34]).max() < 1e-12 and (wa34 != 0).sum() > 0,
      f'fcarry n {int((wa34 != 0).sum())}')
C.FUND_TIEBREAK = 0
w_old34 = S.fund_carry(panel, mu)
C.FUND_TIEBREAK = 1
check('T34e FUND_TIEBREAK=0 reproduces the argsort order (before reg. 4)', np.allclose(w_old34, q_res(np.where(mu_res & np.isfinite(sig_lvl), sig_lvl, np.nan), 0.2), atol=1e-12))

# T35: ml8 features ranked over the TRAINING universe (age >= 30 d, top-150, exp17), sleeve trades the >= 180 d names (reg. 4).
# The test window is ~6 months, so ages are shifted by 150 days to populate the 180 d set; research universes rebuilt with the same ages.
if 8 in S.models:
    from ml_features_8h import compute_features_8h as _cf8, FEATURES_8H as _F8
    from ml_features import rank_norm as _rn
    age35 = age_h + 150 * 24.0
    def _runi(min_age_h):
        t_ = d.t24[i]; m_ = (age35 >= min_age_h) & d.valid[i] & np.isfinite(t_) & (t_ >= 1e7)
        o_ = np.argsort(-np.where(m_, t_, -np.inf)); r_ = np.empty(N, dtype=np.int64); r_[o_] = np.arange(N)
        return m_ & (r_ < 150)
    u30, ub35 = _runi(720), _runi(180 * 24)
    panel35 = dict(panel, data_age=age35)          # bars on the exchange as old as the shifted age (as in research: segment age = bars)
    S.uni.prev = {}; m8t = S.ml8_universe(panel35, age35); m8r = S.ml8_rank_universe(panel35, age35)
    nh = 2161; jb35 = syms.index('BTCUSDT')
    F35 = _cf8(d.cff[i - nh + 1:i + 1], d.fund[i - nh + 1:i + 1], d.t24[i - nh + 1:i + 1], d.prem[i - nh + 1:i + 1], d.high[i - nh + 1:i + 1], d.low[i - nh + 1:i + 1], age35, jb35, 0)
    ix35 = np.flatnonzero(u30); p30 = np.full(N, np.nan)
    p30[ix35] = S.models[8].predict(_rn(np.column_stack([F35[f][ix35] for f in _F8]).astype(np.float32)).astype(np.float64))
    w_res35 = q_res(np.where(ub35, p30, np.nan), C.ML8_TOP_FRAC, inv_vol=vol_res)
    w_bot35 = S.ml8(panel35, m8t, age35, 0, rank_mask=m8r)
    w_old35 = S.ml8(panel35, m8t, age35, 0)
    check('T35a ml8 rank universe == research training universe (age>=30d top-150), trade set == age>=180d', (m8r == u30).all() and (m8t == ub35).all()
          and ub35.sum() > 20 and (u30 != ub35).any(), f"differ {int((u30 != ub35).sum())} "
          f"rank {int(m8r.sum())}/{int(u30.sum())} trade {int(m8t.sum())}/{int(ub35.sum())}")
    check('T35b ml8 weights == research (rank over 30d, trade >= 180d)', np.abs(w_bot35 - w_res35).max() < 1e-9 and (w_bot35 != 0).sum() > 10,
          f'n {int((w_bot35 != 0).sum())} max diff {np.abs(w_bot35 - w_res35).max():.1e}; old path (rank over 180d) 0.5*L1 {0.5 * np.abs(w_old35 - w_res35).sum():.3f}')
    _sl35 = C.SLEEVES
    C.SLEEVES = ['ml8']; S.uni.prev = {}
    _w, _p, inf35 = S.combined(panel35, age35, set(), daily_update=True, hour_utc=0)
    C.SLEEVES = _sl35
    check('T35c combined() ranks ml8 over the rank universe', inf35.get('ml8_rank_uni') == int(u30.sum()) and np.abs(_p['ml8'] - w_res35).max() < 1e-9, f"{inf35.get('ml8_rank_uni')}")
else:
    print('SKIP  T35 (no ml8 model)')

# T36: EMA applied once per rebalance slot: a retried slot (aborted / failed execution) gives the same targets as its first attempt
m36 = Memory(); m36.save = lambda: None; R36 = RiskEngine(m36, lambda *a: None); m36['lev'] = 1.0; m36['initial'] = 10000.0
sy36 = ['AUSDT', 'BUSDT', 'CUSDT', 'DUSDT']; w36 = np.array([0.01, -0.01, 0.0, 0.0]); m36['w_prev'] = {'CUSDT': 0.01, 'DUSDT': -0.01}
t1, _ = R36.build_targets(w36, sy36, 10000.0, np.full(4, 5e8), 1.0, set(), slot='2026-11-01T01')
t2, _ = R36.build_targets(w36, sy36, 10000.0, np.full(4, 5e8), 1.0, set(), slot='2026-11-01T01')
t3, _ = R36.build_targets(w36, sy36, 10000.0, np.full(4, 5e8), 1.0, set(), slot='2026-11-01T09')
check('T36 EMA once per slot: retry == first attempt (0.5/0.5), next slot smooths from it',
      np.allclose(t1, [50, -50, 50, -50]) and np.allclose(t1, t2) and np.allclose(t3 / t1, [1.5, 1.5, 0.5, 0.5]),
      f't1 {t1} t2 {t2} t3 {t3}')

# T37: ml8 kill-rule memory carried over from the previous stage's state file (ops/stage.sh -> ML8_CARRY_FROM), only those keys
tmp37 = tempfile.mkdtemp(); old37 = os.path.join(tmp37, 'trial_100000.json')
hist37 = [{'date': f'd{k_}', 'eq': 100000.0, 'ml8': 1.0, 'ml8_shadow': 3.0 if k_ % 2 else 1.0, 'ml8_shadow_cost': 0.1} for k_ in range(190)]
json.dump(dict(sleeve_pnl_hist=hist37, ml8_mult=0.5, ml8_mult_date='2026-10-01', ml8_changes=[['2026-10-01', 1.0, 0.5, 0.3]], initial=100000.0,
               eq_peak=101000.0, w_prev={'AUSDT': 0.1}, sleeve_pnl_day={'ml8': 5.0}, attrib_shadow_ml8={'AUSDT': 10.0}), open(old37, 'w'))
b37 = Bot.__new__(Bot); b37.mem = Memory(); b37.mem.save = lambda: None
_cf37 = C.ML8_CARRY_FROM; C.ML8_CARRY_FROM = old37
n37 = b37.carry_ml8_memory(); C.ML8_CARRY_FROM = _cf37
mm37 = b37.mem
check('T37a kill-rule memory carried (history, multiplier, dates), nothing else', n37 == 190 and len(mm37['sleeve_pnl_hist']) == 190 and mm37['ml8_mult'] == 0.5
      and mm37['ml8_mult_date'] == '2026-10-01' and 'initial' not in mm37 and 'eq_peak' not in mm37 and not mm37.get('w_prev') and 'sleeve_pnl_day' not in mm37
      and 'attrib_shadow_ml8' not in mm37, f'n {n37} keys {sorted(k for k in mm37 if not k.startswith("c"))}')
b37.reb_every_base = 8; C.ML8_KILL_RULE = 1
b37.apply_ml8_kill_rule(10000.0, datetime(2026, 11, 15, tzinfo=timezone.utc))
check('T37b the carried rule acts at once (>= 180 days): shadow net Sharpe > 1 but restore waits 60 d -> stays 0.5',
      mm37['ml8_mult'] == 0.5 and abs(C.SLEEVE_SCALES['ml8'] - C.ML8_BASE_SCALE * 0.5) < 1e-12, f"mult {mm37['ml8_mult']}")
C.SLEEVE_SCALES['ml8'] = C.ML8_BASE_SCALE; C.REBAL_EVERY_H = 8
check('T37c ML8_CARRY_FROM is not part of the config hash', _cfg_hash() == _cfg_hash(ML8_CARRY_FROM='/opt/propsleeves/state/x.json'))

print(f'\n{sum(1 for _, ok in res if ok)}/{len(res)} passed')
sys.exit(0 if all(ok for _, ok in res) else 1)
