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
C.EQ_CONFIRM_S = 0.0          # F2 confirmation wait: no real sleeping in tests (the F2 tests set their own value)
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
import subprocess, inspect, types, shutil
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

# ---------------------------------------------------------------- Reg. 5: partial rebalance (REBAL_PARTIAL), fchg off
# T38: Executor.plan(frac, start): partial step = frac x gap from the slot-start position; full close stays full; band by the full gap
ex38 = Executor(None, lambda *a: None, None, True)
i38 = {s: dict(qtyStep=0.1, minQty=0.1, maxQty=1e9, minNotional=5.0, tickSize=0.001) for s in ('AUSDT', 'BUSDT', 'CUSDT', 'DUSDT', 'EUSDT')}
tk38 = {s: dict(mark=1.0, last=1.0, bid=1.0, ask=1.0) for s in i38}
def pos38(d_): return {s: dict(qty=q, mark=1.0, entry=1.0, upnl=0, im=1, legs=[]) for s, q in d_.items() if q != 0}
def net38(legs): return {l[0]: (1 if l[1] == 'Buy' else -1) * l[2] for l in legs}
p0 = {'AUSDT': 100.0, 'BUSDT': -100.0, 'CUSDT': 200.0, 'DUSDT': 10.0}
tg38 = {'AUSDT': 300.0, 'BUSDT': -100.0, 'CUSDT': 0.0, 'EUSDT': 400.0, 'DUSDT': -190.0}
L38 = ex38.plan(tg38, pos38(p0), tk38, i38, 0.5, 15.0, frac=0.5, start=dict(p0))
n38 = net38([l for l in L38 if not (l[0] == 'DUSDT' and l[4])])
c38 = [l for l in L38 if l[0] == 'CUSDT']
check('T38a partial: half of the gap (A +100 of 200, E +200 of 400), unchanged name skipped, full close of C stays full (200)',
      abs(n38['AUSDT'] - 100) < 1e-9 and abs(n38['EUSDT'] - 200) < 1e-9 and 'BUSDT' not in n38 and len(c38) == 1 and abs(c38[0][2] - 200) < 1e-9 and c38[0][4], str(L38))
d38 = [l for l in L38 if l[0] == 'DUSDT']
check('T38b flip D +10 -> -190 is a partial step to -90: close 10 + open 90 (same arithmetic as bt.run trade_frac)',
      len(d38) == 2 and d38[0][4] and abs(d38[0][2] - 10) < 1e-9 and abs(d38[1][2] - 90) < 1e-9 and d38[1][1] == 'Sell', str(d38))
L38f = ex38.plan(tg38, pos38(p0), tk38, i38, 0.5, 15.0)
check('T38c frac=1.0 (default) is the unchanged full step', abs(net38([l for l in L38f if l[0] == 'AUSDT'])['AUSDT'] - 200) < 1e-9 and ex38.plan(tg38, pos38(p0), tk38, i38, 0.5, 15.0, frac=1.0, start=dict(p0)) == L38f)
# band by the FULL gap: gap 40 on target 100 (< 0.5 x 100 = 50) -> no trade; gap 60 -> trade of 30
check('T38d band is decided on the full gap, the step is half of it',
      ex38.plan({'AUSDT': 100.0}, pos38({'AUSDT': 60.0}), tk38, i38, 0.5, 15.0, frac=0.5, start={'AUSDT': 60.0}) == []
      and abs(net38(ex38.plan({'AUSDT': 100.0}, pos38({'AUSDT': 40.0}), tk38, i38, 0.5, 15.0, frac=0.5, start={'AUSDT': 40.0}))['AUSDT'] - 30) < 1e-9)
# T38e: idempotency. attempt 1 fills only A (E fails); the retry of the SAME slot (start fixed) completes E's half step and does NOT step A again;
# without `start` (position-based) the retry would walk A to 75 % of the way.
cur38 = dict(p0); first = ex38.plan(tg38, pos38(cur38), tk38, i38, 0.2, 15.0, frac=0.5, start=dict(p0))
cur38['AUSDT'] += net38(first)['AUSDT']
retry = ex38.plan(tg38, pos38(cur38), tk38, i38, 0.2, 15.0, frac=0.5, start=dict(p0))
naive = ex38.plan(tg38, pos38(cur38), tk38, i38, 0.2, 15.0, frac=0.5)
check('T38e retry of the same slot: A already at its step (no new leg), E still 200; position-based retry would step A again (+50)',
      'AUSDT' not in net38(retry) and abs(net38(retry)['EUSDT'] - 200) < 1e-9 and abs(net38(naive).get('AUSDT', 0) - 50) < 1e-9, f'retry {net38(retry)} naive {net38(naive)}')
# T38f: the whole chain in Bot.rebalance state handling: slot start positions are fixed on the first attempt of a slot only
src38 = inspect.getsource(Bot.rebalance) if hasattr(Bot, 'rebalance') else ''
check('T38f main.py fixes the slot-start positions once per slot and passes frac/start to plan', "slot_pos0_slot') != slot_key" in src38 and 'frac=C.REBAL_PARTIAL' in src38 and 'start=start' in src38)

# T39: EMA once per slot + partial step compose: retry of a slot gives the same smoothed targets AND the same step orders as the first attempt
m39 = Memory(); m39.save = lambda: None; R39 = RiskEngine(m39, lambda *a: None); m39['lev'] = 1.0; m39['initial'] = 10000.0
sy39 = ['AUSDT', 'BUSDT', 'CUSDT', 'DUSDT']; w39 = np.array([0.01, -0.01, 0.0, 0.0]); m39['w_prev'] = {'CUSDT': 0.01, 'DUSDT': -0.01}
a39, _ = R39.build_targets(w39, sy39, 10000.0, np.full(4, 5e8), 1.0, set(), slot='S1')
pstart39 = {'AUSDT': 0.0, 'BUSDT': 0.0, 'CUSDT': 0.0, 'DUSDT': 0.0}
g1 = ex38.plan({s: float(v) for s, v in zip(sy39, a39) if v}, {}, tk38, i38, 0.0, 15.0, frac=0.5, start=pstart39)
b39, _ = R39.build_targets(w39, sy39, 10000.0, np.full(4, 5e8), 1.0, set(), slot='S1')          # retry, nothing filled
g2 = ex38.plan({s: float(v) for s, v in zip(sy39, b39) if v}, {}, tk38, i38, 0.0, 15.0, frac=0.5, start=pstart39)
check('T39 EMA-once-per-slot + partial step: retry == first attempt (targets and orders)', np.allclose(a39, b39) and g1 == g2 and len(g1) == 4, f'{g1}')

# T40: partial rebalance in the engine and in the bot make the same moves; REBAL_PARTIAL=1.0 is out of the hash, 0.5 is in it
from bt import run as bt_run
i0_40 = T - 24 * 40; ok40 = np.flatnonzero(d.valid[i0_40:i0_40 + 24 * 8].all(0))[:10]
w40 = np.zeros(N); w40[ok40[:5]] = [0.3, 0.3, 0.2, 0.1, 0.1]; w40[ok40[5:10]] = -np.array([0.3, 0.3, 0.2, 0.1, 0.1])
R40 = {}; _si40 = d.start_i
for fr_ in (1.0, 0.5):
    rec40 = []; d.start_i = i0_40
    bt_run(d, lambda dd, i_, mk_: w40, reb_h=24, band=0.0, fee_bps=0.0, slip_bps=0.0, gross=1.0, pos_cap=1.0, cap_short=1.0, max_lev=1.0, min_trade=0.0,
           trade_frac=fr_, verbose=False, rec=rec40, label='t40', uni_kwargs=dict(min_age_h=0, min_turn=0, top_n=0), end_i=i0_40 + 24 * 6)
    R40[fr_] = np.array([p_ for _, p_ in rec40])
d.start_i = _si40
P1, P5_ = R40[1.0], R40[0.5]
T40 = P1[0]
ratios = np.array([(P5_[k_] @ T40) / (T40 @ T40) for k_ in range(4)])
check('T40a engine trade_frac=0.5: position goes 1/2, 3/4, 7/8 of the way to a constant target (1.0: at once)',
      abs(ratios[0] - 0.5) < 0.02 and abs(ratios[1] - 0.75) < 0.03 and abs(ratios[2] - 0.875) < 0.04 and abs((P1[1] @ T40) / (T40 @ T40) - 1) < 0.03, f'{np.round(ratios, 3)}')
st40 = {}; cur40 = np.zeros(N); tk40 = {s: dict(mark=1.0, last=1.0, bid=1.0, ask=1.0) for s in syms}; i40 = {s: dict(qtyStep=1e-6, minQty=1e-6, maxQty=1e18, minNotional=0.0, tickSize=0.001) for s in syms}
tgt40 = {s: 1000.0 * float(v) for s, v in zip(syms, w40) if v}; bot_r = []
for k_ in range(4):
    pp40 = {s: dict(qty=float(q_), mark=1.0, entry=1.0, upnl=0, im=1, legs=[]) for s, q_ in zip(syms, cur40) if q_}
    legs40 = ex38.plan(tgt40, pp40, tk40, i40, 0.0, 0.0, frac=0.5, start={s: float(q_) for s, q_ in zip(syms, cur40)})
    for l_ in legs40: cur40[syms.index(l_[0])] += (1 if l_[1] == 'Buy' else -1) * l_[2]
    bot_r.append(float(cur40 @ (w40 * 1000.0)) / float((w40 * 1000.0) @ (w40 * 1000.0)))
check('T40b bot plan(frac=0.5) walks 1/2, 3/4, 7/8, 15/16 of the way, as the engine', np.allclose(bot_r, [0.5, 0.75, 0.875, 0.9375], atol=1e-6), f'{np.round(bot_r, 4)}')

# T41: fchg off via the sleeve configuration: SLEEVES without fchg == fchg kept with scale 0 == book without it; profile (evns/base.env) is reg. 5
def _envfile(p_):
    o_ = {}
    for ln_ in open(os.path.join(ROOT, 'evns', p_), encoding='utf-8'):
        ln_ = ln_.strip()
        if '=' in ln_ and not ln_.startswith('#'): k_, v_ = ln_.split('=', 1); o_[k_] = v_
    return o_
be41 = _envfile('base.env')
_sl41, _sc41 = C.SLEEVES, dict(C.SLEEVE_SCALES)
C.SLEEVES = ['listing', 'core', 'ml3', 'ml7', 'fchg', 'fcarry']; C.SLEEVE_SCALES = {'listing': 1, 'core': 1, 'ml3': 1, 'ml7': 1, 'fchg': 1, 'fcarry': 1}
S41 = Sleeves(log=lambda *a: None); S41.uni.prev = {}
w_with, p_with, _ = S41.combined(panel, age_h, set(), daily_update=True, hour_utc=0)
C.SLEEVE_SCALES['fchg'] = 0.0; S41 = Sleeves(log=lambda *a: None); S41.uni.prev = {}
w_zero, p_zero, _ = S41.combined(panel, age_h, set(), daily_update=True, hour_utc=0)
C.SLEEVES = ['listing', 'core', 'ml3', 'ml7', 'fcarry']; S41 = Sleeves(log=lambda *a: None); S41.uni.prev = {}
w_out, p_out, _ = S41.combined(panel, age_h, set(), daily_update=True, hour_utc=0)
C.SLEEVES, C.SLEEVE_SCALES = _sl41, _sc41
exp41 = (p_out['listing'] + p_out['core'] + p_out['ml3'] + p_out['ml7'] + p_out['fcarry']) / 5.0
check('T41a fchg weight 0 (scale 0) == sleeve removed == mean of the other five sleeves (sum of scales renormalised)',
      np.abs(w_zero - w_out).max() < 1e-12 and 'fchg' in p_zero and 'fchg' not in p_out and np.abs(w_out - exp41).max() < 1e-12 and np.abs(w_with - w_out).max() > 1e-6,
      f'diff zero/out {np.abs(w_zero - w_out).max():.1e}, with/out {np.abs(w_with - w_out).max():.2e}')
sl41 = be41['SLEEVES'].split(','); sc41 = dict((a.split(':')[0], float(a.split(':')[1])) for a in be41['SLEEVE_SCALES'].split(','))
check('T41b profile evns/base.env (reg. 5): no fchg in SLEEVES / SLEEVE_SCALES, ml8 weight 2, REBAL_PARTIAL=0.5, REBAL_MINUTE=1',
      sl41 == ['listing', 'core', 'ml3', 'ml7', 'ml8', 'fcarry'] and set(sc41) == set(sl41) and sc41['ml8'] == 2.0 and float(be41['REBAL_PARTIAL']) == 0.5 and be41['REBAL_MINUTE'] == '1', str(be41.get('SLEEVES')))
h41a = _cfg_hash(); h41b = _cfg_hash(REBAL_PARTIAL='1.0'); h41c = _cfg_hash(REBAL_PARTIAL='0.5')
check('T41c REBAL_PARTIAL=1.0 (off) leaves the registered config hash unchanged, 0.5 changes it', h41a == h41b and h41a != h41c, f'{h41a} {h41c}')

# ---------------------------------------------------------------- Reg. 5b: liquidation cascade filter (v4/cascade.py)
import cascade as CAS, io, contextlib
rng42 = np.random.default_rng(5)
days42 = [(datetime(2024, 1, 1).date() + timedelta(days=k)).isoformat() for k in range(500)]
C42 = {dd: float(0.01 + 0.002 * rng42.random()) for dd in days42}          # calm series, all < 0.012
C42[days42[400]] = 0.5; C42[days42[450]] = 0.4                                # two spikes
tr42 = CAS.trigger_days(C42, 0.95, 365, 180)
check('T42a trigger: both spikes trigger, nothing triggers before 180 valid days, a calm random series triggers only about 5 % of days', days42[400] in tr42 and days42[450] in tr42 and len(tr42) <= 40 and not any(dd in tr42 for dd in days42[:180]), f'{len(tr42)} triggers')
thr42 = CAS.threshold_series(C42, 0.95, 365, 180)
w42 = np.array([C42[dd] for dd in days42[400 - 364:401]]); check('T42b threshold = 95th percentile (linear) of the trailing 365 days incl. the day', abs(thr42[days42[400]] - np.quantile(w42, 0.95)) < 1e-12 and thr42[days42[100]] is None)
D42 = lambda k: datetime.strptime(days42[k], '%Y-%m-%d').date()
mm42 = [CAS.multiplier_for(D42(k), tr42, 2, 0.5) for k in (400, 401, 402, 403, 450, 451, 452, 453)]
check('T42c cut on d+1 and d+2 only (not on d, not on d+3)', mm42 == [1.0, 0.5, 0.5, 1.0, 1.0, 0.5, 0.5, 1.0], str(mm42))
C42b = dict(C42); C42b[days42[400]] = 0.001                                   # causality: changing day 401+ never changes the multiplier of day 401
C42c = {dd: v for dd, v in C42.items() if dd <= days42[400]}
check('T42d causality: trigger set of the series truncated at day d equals the full-series trigger set up to d (no look-ahead); lowering day d removes its trigger',
      CAS.trigger_days(C42c) == {dd for dd in tr42 if dd <= days42[400]} and days42[400] not in CAS.trigger_days(C42b))
# bot-side filter on a fake API: bootstrap, trigger, failure, key never printed
SECRET42 = 'SECRET-KEY-DO-NOT-PRINT'; os.environ['COINALYZE_API_KEY'] = SECRET42
td42 = tempfile.mkdtemp(prefix='cas42_'); NOW42 = datetime(2025, 5, 20, 0, 6, tzinfo=timezone.utc); T0_42 = NOW42.timestamp()
UNI42 = [f'C{j:02d}USDT' for j in range(30)]
calls42 = []
def fake_http_factory(spike_day=None, fail=False, status=200):
    def http(path, params, key):
        calls42.append((path, key == SECRET42, len(params.get('symbols', '').split(',')) if params.get('symbols') else 0))
        if fail: return status, None, 1.0
        if path == 'future-markets':
            return 200, [dict(exchange=ex, symbol_on_exchange=s, symbol=f'{s}.{ex}', is_perpetual=True, quote_asset='USDT') for ex in 'A6' for s in UNI42], 1.0
        out = []
        for code in params['symbols'].split(','):
            h = []; t = int(params['from']) // 86400 * 86400
            while t < int(params['to']):
                dd = datetime.fromtimestamp(t, timezone.utc).date().isoformat(); spike = (dd == spike_day)
                h.append({'t': t, 'l': 5000.0 if spike else 10.0, 's': 5000.0 if spike else 10.0} if 'liquidation' in path else {'t': t, 'c': 1e6})
                t += 86400
            out.append({'symbol': code, 'history': h})
        return 200, out, 1.0
    return http
sl42 = []; msgs42 = []
def mk42(**kw):
    return CAS.CascadeFilter(directory=td42, per_min=38, http=kw.pop('http'), clock=lambda: T0_42 + sum(sl42), sleep=lambda x: sl42.append(x), notify=msgs42.append,
                             log=lambda *a: None, universe_fn=lambda: UNI42, **kw)
f42 = mk42(http=fake_http_factory(spike_day='2025-05-17'))
buf42 = io.StringIO()
with contextlib.redirect_stdout(buf42):
    ok42 = f42.update(NOW42)
nc42 = len(calls42)
check('T42e bootstrap: backfill >= 365 d for universe x (A, 6), throttled <= 40 symbol-calls/min, key sent only in the header',
      ok42 and all(c[1] for c in calls42) and sum(c[2] for c in calls42) >= 2 * 60 and all(c[2] <= 20 for c in calls42) and sum(sl42) > 60.0 * (sum(c[2] for c in calls42) / 40.0 - 1.5),
      f'ok {ok42}, symbol-calls {sum(c[2] for c in calls42)}, waited {sum(sl42):.0f}s')
f42.cache['last_ok'] = '2025-05-19'    # data through the 19th: decisions on the 18th..20th
r42 = {k: CAS.multiplier_for(datetime(2025, 5, k).date(), {dd for dd, v in f42.cache['series'].items() if v[2]}, 2, 0.5) for k in (17, 18, 19, 20)}
check('T42f trigger day 2025-05-17 (spike): ×0.5 on 05-18 and 05-19, 1.0 on 05-17 and 05-20', r42 == {17: 1.0, 18: 0.5, 19: 0.5, 20: 1.0}, str(r42))
f42b = CAS.CascadeFilter(directory=tempfile.mkdtemp(), http=fake_http_factory(), sleep=lambda x: None, universe_fn=lambda: UNI42, log=lambda *a: None)
m42, why42 = f42b.multiplier(datetime(2025, 5, 19, 1, 1, tzinfo=timezone.utc))          # empty cache (nothing fetched yet)
f42.cache['last_ok'] = '2025-05-18'
m42b, _ = f42.multiplier(datetime(2025, 5, 19, 1, 1, tzinfo=timezone.utc))
m42c, _ = f42.multiplier(datetime(2025, 5, 21, 1, 1, tzinfo=timezone.utc))
check('T42g multiplier reads the cache only: 05-19 with data through 05-18 -> 0.5 (trigger 05-17 = d-2); empty cache -> 1.0; stale cache (no 05-20) for 05-21 -> 1.0',
      m42b == 0.5 and m42 == 1.0 and m42c == 1.0, f'{m42} {m42b} {m42c} {why42}')
# failure -> 1.0 and exactly one Telegram warning; key never printed
cache_txt42 = open(f42.path).read(); shutil.rmtree(td42); os.makedirs(td42)
msgs42.clear(); calls42.clear(); logs42 = []
f42f = CAS.CascadeFilter(directory=td42, http=fake_http_factory(fail=True, status=503), clock=lambda: T0_42 + sum(sl42), sleep=lambda x: None, notify=msgs42.append,
                         log=logs42.append, universe_fn=lambda: UNI42, tries=2)
r1 = f42f.update(NOW42); r2 = f42f.update(NOW42)
check('T42h API failure: update False, multiplier 1.0, ONE Telegram warning (not two)', (not r1) and (not r2) and f42f.multiplier(NOW42)[0] == 1.0 and len(msgs42) == 1 and msgs42[0].startswith('⚠️'), str(msgs42))
os.environ.pop('COINALYZE_API_KEY'); msgs42.clear()
f42n = CAS.CascadeFilter(directory=tempfile.mkdtemp(), http=fake_http_factory(), sleep=lambda x: None, notify=msgs42.append, log=logs42.append, universe_fn=lambda: UNI42)
check('T42i no key in env: update False, multiplier 1.0, warning sent', (not f42n.update(NOW42)) and f42n.multiplier(NOW42)[0] == 1.0 and len(msgs42) == 1)
os.environ['COINALYZE_API_KEY'] = SECRET42
dump42 = buf42.getvalue() + ' '.join(map(str, logs42)) + ' '.join(msgs42) + cache_txt42
check('T42j the key never appears in output, logs, Telegram text or the cache file', SECRET42 not in dump42 and len(dump42) > 1000)
cfg_src42 = open(os.path.join(HERE, 'config.py')).read()
check('T42k key is not a config value (no COINALYZE in config.py code lines) and not in the hash', all('COINALYZE' not in l_ or l_.strip().startswith('#') for l_ in cfg_src42.splitlines()) and 'COINALYZE_API_KEY' not in json.dumps(C.effective_config()[0]))
h42a = _cfg_hash(); h42b = _cfg_hash(CASCADE_FILTER='0', CASCADE_PCT='0.9', CASCADE_MULT='0.3', CASCADE_DIR='/x', COINALYZE_API_KEY='k'); h42c = _cfg_hash(CASCADE_FILTER='1'); h42d = _cfg_hash(CASCADE_FILTER='1', CASCADE_MULT='0.3'); h42e = _cfg_hash(CASCADE_FILTER='1', COINALYZE_API_KEY='zzz', CASCADE_DIR='/y')
check('T42l CASCADE_FILTER=0: hash unchanged by any CASCADE_* / key; on: hash changes with the parameters but not with key / cache dir', h42a == h42b and h42c != h42a and h42d != h42c and h42e == h42c, f'{h42a} {h42c} {h42d}')
# wiring: build_targets scales like leverage; main passes the multiplier; profiles
mem42 = Memory(); mem42['lev'] = 1.0; rk42 = RiskEngine(mem42, lambda *a: None); wt42 = np.zeros(N); wt42[0] = 0.01; wt42[1] = -0.01
t1_, _i1 = rk42.build_targets(wt42, syms, 10000.0, np.full(N, 1e9), 1.0, set(), slot='a'); mem42.pop('w_prev', None)
t2_, _i2 = rk42.build_targets(wt42, syms, 10000.0, np.full(N, 1e9), 1.0, set(), slot='b', cascade=0.5)
check('T42m build_targets(cascade=0.5) halves every target exactly (like a leverage factor); default 1.0 unchanged', np.allclose(t2_, 0.5 * t1_) and np.abs(t1_).sum() > 0 and _i2['cascade'] == 0.5, f'{np.abs(t1_).sum():.0f} {np.abs(t2_).sum():.0f}')
src42 = open(os.path.join(HERE, 'main.py')).read()
check('T42n main.py: multiplier read from the cache and passed to build_targets; background job started only when CASCADE_FILTER', 'cascade=cas_m' in src42 and 'self.cascade.multiplier(' in src42 and 'if C.CASCADE_FILTER else None' in src42)
pf42 = {p_: _envfile(p_) for p_ in ('base.env', 'mubite_challenge.env', 'mubite_funded.env', 'personal.env')}
stg42 = open(os.path.join(ROOT, 'ops', 'stage.sh')).read()
check('T42o profiles: CASCADE_FILTER not set to 1 in any env file (off for phase1/phase2/funded/personal); stage.sh enables it for trial only and keeps COINALYZE_API_KEY locally',
      all(pf42[p_].get('CASCADE_FILTER', '0') == '0' for p_ in pf42) and 'trial ]] && echo "CASCADE_FILTER=1"' in stg42 and 'COINALYZE_API_KEY' in stg42)

# ---------------------------------------------------------------- Reg. 5c: maker-only rebalancing + 2 rebalances/day (01:01 / 13:01)
# T43a: normal legs never go to market with MAKER_ONLY; forced symbols and risk passes still do
a43 = _Api24(); ex43 = Executor(a43, lambda *a: None, None, False); ex43.abort = threading.Event()
st43, _er43 = ex43.execute(legs24, meta24, a43.tickers(), lambda: {}, maker_attempts=2, wait_s=0, maker_only=True)
check('T43a maker_only: 2 quote rounds, NO market fallback, the unfilled leg is counted and cancelled', [x[0] for x in a43.sent] == ['PO', 'PO'] and st43['market_sent'] == 0 and st43['unfilled_legs'] == 1, f'{a43.sent} {st43}')
a43b = _Api24(); ex43b = Executor(a43b, lambda *a: None, None, False); ex43b.abort = threading.Event()
ex43b.execute(legs24, meta24, a43b.tickers(), lambda: {}, maker_attempts=2, wait_s=0, maker_only=True, force_market={'AAAUSDT'})
check('T43b forced symbol (delisting / cooldown / blocked) still finishes at market', [x[0] for x in a43b.sent] == ['PO', 'PO', 'MKT'], f'{a43b.sent}')
C.MAKER_ONLY = 1
a43c = _Api24(); ex43c = Executor(a43c, lambda *a: None, None, False); ex43c.abort = threading.Event()
ex43c.execute(legs24, meta24, a43c.tickers(), lambda: {}, maker_attempts=2, wait_s=0)
a43d = _Api24(); ex43d = Executor(a43d, lambda *a: None, None, False); ex43d.abort = threading.Event()
ex43d.execute(legs24, meta24, a43d.tickers(), lambda: {}, maker_attempts=2, wait_s=0, maker_only=False)
C.MAKER_ONLY = 0
check('T43c MAKER_ONLY=1 is the default of execute(); maker_only=False (scale pass) keeps the market fallback', 'MKT' not in [x[0] for x in a43c.sent] and [x[0] for x in a43d.sent] == ['PO', 'PO', 'MKT'], f'{a43c.sent} {a43d.sent}')
src43 = open(os.path.join(HERE, 'main.py')).read()
check('T43d main.py: scale pass passes maker_only=False, the rebalance passes force_market (blocked + delisted); flatten_all / day marker use market only',
      'wait_s=60, maker_only=False' in src43 and 'force_market=force_mkt' in src43 and 'self.api.market(' in src43 and 'maker_only' not in inspect.getsource(Executor.flatten_all))
# T43e: unfilled remainder is carried: next slot steps from the position at ITS start (target recomputed), nothing is lost
tk43 = {'AAAUSDT': dict(mark=10.0, last=10.0, bid=9.99, ask=10.01)}
l1 = ex38.plan({'AAAUSDT': 1000.0}, {}, tk43, {'AAAUSDT': dict(qtyStep=0.1, minQty=0.1, maxQty=1e9, minNotional=5.0)}, 0.0, 15.0, frac=0.5, start={})
pos43 = {'AAAUSDT': dict(qty=20.0, mark=10.0, entry=10.0, upnl=0, im=1, legs=[])}          # only 200 of the 500 step filled
l2 = ex38.plan({'AAAUSDT': 1000.0}, pos43, tk43, {'AAAUSDT': dict(qtyStep=0.1, minQty=0.1, maxQty=1e9, minNotional=5.0)}, 0.0, 15.0, frac=0.5, start={'AAAUSDT': 20.0})
check('T43e carry-over: slot 1 step = 500 of 1000; next slot (position 200 at its start) steps half of the remaining 800 = 400', abs(l1[0][2] - 500.0) < 1e-6 and abs(l2[0][2] - 400.0) < 1e-6, f'{l1} {l2}')
# T43f: slots at 01:01 and 13:01 with REBAL_EVERY_H=12
_e43, _m43 = C.REBAL_EVERY_H, C.REBAL_MINUTE; C.REBAL_EVERY_H = 12; C.REBAL_MINUTE = 1; C.REBAL_HOUR_UTC = 1
U43 = lambda h, mi, dd=3: datetime(2026, 5, dd, h, mi, tzinfo=timezone.utc)
check('T43f slots 01:01 / 13:01: 01:30 -> 01:01, 12:59 -> 01:01, 13:01 -> 13:01, 00:30 -> previous 13:01',
      _slot(U43(1, 30)) == U43(1, 1) and _slot(U43(12, 59)) == U43(1, 1) and _slot(U43(13, 1)) == U43(13, 1) and _slot(U43(0, 30)) == U43(13, 1, 2))
# day marker: next rebalance is 12 h after the slot -> the "still open at the next rebalance" test uses the same grid
check('T43g day marker horizon: from the 01:01 slot the next one is 13:01', _slot(U43(5, 0)) + timedelta(hours=12) == U43(13, 1))
# T43h: ml8 holds the weights of the last 8h bar (00/08/16) between them
ep43 = int(datetime(2026, 5, 3, 12, tzinfo=timezone.utc).timestamp() // 3600)           # last closed bar opens at 12:00
pan43 = dict(panel); pan43['grid'] = np.arange(ep43 - 2199, ep43 + 1)
pan43_t = {k_: (v_[:-4] if k_ in ('grid', 'cff', 'high', 'low', 't24', 'fund', 'prem', 'valid') else v_) for k_, v_ in pan43.items()}    # ends at the 08:00 bar
m43 = mu25
C.REBAL_EVERY_H = 8; w_ref43 = S.ml8(pan43_t, m43, age_h, 8)
C.REBAL_EVERY_H = 12; w_hold43 = S.ml8(pan43, m43, age_h, 13)
lags43 = [Sleeves.ml8_bar_lag(dict(grid=np.array([ep_]))) for ep_ in (ep43 - 12, ep43 - 4, ep43, ep43 + 4, ep43 + 8)]   # bar opens 00, 08, 12, 16, 20
C.REBAL_EVERY_H = 8; lag8_43 = Sleeves.ml8_bar_lag(dict(grid=np.array([ep43])))
C.REBAL_EVERY_H = _e43; C.REBAL_MINUTE = _m43
check('T43h ml8 at the 13:01 slot (bar 12) == ml8 computed on the 08:00 bar; lags 0/0/4/0/4; cadence 8h: lag 0 (unchanged)',
      (np.abs(w_ref43).sum() > 0 or not S.models.get(8)) and np.allclose(w_ref43, w_hold43) and lags43 == [0, 0, 4, 0, 4] and lag8_43 == 0, f'{lags43} {lag8_43}')
# T43i: hash unchanged when the flags are off; MAKER_ONLY=1 changes it; the profile registers 5c
h43a = _cfg_hash(); h43b = _cfg_hash(MAKER_ONLY='0'); h43c = _cfg_hash(MAKER_ONLY='1'); h43d = _cfg_hash(REBAL_EVERY_H='12')
be43 = _envfile('base.env')
check('T43i MAKER_ONLY=0 leaves the hash unchanged, 1 and REBAL_EVERY_H=12 change it; evns/base.env: MAKER_ONLY=1, REBAL_EVERY_H=12, REBAL_HOUR_UTC=1, REBAL_MINUTE=1',
      h43a == h43b and h43a != h43c and h43a != h43d and be43.get('MAKER_ONLY') == '1' and be43.get('REBAL_EVERY_H') == '12' and be43.get('REBAL_HOUR_UTC') == '1' and be43.get('REBAL_MINUTE') == '1', f'{h43a} {h43c} {h43d}')


# ---------------------------------------------------------------- Audit hardening 2026-11-13 (F1-F10, B1, B4): fakes only, no network
import main as botmain, state as botstate, bybit as bybit_mod
from bybit import BatchError, Bybit
import cascade as CASF
_meta_f = dict(qtyStep=0.1, minQty=0.1, maxQty=1e9, minNotional=5.0, tickSize=0.001)
_pos_f = lambda **kw: {**dict(qty=10.0, mark=5.0, entry=5.0, upnl=0.0, rpnl=0.0, im=1.0, legs=[]), **kw}
class _ApiF:
    """Fake exchange for the guard tests: equity readings are consumed in order (the last one repeats)."""
    def __init__(self, eqs, positions=None): self.eqs = list(eqs); self.pos = positions if positions is not None else {}; self.orders = []; self.eq_calls = 0
    def equity(self):
        self.eq_calls += 1
        return self.eqs.pop(0) if len(self.eqs) > 1 else self.eqs[0]
    def positions(self): return {k: dict(v) for k, v in self.pos.items()}
    def market(self, sym, qty, side, reduce_only=False, position_idx=0): self.orders.append((sym, qty, side)); return {}
    def cancel_all(self, *a, **k): return None
    def tickers(self): return {}
class _ExF:
    def __init__(self): self.flat = []
    def flatten_all(self, instr, reason, quiet=False): self.flat.append((reason, quiet)); return 1
def mk_botf(eqs, pos=None):
    b = Bot.__new__(Bot); b.api = _ApiF(eqs, pos); b.mem = Memory(); b.mem.save = lambda: None; b.risk = RiskEngine(b.mem, lambda *a: None)
    b.lock = threading.Lock(); b.abort_event = threading.Event(); b.instr = {'AAAUSDT': dict(_meta_f), 'BBBUSDT': dict(_meta_f)}; b.ex = _ExF()
    b.mem['initial'] = 10000.0; b.mem['eq_peak'] = 10000.0; b.mem['day_date'] = '2026-05-03'; b.mem['day_anchor'] = 10000.0; b.mem['day_peak'] = 10000.0
    b.now = lambda: datetime(2026, 5, 3, 12, 0, tzinfo=timezone.utc)
    return b
_tgo_f = []; _orig_tgo = botmain.tg_once; botmain.tg_once = lambda key, text, hours=24.0: _tgo_f.append((key, text))
_orig_tg_f = botmain.tg; botmain.tg = lambda text: _tgo_f.append(('tg', text)); _orig_tg_state = botstate.tg; botstate.tg = botmain.tg

# T44 (F1): a latched halt re-flattens every 5 min while positions remain; quiet (tg_once style), nothing when flat
b44 = mk_botf([10000.0], {'AAAUSDT': _pos_f()}); b44.mem['halt_total'] = True
b44.watchdog(); n1_44 = len(b44.ex.flat)
b44.watchdog(); n2_44 = len(b44.ex.flat)
b44._reflat_t -= C.REFLATTEN_S + 1; b44.watchdog(); n3_44 = len(b44.ex.flat)
b44.api.pos.clear(); b44._reflat_t -= C.REFLATTEN_S + 1; b44.watchdog(); n4_44 = len(b44.ex.flat)
b44.mem['halt_total'] = False; b44.mem['halted_day'] = '2026-05-03'; b44.api.pos['AAAUSDT'] = _pos_f(); b44._reflat_t -= C.REFLATTEN_S + 1; b44.watchdog(); n5_44 = len(b44.ex.flat)
b44.mem['halted_day'] = '2026-05-02'; b44._reflat_t -= C.REFLATTEN_S + 1; b44.watchdog(); n6_44 = len(b44.ex.flat)
check('T44a halt_total with positions: re-flatten at once, NOT again within 5 min, again after 5 min, nothing when flat',
      (n1_44, n2_44, n3_44, n4_44) == (1, 1, 2, 2) and all(q for _, q in b44.ex.flat), f'{[n1_44, n2_44, n3_44, n4_44]}')
check('T44b daily halt (halted_day today) re-flattens too; an old halted_day does not; messages go through tg_once (throttled), not tg',
      n5_44 == 3 and n6_44 == 3 and any(k.startswith('reflatten_') for k, _ in _tgo_f) and not any(k == 'tg' and 'Закрываю' in t for k, t in _tgo_f), f'{n5_44} {n6_44}')
check('T44c Executor.flatten_all(quiet=True): no "closing" message, failure message via the throttled sender',
      'quiet' in inspect.signature(Executor.flatten_all).parameters and "_alert('flatten_fail'" in inspect.getsource(Executor.flatten_all))

# T45 (F2): equity plausibility
slp_f = []; _sleep_orig = time.sleep; time.sleep = lambda x: slp_f.append(x); C.EQ_CONFIRM_S = 20.0
try:
    bad_f = []
    for v in (None, 0, -5.0, float('nan'), 'x'):
        b = mk_botf([v]); b.guards = lambda eq: (_ for _ in ()).throw(AssertionError('guards must not run on an implausible equity'))
        b.watchdog(); bad_f.append(b.checked_equity() is None and not b.mem['halt_total'])
    check('T45a equity None / 0 / negative / NaN / junk is never acted on (guards not called, no latch)', all(bad_f), str(bad_f))
    b = mk_botf([10000.0, 9040.0, 10000.0])                    # 9040 = below the floor guard (9050) AND a 9.6 % drop; the second reading is normal
    b.watchdog(); b.watchdog()
    check('T45b a floor-breaking drop that the 2nd reading (20 s later) does not confirm is a glitch: no halt, no flatten', slp_f == [20.0] and not b.mem['halt_total'] and not b.ex.flat, f'{slp_f} {b.ex.flat}')
    slp_f.clear(); b = mk_botf([10000.0, 9040.0, 9040.0]); b.watchdog(); b.watchdog()
    check('T45c the same drop confirmed by the 2nd reading acts: total halt latched and flatten', slp_f == [20.0] and b.mem['halt_total'] and len(b.ex.flat) == 1, f'{slp_f} {b.ex.flat}')
    slp_f.clear(); b = mk_botf([10000.0, 9800.0]); b.watchdog(); b.watchdog()
    check('T45d a small fall (2 % of initial) needs no confirmation (no sleep)', slp_f == [] and b._eq_ok == 9800.0)
    b = mk_botf([0.0]); b.rebalance(); b2 = mk_botf([None]); b2.scale_pass() if False else None
    check('T45e rebalance with an implausible equity returns without trading or flattening', not b.ex.flat and b.api.eq_calls == 1)
    check('T45f the limit_breach path gets the checked reading (watchdog passes it); EQ keys are safety keys outside the hash',
          'self.checked_equity()' in inspect.getsource(Bot.watchdog) and all(k not in C.effective_config()[0] for k in ('EQ_JUMP_FRAC', 'EQ_CONFIRM_S', 'RISK_CUT_MARKET', 'REFLATTEN_S', 'STATE_SAVE_ALERT_H')))
finally:
    time.sleep = _sleep_orig; C.EQ_CONFIRM_S = 0.0

# T46 (F3): unfilled legs in the summary; risk-reducing legs at market after the maker rounds
b46 = mk_botf([10000.0])
legs46 = [('AAAUSDT', 'Sell', 300.0, True, False), ('BBBUSDT', 'Sell', 200.0, True, True), ('BBBUSDT', 'Buy', 100.0, False, False), ('CCCUSDT', 'Sell', 150.0, True, True), ('DDDUSDT', 'Buy', 90.0, False, False)]
pos46 = {'AAAUSDT': _pos_f(), 'BBBUSDT': _pos_f(), 'CCCUSDT': _pos_f()}; tg46 = {'AAAUSDT': 100.0, 'BBBUSDT': -50.0, 'DDDUSDT': 90.0}; pool46 = {'AAAUSDT', 'BBBUSDT', 'DDDUSDT'}
r_cut = b46.risk_cut_market(legs46, pos46, tg46, pool46, 0.5, 1.0)
r_nocut = b46.risk_cut_market(legs46, pos46, tg46, pool46, 0.9, 1.0)
C.RISK_CUT_MARKET = 0; r_off = b46.risk_cut_market(legs46, pos46, tg46, pool46, 0.5, 1.0); C.RISK_CUT_MARKET = 1
check('T46a K fell below 0.8 x previous: reducing-only symbols (AAA, CCC) go to market; the flip BBB (close + open) and the opener DDD do not',
      r_cut == {'AAAUSDT', 'CCCUSDT'} and r_off == set(), f'{r_cut} {r_off}')
check('T46b K barely moved: only the close of a name that left the candidate pool (CCC) is forced; flag off -> nothing', r_nocut == {'CCCUSDT'}, str(r_nocut))
class _Api46(_Api24):
    def tickers(self): return {s_: dict(mark=10.0, last=10.0, bid=9.99, ask=10.01) for s_ in ('AAAUSDT', 'BBBUSDT')}
a46 = _Api46(); ex46 = Executor(a46, lambda *a: None, None, False); ex46.abort = threading.Event()
st46, _e = ex46.execute([('AAAUSDT', 'Sell', 500.0, True, False), ('BBBUSDT', 'Buy', 500.0, False, False)], {'AAAUSDT': meta24['AAAUSDT'], 'BBBUSDT': dict(meta24['AAAUSDT'])},
                        {'AAAUSDT': dict(mark=10.0, last=10.0, bid=9.99, ask=10.01), 'BBBUSDT': dict(mark=10.0, last=10.0, bid=9.99, ask=10.01)},
                        lambda: {'AAAUSDT': dict(qty=50.0, mark=10.0)}, maker_attempts=1, wait_s=0, maker_only=True, force_market=r_cut)
mk46 = [x for x in a46.sent if x[0] == 'MKT']
check('T46c maker_only + force_market: the reducing leg finishes at market, the opening leg is counted as unfilled with its notional',
      len(mk46) == 1 and mk46[0][1] == 'AAAUSDT' and st46['unfilled_legs'] == 1 and abs(st46['unfilled_notional'] - 500.0) < 1.0, f'{a46.sent} {st46.get("unfilled_legs")} {st46.get("unfilled_notional")}')
src_main = open(os.path.join(HERE, 'main.py')).read()
check('T46d rebalance: Telegram summary carries unfilled legs + notional; risk_cut_market feeds force_mkt', 'не исполнено ног' in src_main and 'unfilled_notional' in src_main and 'force_mkt |= rc' in src_main)

# T47 (F4): partial batch failure, resend only the rest; cancel verification; cleanup on an exception
class _BatchApi:
    def __init__(self, fail_after, resting=()):
        self.fail_after = fail_after; self.singles = []; self.resting = list(resting); self.cancels = 0; self.accepted = []
    def create_batch(self, orders):
        self.accepted += [o['symbol'] for o in orders[:self.fail_after]]
        raise BatchError([(0, '')] * self.fail_after, RuntimeError('timeout in chunk 2'))
    def post_only(self, sym, q, side, price, reduce_only=False): self.singles.append(sym)
    def market(self, *a, **k): self.singles.append('MKT')
    def cancel_all(self, *a, **k): self.cancels += 1
    def open_orders(self): return [{'symbol': s, 'side': 'Buy'} for s in self.resting]
q47 = [(f'S{j:02d}USDT', 'Buy', 1.0, False, '10.0', dict(meta, maxQty=1e9)) for j in range(12)]
a47 = _BatchApi(10); ex47 = Executor(a47, lambda *a: None, None, False); r47 = ex47._send_quotes(q47)
check('T47a batch fails in chunk 2 after 10 accepted: only the 2 unaccepted legs are resent as singles, all 12 reported accepted, no duplicates',
      a47.singles == ['S10USDT', 'S11USDT'] and len(r47) == 12 and all(v[0] == 0 for v in r47.values()), f'{a47.singles}')
a47b = _BatchApi(10, resting=['S10USDT']); ex47b = Executor(a47b, lambda *a: None, None, False); r47b = ex47b._send_quotes(q47)
check('T47b of the unaccepted legs one already rests on the book (the failed request had been processed): not resent', a47b.singles == ['S11USDT'] and r47b['S10USDT'] == (0, ''), f'{a47b.singles}')
bb47 = Bybit('http://x', 'k', 's'); calls47 = []
def _req47(method, path, params=None, auth=False, retries=4, raw=False):
    calls47.append(len(params['request']))
    if len(calls47) == 2: raise RuntimeError('timeout')
    return {'retExtInfo': {'list': [{'code': 0, 'msg': 'OK'}] * len(params['request'])}}
bb47._req = _req47; _t0 = bybit_mod.time.sleep; bybit_mod.time.sleep = lambda x: None
try:
    bb47.create_batch([{'symbol': f'S{j}', 'side': 'Buy'} for j in range(23)]); err47 = None
except BatchError as e:
    err47 = e
finally:
    bybit_mod.time.sleep = _t0
check('T47c Bybit.create_batch: a failing LATER chunk raises BatchError listing the 10 orders of the accepted first chunk', err47 is not None and len(err47.done) == 10 and calls47 == [10, 10], f'{err47} {calls47}')
class _CxApi:
    def __init__(self, stuck): self.stuck = stuck; self.cancels = 0
    def cancel_all(self, *a, **k): self.cancels += 1
    def open_orders(self): return [{'symbol': 'AAAUSDT'}] if self.stuck else []
tgm47 = []
cx = _CxApi(True); exc = Executor(cx, lambda *a: None, tgm47.append, False); ok_cx = exc._cancel_verified()
cx2 = _CxApi(False); exc2 = Executor(cx2, lambda *a: None, tgm47.append, False); ok_cx2 = exc2._cancel_verified()
check('T47d cancel_all verified by open_orders: clean -> one cancel, no alert; orders left -> retried once (2 cancels) and ONE Telegram alert',
      ok_cx2 and cx2.cancels == 1 and (not ok_cx) and cx.cancels == 2 and len(tgm47) == 1 and 'AAAUSDT' in tgm47[0], f'{cx.cancels} {cx2.cancels} {tgm47}')
class _BoomApi(_Api24):
    def __init__(self): super().__init__(); self.n = 0
    def tickers(self): raise RuntimeError('boom')
bo47 = _BoomApi(); exb = Executor(bo47, lambda *a: None, None, False); exb.abort = threading.Event(); cc = []; bo47.cancel_all = lambda *a, **k: cc.append(1)
try:
    exb.execute(legs24, meta24, bo47.tickers.__func__ and {'AAAUSDT': dict(mark=10.0, last=10.0, bid=9.99, ask=10.01)}, lambda: {}, maker_attempts=2, wait_s=0, maker_only=True); exc_raised = False
except RuntimeError:
    exc_raised = True
check('T47e an exception in the middle of execute() propagates and the finally-block still cancels the resting quotes (round-1 cancel + final cancel)', exc_raised and len(cc) >= 2 and [x[0] for x in bo47.sent] == ['PO'], f'{cc} {bo47.sent}')

# T48 (F5): per-trade guard = unrealised + realised, base min(equity, initial); also in the rebalance limit_breach path
b48 = mk_botf([10000.0], {'AAAUSDT': _pos_f(upnl=-200.0, rpnl=-80.0)})
br48 = b48.per_trade_breaches(b48.api.positions(), 10000.0)
b48b = mk_botf([10000.0], {'AAAUSDT': _pos_f(upnl=-200.0, rpnl=0.0)})
br48b = b48b.per_trade_breaches(b48b.api.positions(), 10000.0); br48c = b48b.per_trade_breaches(b48b.api.positions(), 7000.0)
check('T48a loss = upnl + realised: -200 alone is inside 0.9 x 3 % x 10 000 = 270, -280 is not; with equity 7000 the base falls to equity (limit 189) and -200 breaches',
      len(br48) == 1 and abs(br48[0][2] - 270.0) < 1e-9 and br48b == [] and len(br48c) == 1 and abs(br48c[0][2] - 189.0) < 1e-9, f'{br48} {br48b} {br48c}')
b48._watchdog_locked(10000.0)
check('T48b watchdog closes the position (reduce-only market), quarantines it', b48.api.orders and b48.api.orders[0][0] == 'AAAUSDT' and 'AAAUSDT' in b48.mem['cooldown'], f'{b48.api.orders}')
check('T48c limit_breach (lock held by a rebalance) reports the per-trade breach; healthy positions -> None', (b48.limit_breach(10000.0) or '').startswith('PER-TRADE') and b48b.limit_breach(10000.0) is None, str(b48.limit_breach(10000.0)))
pos_src = open(os.path.join(HERE, 'bybit.py')).read()
check('T48d positions() reads curRealisedPnl', "curRealisedPnl" in pos_src and "rpnl=sum(" in pos_src)

# T49 (F6): recover_day_anchor sign logic unchanged for realised P&L
class _TxApi:
    def __init__(self, v): self.v = v
    def transaction_log_sum(self, ms): return self.v
b49 = Bot.__new__(Bot); b49.now = lambda: datetime(2026, 5, 3, 12, 0, tzinfo=timezone.utc)
an = []
for v in (50.0, -100.0, 300.0):
    b49.api = _TxApi(v); an.append(b49.recover_day_anchor(10000.0)[0])
check('T49 recover_day_anchor: anchor = equity - realised today (+50 -> 9950, -100 -> 10100); a +300 day (anchor < 99 % of equity) falls back to equity', an == [9950.0, 10100.0, 10000.0], str(an))

# T50 (F8): strict env parsing, empty hash, hash mismatch -> guard-only (no crash loop), no rebalance, guards stay on
r50 = subprocess.run([sys.executable, os.path.join(HERE, 'config.py')], env=dict(os.environ, VT_TARGET_DVOL='0,006'), capture_output=True, text=True, timeout=60)
check('T50a an unparsable value stops the start with a clear message naming the variable (no silent default)', r50.returncode != 0 and 'VT_TARGET_DVOL' in (r50.stdout + r50.stderr) and 'FATAL' in (r50.stdout + r50.stderr), (r50.stderr or r50.stdout)[-200:])
class _Stop(Exception): pass
class _StartApi:
    def switch_one_way(self): raise _Stop()
td50 = tempfile.mkdtemp(prefix='f8_'); _sf, _eh, _dr = C.STATE_FILE, C.EXPECTED_CONFIG_HASH, C.DRY_RUN
C.STATE_FILE = os.path.join(td50, 'st.json'); C.STATE_BACKUP_FILE = ''
def mk_start50():
    b = Bot.__new__(Bot); b.api = _StartApi(); b.mem = Memory(); b.mem.save = lambda: None; b.risk = RiskEngine(b.mem, lambda *a: None)
    b.sleeves = types.SimpleNamespace(held={}); return b
try:
    C.EXPECTED_CONFIG_HASH = ''; C.DRY_RUN = False; b = mk_start50()
    try: b.start(); e50 = None
    except SystemExit as e: e50 = e
    C.DRY_RUN = True; b = mk_start50()
    try: b.start()
    except _Stop: stop50 = True
    else: stop50 = False
    check('T50b empty EXPECTED_CONFIG_HASH: refuses to start unless DRY_RUN=1 (dry run passes the gate)', e50 is not None and stop50 and not b.guard_only, f'{e50}')
    C.DRY_RUN = False; C.EXPECTED_CONFIG_HASH = 'deadbeefdeadbeef'; _tgo_f.clear(); b = mk_start50()
    try: b.start()
    except _Stop: pass
    sent1 = [t for k, t in _tgo_f if k == 'tg' and 'не совпадает' in t]
    b2 = mk_start50()
    try: b2.start()
    except _Stop: pass
    sent2 = [t for k, t in _tgo_f if k == 'tg' and 'не совпадает' in t]
    check('T50c hash mismatch: the process stays alive in guard-only mode (flag set, /status says so) and ONE Telegram message, also across a restart (persistent throttle)',
          b.guard_only and b2.guard_only and len(sent1) == 1 and len(sent2) == 1 and botstate.get_status(True).get('guard_only') is True, f'{len(sent1)} {len(sent2)}')
    gb = mk_botf([10000.0], {'AAAUSDT': _pos_f(upnl=-400.0)}); gb.guard_only = True; gb.guard_reason = 'test'
    gb.api.equity = lambda: (_ for _ in ()).throw(AssertionError('rebalance must not run'))
    gb.rebalance(); gb.scale_pass(); gb.api.equity = lambda: 10000.0
    gb.watchdog()
    check('T50d guard-only: rebalance / scale pass do nothing, the watchdog guards still work (per-trade loss closed the position)', gb.api.orders and gb.api.orders[0][0] == 'AAAUSDT', f'{gb.api.orders}')
    check('T50e loop body skips the rebalance in guard-only mode', 'if self.guard_only:' in inspect.getsource(Bot.loop_body) and 'raise SystemExit' not in inspect.getsource(Bot.enter_guard_only))
finally:
    C.STATE_FILE, C.EXPECTED_CONFIG_HASH, C.DRY_RUN = _sf, _eh, _dr

# T51 (F9 + B1): state save durability / alert, token never in logs
td51 = tempfile.mkdtemp(prefix='f9_'); _paths = (C.STATE_FILE, C.STATE_BACKUP_FILE, C.TG_TOKEN, C.TG_CHAT)
C.STATE_FILE = os.path.join(td51, 'state', 's.json'); C.STATE_BACKUP_FILE = os.path.join(td51, 'state_backup', 's.backup.json'); C.TG_TOKEN = ''
fs51 = []; _fs = os.fsync; os.fsync = lambda fd: (fs51.append(fd), _fs(fd))[1]
try:
    m51 = Memory(); m51['initial'] = 123.0; m51.save()
finally:
    os.fsync = _fs
check('T51a save: fsync before the rename, state and backup copy written, the copy in a DIFFERENT directory',
      len(fs51) == 2 and json.load(open(C.STATE_FILE))['initial'] == 123.0 and json.load(open(C.STATE_BACKUP_FILE))['initial'] == 123.0
      and os.path.dirname(C.STATE_FILE) != os.path.dirname(C.STATE_BACKUP_FILE), f'{len(fs51)}')
msgs51 = []; _tg51 = botstate.tg; botstate.tg = msgs51.append; botstate._ALERTS.clear()
C.STATE_FILE = '/proc/nonexistent_dir/s.json'; C.STATE_BACKUP_FILE = ''
m52 = Memory(); m52.save(); m52.save()
n_a = len(msgs51); botstate._ALERTS['state_save_fail'] -= 3601; m52.save()
check('T51b a failing save goes to Telegram, at most once an hour (2 failures -> 1 message, after an hour -> 2nd)', n_a == 1 and len(msgs51) == 2 and 'state' in msgs51[0], f'{n_a} {len(msgs51)}')
botstate.tg = _tg51
tok51 = '123456789:AAHdqTcvCH1vGWJxfSeofSAs0K5PALDsaw0'; C.TG_TOKEN = tok51; C.TG_CHAT = '42'; logs51 = []
_log51, _post = botstate.log, botstate.requests.post; botstate.log = logs51.append
def _boom_post(url, **kw): raise requests.ConnectionError(f'HTTPSConnectionPool: Max retries exceeded with url: /bot{tok51}/sendMessage')
botstate.requests.post = _boom_post
try:
    botstate.tg('hello'); botstate.Memory._push('{}')
finally:
    botstate.log, botstate.requests.post = _log51, _post
check('T51c B1: the Telegram token never reaches the log (tg and the state push); scrub() also catches a token-shaped string', logs51 and all(tok51 not in l and '[REDACTED]' not in l for l in logs51) and tok51 not in botstate.scrub(f'x /bot{tok51}/y') and 'bot***' in botstate.scrub(f'/bot{tok51}/y', 'other'), str(logs51))
C.STATE_FILE, C.STATE_BACKUP_FILE, C.TG_TOKEN, C.TG_CHAT = _paths

# T52 (F10): cascade keeps a known trigger window when the cache is behind; extra retry at 00:50; legs without instrument meta are alerted
cf52 = CASF.CascadeFilter(directory=tempfile.mkdtemp(), http=lambda *a: (500, None, 1), sleep=lambda x: None, universe_fn=lambda: [], log=lambda *a: None)
cf52.cache = {'liq': {}, 'oi': {}, 'universe': [], 'codes': {}, 'uni_ts': 0, 'warn': False, 'err': '', 'last_ok': '2025-05-19',
              'series': {'2025-05-18': [0.01, 0.02, False], '2025-05-19': [0.05, 0.02, True]}}
m_a = cf52.multiplier(datetime(2025, 5, 21, 1, 1, tzinfo=timezone.utc))       # yesterday 05-20 missing (cache 1 day behind), trigger 05-19 is d-2
m_b = cf52.multiplier(datetime(2025, 5, 22, 1, 1, tzinfo=timezone.utc))       # 2 days behind: the window d-1, d-2 holds no known trigger
m_c = cf52.multiplier(datetime(2025, 5, 25, 1, 1, tzinfo=timezone.utc))       # 5 days behind
check('T52a cache 1 day behind with a known trigger in the window: x0.5 kept (reason says the cache lags); 2+ days behind without a known trigger: 1.0',
      m_a[0] == 0.5 and 'отстаёт' in m_a[1] and m_b[0] == 1.0 and m_c[0] == 1.0, f'{m_a} {m_b} {m_c}')
cf52.cache['last_ok'] = '2025-05-18'; cf52.last_try = 5000.0; cf52.clock = lambda: 5100.0
d_ = lambda h, mi: datetime(2025, 5, 20, h, mi, tzinfo=timezone.utc)
due52 = [cf52.due(d_(0, 49)), cf52.due(d_(0, 50)), cf52.due(d_(0, 55)), cf52.due(d_(1, 5))]
check('T52b extra Coinalyze retry at 00:50 UTC (once), not earlier, regardless of the 20-minute spacing; later only on the normal spacing', due52 == [False, True, False, False], str(due52))
tgm52 = []; a52 = _Api24(); ex52 = Executor(a52, lambda *a: None, tgm52.append, False); ex52.abort = threading.Event()
ex52.execute([('ZZZUSDT', 'Sell', 100.0, True, True), ('AAAUSDT', 'Buy', 500.0, False, False)], meta24, {'AAAUSDT': dict(mark=10.0, last=10.0, bid=9.99, ask=10.01)}, lambda: {'ZZZUSDT': dict(qty=5.0, mark=20.0)}, maker_attempts=0, wait_s=0, maker_only=False)
check('T52c a leg whose symbol has no instrument meta is announced in Telegram (not silently dropped); the other leg still trades', any('ZZZUSDT' in t for t in tgm52) and [x[0] for x in a52.sent] == ['MKT'], f'{tgm52} {a52.sent}')

# T53 (B4): API key with money-moving permissions -> refuse to trade
class _KeyApi:
    def __init__(self, risks=None, err=None): self.risks, self.err = risks, err
    def api_key_wallet_risks(self):
        if self.err: raise self.err
        return self.risks
_sf2 = C.STATE_FILE; C.STATE_FILE = os.path.join(td50, 'st53.json'); _tgo_f.clear()
try:
    bk = mk_botf([10000.0]); bk.api = _KeyApi(['Withdraw', 'AccountTransfer']); bk.check_key_permissions()
    bk2 = mk_botf([10000.0]); bk2.api = _KeyApi([]); bk2.check_key_permissions()
    bk3 = mk_botf([10000.0]); bk3.api = _KeyApi(err=ApiError(10003, 'bad key')); bk3.check_key_permissions()
finally:
    C.STATE_FILE = _sf2
bb53 = Bybit('http://x', 'k', 's'); bb53._req = lambda *a, **k: {'permissions': {'ContractTrade': ['Order', 'Position'], 'Wallet': ['AccountTransfer', 'SubMemberTransfer']}, 'expiredAt': ''}
bb53b = Bybit('http://x', 'k', 's'); bb53b._req = lambda *a, **k: {'permissions': {'ContractTrade': ['Order', 'Position'], 'Wallet': []}}
check('T53 key with Withdraw / AccountTransfer / SubMemberTransfer: guard-only (no trading) + Telegram; clean key and an unreadable check do not block; the parser sees the Wallet list',
      bk.guard_only and 'Withdraw' in bk.guard_reason and any('права на вывод' in t for k, t in _tgo_f if k == 'tg') and not bk2.guard_only and not bk3.guard_only
      and bb53.api_key_wallet_risks() == ['AccountTransfer', 'SubMemberTransfer'] and bb53b.api_key_wallet_risks() == [] and 'self.check_key_permissions()' in inspect.getsource(Bot.start))
botmain.tg_once = _orig_tgo; botmain.tg = _orig_tg_f; botstate.tg = _orig_tg_state
for _k in ('guard_only', 'guard_reason', 'error', 'ok'): botstate.STATUS.pop(_k, None)

print(f'\n{sum(1 for _, ok in res if ok)}/{len(res)} passed')
sys.exit(0 if all(ok for _, ok in res) else 1)
