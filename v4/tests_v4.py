#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit/consistency tests for PROP-SLEEVES v4 against the research engine (small data window)."""
import os, sys, json, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
os.environ.update(dict(FIRM='hyrotrader_2step', MODE='challenge', ACCOUNT_SIZE='10000', BYBIT_API_KEY='x', BYBIT_API_SECRET='x',
                       USE_BINANCE_CLOCK='0', TG_BOT_TOKEN='', MODEL_DIR=os.path.join(ROOT, 'models')))
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)
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
sig_chg = -(FUND0[i - 23:i + 1].sum(0) - FUND0[i - 47:i - 23].sum(0)); w_chg_res = q_res(np.where(mu_res & np.isfinite(sig_chg), sig_chg, np.nan), 0.2)
check('T12b fund_change weights == research', np.allclose(S.fund_change(panel, mu), w_chg_res, atol=1e-12), f'n {int((w_chg_res != 0).sum())}')
sig_lvl = -FUND0[i - 71:i + 1].sum(0); w_lvl_res = q_res(np.where(mu_res & np.isfinite(sig_lvl), sig_lvl, np.nan), 0.2)
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
check('T15b secrets excluded', not any(k in cfg1 for k in ('BYBIT_API_KEY', 'BYBIT_API_SECRET', 'TG_BOT_TOKEN', 'STATUS_TOKEN')))
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

print(f'\n{sum(1 for _, ok in res if ok)}/{len(res)} passed')
