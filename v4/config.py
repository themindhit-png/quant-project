#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PROP-SLEEVES v4 — configuration: environment variables + prop-firm rule presets.
Every firm parameter can be overridden by an explicit env var (same name)."""
import os

VERSION = '4.0.0'


def _env(name, default, cast=str):
    v = os.environ.get(name)
    if v is None or v == '':
        return default
    try:
        return cast(v)
    except Exception:
        return default


def _bool(name, default=False):
    v = os.environ.get(name)
    return default if v is None else v.strip().lower() in ('1', 'true', 'yes', 'on')


# ---------------------------------------------------------------- firm presets
# total_dd_mode / daily_dd_mode: 'static' (from initial balance / day-start equity) or 'trailing'
# (from equity high-water mark / intraday equity peak). Amounts are % of INITIAL balance.
FIRM_PRESETS = {
    # HyroTrader Two-Step, Standard: verified on hyrotrader.com FAQ (Sep 2026). Daily DD trailing from the
    # day's equity peak incl. unrealised; max DD trailing from equity high (their blog); 40% profit
    # distribution rule during evaluation; realised loss per trade <= 3% of initial; funded: notional <= 2x
    # initial, margin <= 25% initial; low-cap exposure <= 5% initial.
    'hyrotrader_2step': dict(targets=[10.0, 5.0], total_dd=10.0, daily_dd=5.0, total_dd_mode='trailing',
                             daily_dd_mode='trailing', consistency=40.0, min_days=[5, 5], per_trade_loss=3.0,
                             notional_x=2.0, margin_pct=25.0, lowcap_pct=5.0, lowcap_turn=5e6),
    # Swing upgrade: static daily and (per HyroTrader comparison pages) static max DD. VERIFY IN WRITING.
    'hyrotrader_2step_swing': dict(targets=[10.0, 5.0], total_dd=10.0, daily_dd=5.0, total_dd_mode='static',
                                   daily_dd_mode='static', consistency=40.0, min_days=[5, 5], per_trade_loss=3.0,
                                   notional_x=2.0, margin_pct=25.0, lowcap_pct=5.0, lowcap_turn=5e6),
    'hyrotrader_1step': dict(targets=[10.0], total_dd=6.0, daily_dd=4.0, total_dd_mode='trailing',
                             daily_dd_mode='trailing', consistency=40.0, min_days=[5], per_trade_loss=3.0,
                             notional_x=2.0, margin_pct=25.0, lowcap_pct=5.0, lowcap_turn=5e6),
    # Mubite Two-Step (challengeRules, Sep 2026): 10% (8% with add-on) + 5%, max DD 8% static from starting
    # balance, daily 5% (static), min 10 + 3 trading days, 3% realised loss per trade (of equity at open),
    # funded: 2x per position / 3x cumulative of initial. No consistency rule found on official pages.
    'mubite_2step': dict(targets=[10.0, 5.0], total_dd=8.0, daily_dd=5.0, total_dd_mode='static',
                         daily_dd_mode='static', consistency=0.0, min_days=[10, 3], per_trade_loss=3.0,
                         notional_x=3.0, position_x=2.0, margin_pct=100.0, lowcap_pct=100.0, lowcap_turn=0.0),
    'mubite_2step_8': dict(targets=[8.0, 5.0], total_dd=8.0, daily_dd=5.0, total_dd_mode='static',
                           daily_dd_mode='static', consistency=0.0, min_days=[10, 3], per_trade_loss=3.0,
                           notional_x=3.0, position_x=2.0, margin_pct=100.0, lowcap_pct=100.0, lowcap_turn=0.0),
    # Mubite with both add-ons: Easier Target (8%) + Extra 2% Drawdown Room (10% static) — +40% fee (~$154 for $10k)
    'mubite_8_dd10': dict(targets=[8.0, 5.0], total_dd=10.0, daily_dd=5.0, total_dd_mode='static',
                          daily_dd_mode='static', consistency=0.0, min_days=[10, 3], per_trade_loss=3.0,
                          notional_x=3.0, position_x=2.0, margin_pct=100.0, lowcap_pct=100.0, lowcap_turn=0.0),
    # HyroTrader 1-step with the Swing upgrade (static daily DD from day-start balance; max-loss mode not published -> trailing kept)
    'hyrotrader_1step_swing': dict(targets=[10.0], total_dd=6.0, daily_dd=4.0, total_dd_mode='trailing',
                                   daily_dd_mode='static', consistency=40.0, min_days=[5], per_trade_loss=3.0,
                                   notional_x=2.0, margin_pct=25.0, lowcap_pct=5.0, lowcap_turn=5e6),
    # Plutus Trade Base "Bybit / Freedom" (plutustradebase.com/bybit, 2026-09-09): challenge 5% target, 5% static max loss, no daily
    # cap, 7 profitable days (modelled as min trading days); funded: 4% trailing DD, no daily cap. Bots/EAs allowed. Entity thin.
    'ptb_bybit': dict(targets=[5.0], total_dd=5.0, daily_dd=100.0, total_dd_mode='static',
                      daily_dd_mode='static', consistency=0.0, min_days=[7], per_trade_loss=0.0,
                      notional_x=10.0, position_x=10.0, margin_pct=100.0, lowcap_pct=100.0, lowcap_turn=0.0),
    'ptb_bybit_funded': dict(targets=[5.0], total_dd=4.0, daily_dd=100.0, total_dd_mode='trailing',
                             daily_dd_mode='static', consistency=0.0, min_days=[7], per_trade_loss=0.0,
                             notional_x=10.0, position_x=10.0, margin_pct=100.0, lowcap_pct=100.0, lowcap_turn=0.0),
    # OWN account (no firm): the only floors are self-imposed — a 25% trailing drawdown stop (total) and no daily halt;
    # use with MODE=funded, NOTIONAL_BASE=equity, higher VT_TARGET_DVOL (see PLAN §5). Funding accrues on a real account.
    'personal': dict(targets=[1e9], total_dd=25.0, daily_dd=100.0, total_dd_mode='trailing',
                     daily_dd_mode='static', consistency=0.0, min_days=[0], per_trade_loss=0.0,
                     notional_x=5.0, position_x=5.0, margin_pct=100.0, lowcap_pct=100.0, lowcap_turn=0.0),
}

FIRM = _env('FIRM', 'hyrotrader_2step').strip().lower()
if FIRM not in FIRM_PRESETS:
    # an unknown preset must fail loudly: trading with default (wrong) firm limits is worse than not trading
    raise SystemExit(f'FATAL: unknown FIRM={FIRM!r}; valid presets: {sorted(FIRM_PRESETS)}')
_P = dict(FIRM_PRESETS[FIRM])
RULES = dict(
    targets=[float(x) for x in _env('TARGETS', ','.join(str(t) for t in _P['targets'])).split(',')],
    total_dd=_env('TOTAL_DD_PCT', _P['total_dd'], float),
    daily_dd=_env('DAILY_DD_PCT', _P['daily_dd'], float),
    total_dd_mode=_env('TOTAL_DD_MODE', _P['total_dd_mode']),
    daily_dd_mode=_env('DAILY_DD_MODE', _P['daily_dd_mode']),
    consistency=_env('CONSISTENCY_PCT', _P['consistency'], float),
    min_days=[int(x) for x in _env('MIN_DAYS', ','.join(str(t) for t in _P['min_days'])).split(',')],
    per_trade_loss=_env('PER_TRADE_LOSS_PCT', _P['per_trade_loss'], float),
    notional_x=_env('NOTIONAL_X', _P['notional_x'], float),
    position_x=_env('POSITION_X', _P.get('position_x', 2.0), float),
    margin_pct=_env('MARGIN_PCT', _P['margin_pct'], float),
    lowcap_pct=_env('LOWCAP_PCT', _P['lowcap_pct'], float),
    lowcap_turn=_env('LOWCAP_TURN', _P['lowcap_turn'], float),
)

# ---------------------------------------------------------------- account / mode
MODE = _env('MODE', 'challenge').strip().lower()          # challenge | funded
PHASE = _env('PHASE', 1, int)                                # 1..len(targets)
ACCOUNT_SIZE = _env('ACCOUNT_SIZE', 10000.0, float)          # INITIAL balance of the prop account
PHASE_BASE = _env('PHASE_BASE', 0.0, float)                  # 0 = snapshot equity at first cycle of the phase
TARGET_BUFFER_PCT = _env('TARGET_BUFFER_PCT', 0.30, float)   # aim above target to cover closing costs

# ---------------------------------------------------------------- risk engine
# exposure multiplier on the vol-targeted book. MC (exp10b): static-DD firms -> 1.0, trailing-DD -> 0.75
BASE_SCALE = _env('BASE_SCALE', 1.0 if RULES['total_dd_mode'] == 'static' else 0.75, float)
VT_TARGET_DVOL = _env('VT_TARGET_DVOL', 0.006, float)   # daily vol target of the unscaled book
VT_MAX_LEV = _env('VT_MAX_LEV', 3.0, float)
VT_WIN_D = _env('VT_WIN_D', 30, int)
# K = BASE * min(K_MAX/BASE, buffer / (CPPI_FRAC * dd_allowance)); with K_MAX == BASE the policy is the classic
# capped CPPI (profits do not raise risk); with K_MAX > BASE it is CONVEX: exposure grows with the cushion
# above the firm floor (faster when winning, smaller when losing). CPPI_FRAC=1.0 => K=BASE exactly at start.
CPPI_FRAC = _env('CPPI_FRAC', 0.5, float)
K_MAX = _env('K_MAX', 0.0, float)                  # 0 = same as BASE_SCALE (flat cap)
BASE_SCALE_P2 = _env('BASE_SCALE_P2', 0.0, float)  # optional separate base for phase 2 (0 = same)
DAILY_THROTTLE_PCT = _env('DAILY_THROTTLE_PCT', 1.0, float)   # day loss (of initial) at which K halves
DAILY_HALT_FRAC = _env('DAILY_HALT_FRAC', 0.5, float)          # flatten when day loss >= this fraction of firm daily limit
# latch a total halt when the buffer to the firm floor falls below this fraction of the DD allowance.
# CPPI already shrinks exposure near the floor, so a small guard (5%) avoids locking a recoverable account
# (harness: 10% latched the hyrotrader_2step trailing account in 2022-Q1 at -8.5% while the firm limit is -10%).
FLOOR_GUARD_FRAC = _env('FLOOR_GUARD_FRAC', 0.05, float)
# challenge only: when the buffer to the firm floor <= ABANDON_FRAC * allowance, flatten, latch and notify —
# statistically it is faster to buy a new challenge than to crawl out with a tiny K (exp10h). 0 = off.
ABANDON_FRAC = _env('ABANDON_FRAC', 0.0, float)
SAFETY_MARGIN = _env('SAFETY_MARGIN', 0.85, float)              # use at most this fraction of firm notional/margin limits
# base for the firm's notional / margin / per-position limits: 'initial' (firm rules are written on the initial balance)
# or 'equity' (research-style comparison: limits scale with the account; harness --pure uses this)
NOTIONAL_BASE = _env('NOTIONAL_BASE', 'initial').strip().lower()
MAX_GROSS_X_EQ = _env('MAX_GROSS_X_EQ', 2.0, float)             # gross notional <= this x equity
POS_CAP_PCT = _env('POS_CAP_PCT', 2.0, float)                   # long cap % equity
SHORT_CAP_PCT = _env('SHORT_CAP_PCT', 1.5, float)               # short cap % equity
MAJOR_CAP_PCT = _env('MAJOR_CAP_PCT', 30.0, float)              # cap for hedge majors
STOP_SHORT_PCT = _env('STOP_SHORT_PCT', 60.0, float)            # insurance stop: close a short if mark >= entry*(1+60%) (exp13b: least harmful)
STOP_COOLDOWN_H = _env('STOP_COOLDOWN_H', 48, int)
RISK_AFTER_TARGET = _env('RISK_AFTER_TARGET', 0.35, float)

# ---------------------------------------------------------------- strategy
SLEEVES = [s.strip() for s in _env('SLEEVES', 'listing,core,ml3,ml7').split(',') if s.strip()]
SLEEVE_SCALES = {k: float(v) for k, v in (kv.split(':') for kv in _env('SLEEVE_SCALES', 'listing:1,core:1,ml3:1,ml7:1').split(','))}
# intraday sleeve: 'ml8' in SLEEVES -> book rebalances every 8h (01:10 / 09:10 / 17:10 UTC); the daily sleeves are
# recomputed at the daily hour and HELD in between. Research (exp19/19b): Sharpe 1.54 -> ~1.8, corr(daily, 8h) = 0.24.
REBAL_EVERY_H = _env('REBAL_EVERY_H', 8 if 'ml8' in SLEEVES else 24, int)
ML8_TOP_FRAC = _env('ML8_TOP_FRAC', 0.30, float)
# ml8 universe: names >= ML8_MIN_AGE_D old (exp20b: young names are a jump lottery for an 8h book) and inverse-vol
# weights inside each side (ML8_INV_VOL=1, vol = std of hourly returns over ML8_VOL_H closed bars). Research (exp20h,
# fixed VT): 5-sleeve book Sharpe 1.67-1.91 across the smooth x weight grid vs 1.20-1.26 for the daily book.
ML8_MIN_AGE_D = _env('ML8_MIN_AGE_D', 180, int)
ML8_INV_VOL = _env('ML8_INV_VOL', 1, int)
ML8_VOL_H = _env('ML8_VOL_H', 168, int)
VT_SMOOTH = _env('VT_SMOOTH', 0.0, float)          # EMA on the vol-target leverage (0 = none)
# book-beta constraint (Opus review): beta of the combined weights to BTC (30d hourly betas); the EXCESS beyond ±BETA_CAP is
# hedged with BTC. -1 = off, 0 = full neutralisation (research exp21a/22c: +0.05..+0.2 Sharpe, lower turnover), 0.1 = soft cap.
BETA_CAP = _env('BETA_CAP', -1.0, float)
BETA_WIN_H = _env('BETA_WIN_H', 720, int)
# live per-sleeve attribution (daily PnL split by signed weight shares) and the PRE-REGISTERED ml8 kill rule (PREREG.md):
# trailing ML8_KILL_WIN_D-day Sharpe of the ml8 contribution <= 0 -> multiplier 1.0 -> 0.5; <= ML8_KILL_HARD -> 0.0;
# restore one step after ML8_KILL_RESTORE_D days of contribution Sharpe > 0.5; at most one change per ML8_KILL_MIN_GAP_D days.
ATTRIB_SLEEVES = _env('ATTRIB_SLEEVES', 1, int)
ML8_KILL_RULE = _env('ML8_KILL_RULE', 1, int)
# pre-registered values (PREREG.md rule 1): 180d window, halve at <= 0, off at <= -1.0, restore one step after 60d > 0.5.
# harness (signed attribution): no rule 2.47, this rule 2.28, the 90d/-0.5 rule 1.95 (too trigger-happy in 2023).
ML8_KILL_WIN_D = _env('ML8_KILL_WIN_D', 180, int)
ML8_KILL_HARD = _env('ML8_KILL_HARD', 0.0, float)         # v2: net-of-cost shadow Sharpe <= 0 -> sleeve off
ML8_KILL_MIN_GAP_D = _env('ML8_KILL_MIN_GAP_D', 30, int)
ML8_KILL_RESTORE_D = _env('ML8_KILL_RESTORE_D', 60, int)
ML8_BASE_SCALE = SLEEVE_SCALES.get('ml8', 2.0)
# funding sleeves (exp24a/e): 'fchg' = contrarian to the 24h funding change (daily, held), 'fcarry' = 72h funding-level carry (8h)
FUND_TOP_FRAC = _env('FUND_TOP_FRAC', 0.20, float)


def effective_config():
    """Every non-secret configuration value that changes behaviour (strategy + risk + execution), for the pre-registration
    hash (PREREG.md). Secrets, connection settings and model bytes are excluded; models are hashed separately."""
    import hashlib, json
    skip = {'BYBIT_API_KEY', 'BYBIT_API_SECRET', 'TG_BOT_TOKEN', 'TG_CHAT_ID', 'STATUS_TOKEN', 'BYBIT_BASE_URL', 'STATE_FILE', 'MODEL_DIR', 'DRY_RUN', 'KILL_SWITCH',
            'RESET_TOTAL_STOP', 'EQ_PEAK', 'PHASE_BASE', 'EXPECTED_CONFIG_HASH', 'FIRM_PRESETS', 'DEFAULT_BLACKLIST'}
    out = {}
    for k, v in sorted(globals().items()):
        if k.startswith('_') or k in skip or not k.isupper() or callable(v):
            continue
        if isinstance(v, (set, frozenset)):
            v = sorted(v)
        try:
            json.dumps(v); out[k] = v
        except TypeError:
            out[k] = str(v)
    h = hashlib.sha256(json.dumps(out, sort_keys=True, default=str).encode()).hexdigest()[:16]
    return out, h


# pre-registration guard: refuse to start when the effective strategy/risk configuration differs from the registered hash
EXPECTED_CONFIG_HASH = _env('EXPECTED_CONFIG_HASH', '').strip()
SIG_MIN_DISTINCT = _env('SIG_MIN_DISTINCT', 20, int)     # a sleeve signal with fewer distinct finite values is treated as broken (no weights)
# low-cap rule (HyroTrader: market cap < $100M): threshold and whether to fetch CoinGecko market caps (else turnover proxy)
LOWCAP_MCAP_USD = _env('LOWCAP_MCAP_USD', 1e8, float)
USE_MCAP = _env('USE_MCAP', 1, int)
LOWCAP_YOUNG_D = _env('LOWCAP_YOUNG_D', 90, int)         # without market caps: names younger than this count as low-cap
# ml8 kill rule v2 (Opus round 4): measured on the SHADOW contribution NET of estimated trading costs (turnover x bps)
ML8_KILL_SOFT = _env('ML8_KILL_SOFT', 0.5, float)        # 180d net Sharpe <= SOFT -> multiplier x0.5 ; <= HARD -> 0
ML8_KILL_RESTORE_SHARPE = _env('ML8_KILL_RESTORE_SHARPE', 1.0, float)   # restore one step after RESTORE_D days above this
ML8_COST_BPS_EST = _env('ML8_COST_BPS_EST', 5.0, float)  # blended cost per unit of shadow turnover (fees 70% maker + slippage)
# exchange-side stop loss on every position (some firms require a Bybit TP/SL order at trade open): distance in % of entry
REQUIRE_SL = _env('REQUIRE_SL', 0, int)     # HyroTrader FAQ (2026-09-09): "setting a stop-loss is not mandatory" -> off by default; switch on if a firm requires it
SL_PCT_LONG = _env('SL_PCT_LONG', 50.0, float)          # long: stop at entry x (1 - 50%)
SL_PCT_SHORT = _env('SL_PCT_SHORT', 60.0, float)        # short: stop at entry x (1 + 60%) (= STOP_SHORT_PCT default)
# own account: wider caps so a 1.0-1.5%/day vol target is reachable without clipping the gross (Opus round 4); env still overrides
if FIRM == 'personal':
    MAX_GROSS_X_EQ = _env('MAX_GROSS_X_EQ', 3.5, float)
    POS_CAP_PCT = _env('POS_CAP_PCT', 3.0, float)
    SHORT_CAP_PCT = _env('SHORT_CAP_PCT', 2.0, float)
# funding diagnostics (PREREG rule 6): daily count/sum of funding settlements from the account's transaction log and a
# Bybit-vs-Binance 24h funding comparison on the largest names (basis of the carry book vs the research exchange)
FUNDING_CHECK = _env('FUNDING_CHECK', 1, int)
FUNDING_ALERT_D = _env('FUNDING_ALERT_D', 14, int)       # days without any settlement -> 'funding not accrued' alert (book F)
# state durability: second copy of the state file + periodic push of the state JSON to Telegram (restore: STATE_JSON_B64 env)
STATE_BACKUP_FILE = _env('STATE_BACKUP_FILE', '')
STATE_PUSH_H = _env('STATE_PUSH_H', 6.0, float)
MAJORS = [s.strip() for s in _env('MAJORS', 'BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,XRPUSDT').split(',')]
UNI_MIN_TURN = _env('UNI_MIN_TURN', 1e7, float)        # core/ML universe min 24h turnover (USDT)
UNI_MIN_AGE_D = _env('UNI_MIN_AGE_D', 30, int)
# daily ML sleeves: separate minimum age (exp21c/21d: age-free models on names >= 90d beat the 30d/age-feature models)
ML_MIN_AGE_D = _env('ML_MIN_AGE_D', UNI_MIN_AGE_D, int)
# features excluded from the daily ML models (must match the training spec of models/ml_h{3,7}.txt; e.g. 'age_d')
ML_DROP_FEATS = [f.strip() for f in _env('ML_DROP_FEATS', '').split(',') if f.strip()]
# model file suffix: '' -> models/ml_h3.txt (age features, uni 30d); '_noage' -> models/ml_h3_noage.txt (exp6b_noage_final;
# requires ML_DROP_FEATS=age_d and ML_MIN_AGE_D=90 — the bot refuses to start on a mismatch, see signals.Sleeves)
ML_MODEL_SUFFIX = _env('ML_MODEL_SUFFIX', '').strip()
UNI_TOP_CORE = _env('UNI_TOP_CORE', 100, int)
UNI_TOP_ML = _env('UNI_TOP_ML', 150, int)
# universe hysteresis (held names stay eligible while rank < top+extra). exp14_gap: hysteresis +40 cut Sharpe
# 1.54 -> 0.80 (keeps names whose liquidity collapsed) -> default 0 = none, as in the research engine
UNI_EXIT_EXTRA = _env('UNI_EXIT_EXTRA', 0, int)
CORE_LB_H = 336
CORE_TOP_FRAC = 0.20
ML_TOP_FRAC = 0.30
LIST_AGE_MIN_D = _env('LIST_AGE_MIN_D', 7, int)
LIST_AGE_MAX_D = _env('LIST_AGE_MAX_D', 90, int)
LIST_MIN_TURN = _env('LIST_MIN_TURN', 2e7, float)
LIST_MAX_NAMES = _env('LIST_MAX_NAMES', 30, int)
LIST_W_NAME = _env('LIST_W_NAME', 0.05, float)
LIST_PUMP_EXCL = _env('LIST_PUMP_EXCL', 0.30, float)
REBAL_BAND = _env('REBAL_BAND', 0.30, float)
# EMA smoothing per rebalance step. exp19/19b: at 8h cadence heavier per-step smoothing (0.794) made the combined
# book erratic (stale intraday targets); 0.5 per step was stable (Sharpe 1.72/1.78/1.51 for ml8 weights 1/2/4).
SMOOTH = _env('SMOOTH', 0.5, float)
MIN_TRADE_USDT = _env('MIN_TRADE_USDT', max(15.0, 0.0015 * ACCOUNT_SIZE), float)
# Daily rebalance at 01:10 UTC: the last closed bar is then the 00:00-01:00 bar — exactly the decision bar
# used in research and in ML training (features/targets aligned to the bar with open_time 00:00 UTC).
REBAL_HOUR_UTC = _env('REBAL_HOUR_UTC', 1, int)
REBAL_MINUTE = _env('REBAL_MINUTE', 10, int)
WATCHDOG_SEC = _env('WATCHDOG_SEC', 60, int)
MODEL_DIR = _env('MODEL_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models'))
USE_BINANCE_CLOCK = _bool('USE_BINANCE_CLOCK', True)   # also use Binance onboardDate for token age (if reachable)

# ---------------------------------------------------------------- execution
MAKER_ATTEMPTS = _env('MAKER_ATTEMPTS', 3, int)         # PostOnly re-quotes before falling back to market
MAKER_WAIT_S = _env('MAKER_WAIT_S', 90, int)            # seconds between re-quotes (3 attempts -> <= ~5 min to the market fallback)
CAND_POOL = _env('CAND_POOL', 350, int)                 # top-N by turnover fed to the universes (research: every name with turnover >= 1e7)
SET_LEVERAGE = _env('SET_LEVERAGE', 5.0, float)
DRY_RUN = _bool('DRY_RUN', False)
KILL_SWITCH = _bool('KILL_SWITCH', False)

# ---------------------------------------------------------------- infra
API_KEY = _env('BYBIT_API_KEY', '')
API_SECRET = _env('BYBIT_API_SECRET', '')
BASE_URL = _env('BYBIT_BASE_URL', 'https://api-demo.bybit.com')
TG_TOKEN = _env('TG_BOT_TOKEN', '')
TG_CHAT = _env('TG_CHAT_ID', '')
STATUS_TOKEN = _env('STATUS_TOKEN', '')
PORT = _env('PORT', 10000, int)
STATE_FILE = _env('STATE_FILE', f'/tmp/propsleeves_{FIRM}_{MODE}_p{PHASE}_{int(ACCOUNT_SIZE)}.json')
BLACKLIST_PATTERNS = ('STOCK', 'XAU', 'XAG', 'USDC', 'EURUSD', '-')
DEFAULT_BLACKLIST = {
    'AAOIUSDT', 'AAPLUSDT', 'ALABUSDT', 'AMZNUSDT', 'AXTIUSDT', 'BEUSDT', 'BMNRUSDT', 'BSPUSDT', 'BZUSDT', 'CBRSUSDT',
    'CLUSDT', 'COHRUSDT', 'CRCLUSDT', 'CRDOUSDT', 'CRWVUSDT', 'DELLUSDT', 'DRAMUSDT', 'EWYUSDT', 'GLWUSDT', 'GOOGLUSDT',
    'HPEUSDT', 'INTCUSDT', 'IRENUSDT', 'KORUUSDT', 'LITEUSDT', 'METAUSDT', 'MRVLUSDT', 'MSFTUSDT', 'MUUSDT', 'NBISUSDT',
    'NOKIAUSDT', 'NVDAUSDT', 'ONDSUSDT', 'ORCLUSDT', 'PLTRUSDT', 'QQQUSDT', 'RKLBUSDT', 'SAMSUNGUSDT', 'SKHYNIXUSDT',
    'SKHYUSDT', 'SNDKUSDT', 'SNXXUSDT', 'SOXLUSDT', 'SPCXUSDT', 'TQQQUSDT', 'TSLAUSDT', 'XAGUSDT', 'XAUUSDT',
    'DJTUSDT', 'MRNAUSDT', 'MARAUSDT', 'IONQUSDT', 'PDDUSDT', 'TEMUSDT', 'MRKUSDT', 'SKUUUSDT', 'SKDDUSDT', 'CXMTUSDT',
    'UNITREEUSDT', 'RAMUSDT', 'AMDSTOCKUSDT', 'APPSTOCKUSDT', 'CATSTOCKUSDT', 'CVXSTOCKUSDT',
}
BLACKLIST = DEFAULT_BLACKLIST | {s.strip().upper() for s in _env('BLACKLIST_SYMBOLS', '').split(',') if s.strip()}
