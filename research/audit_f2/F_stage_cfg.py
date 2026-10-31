#!/usr/bin/env python3
"""Audit F2 (agent F): assemble each stage's env exactly as ops/stage.sh does (base.env minus local keys + profile + stage lines),
evaluate v4/config.effective_config() in a clean subprocess and print the hash and the strategy/risk values.
Read-only. Run: python3 research/audit_f2/F_stage_cfg.py"""
import os, re, sys, json, subprocess
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOCAL = re.compile(r'^(BYBIT_|TG_|STATUS_|STATE_|PORT|DRY_RUN|MODEL_DIR|EXPECTED_CONFIG_HASH)')

def read(p):
    out = []
    for line in open(os.path.join(ROOT, 'evns', p), encoding='utf-8'):
        line = line.strip()
        if re.match(r'^[A-Z0-9_]+=', line):
            k, v = line.split('=', 1); out.append((k, v.strip().strip('"\'')))
    return out

def stage_env(stage, size):
    prof, mode, phase = dict(trial=('mubite_challenge.env', 'challenge', 1), phase1=('mubite_challenge.env', 'challenge', 1),
                             phase2=('mubite_challenge.env', 'challenge', 2), funded=('mubite_funded.env', 'funded', 1),
                             personal=('personal.env', 'funded', 1))[stage]
    env = {}
    for k, v in read('base.env'):
        if not LOCAL.match(k) and k not in ('ACCOUNT_SIZE', 'MODE', 'PHASE', 'FIRM'):
            env[k] = v
    for k, v in read(prof):
        if k not in ('MODE', 'PHASE'):
            env[k] = v
    env['BYBIT_BASE_URL'] = 'https://api.bybit.com' if stage == 'personal' else 'https://api-demo.bybit.com'
    env.update(MODE=mode, PHASE=str(phase), ACCOUNT_SIZE=str(size))
    if stage == 'trial':
        env.update(TARGETS='5', MIN_DAYS='2', PROFIT_DAYS='0')
    env['STATE_FILE'] = f'/opt/propsleeves/state/{stage}_{size}_X.json'; env['STATE_BACKUP_FILE'] = env['STATE_FILE'] + '.b'
    return env

CODE = 'import json, config as C; o, h = C.effective_config(); print(json.dumps(dict(hash=h, cfg=o), default=str))'
res = {}
for stage, size in (('trial', 100000), ('trial', 10000), ('phase1', 10000), ('phase2', 10000), ('phase1', 200000), ('funded', 10000),
                    ('funded', 200000), ('personal', 5000)):
    env = stage_env(stage, size); env['PATH'] = os.environ.get('PATH', '/usr/bin:/bin')
    r = subprocess.run([sys.executable, '-c', CODE], cwd=os.path.join(ROOT, 'v4'), env=env, capture_output=True, text=True)
    if r.returncode:
        print(stage, size, 'ERROR', r.stderr[-800:]); continue
    out = json.loads(r.stdout.strip().splitlines()[-1]); res[f'{stage}_{size}'] = out
    print(f'{stage:<8} {size:>7}  hash {out["hash"]}  keys {len(out["cfg"])}')
keys = ['SLEEVES', 'SLEEVE_SCALES', 'FUND_TOP_FRAC', 'ML_TOP_FRAC', 'ML8_TOP_FRAC', 'CORE_TOP_FRAC', 'CORE_LB_H', 'UNI_TOP_CORE', 'UNI_TOP_ML',
        'UNI_MIN_AGE_D', 'ML_MIN_AGE_D', 'ML8_MIN_AGE_D', 'UNI_MIN_TURN', 'UNI_EXIT_EXTRA', 'ML8_INV_VOL', 'ML8_VOL_H', 'ML_DROP_FEATS', 'ML_MODEL_SUFFIX',
        'LIST_AGE_MIN_D', 'LIST_AGE_MAX_D', 'LIST_MIN_TURN', 'LIST_MAX_NAMES', 'LIST_W_NAME', 'LIST_PUMP_EXCL', 'MAJORS',
        'REBAL_BAND', 'SMOOTH', 'REBAL_EVERY_H', 'REBAL_HOUR_UTC', 'REBAL_MINUTE', 'VT_TARGET_DVOL', 'VT_MAX_LEV', 'VT_WIN_D', 'VT_SMOOTH',
        'BETA_CAP', 'HIVOL_MULT', 'DD_HIVOL_BUF', 'DD_HIVOL_MULT', 'NOTIONAL_BASE', 'POS_CAP_PCT', 'SHORT_CAP_PCT', 'MAJOR_CAP_PCT', 'MAX_GROSS_X_EQ',
        'SAFETY_MARGIN', 'SET_LEVERAGE', 'STOP_SHORT_PCT', 'STOP_COOLDOWN_H', 'MIN_TRADE_USDT', 'BASE_SCALE', 'BASE_SCALE_P2', 'K_MAX', 'K_MIN_FRAC',
        'CPPI_FRAC', 'ABANDON_FRAC', 'DAILY_THROTTLE_PCT', 'DAILY_HALT_FRAC', 'FLOOR_GUARD_FRAC', 'RISK_AFTER_TARGET', 'TARGET_BUFFER_PCT',
        'ML8_KILL_RULE', 'ML8_KILL_WIN_D', 'ML8_KILL_SOFT', 'ML8_KILL_HARD', 'ML8_KILL_RESTORE_SHARPE', 'ML8_COST_BPS_EST', 'ML8_KILL_MIN_GAP_D',
        'ML8_KILL_RESTORE_D', 'MAKER_ATTEMPTS', 'MAKER_WAIT_S', 'CAND_POOL', 'LOWCAP_YOUNG_D', 'FUNDING_CHECK', 'FUNDING_ALERT_D', 'REQUIRE_SL',
        'DAY_MARKER_FRAC', 'FIRM', 'MODE', 'PHASE', 'ACCOUNT_SIZE', 'RULES']
print()
cols = list(res)
print('key | ' + ' | '.join(cols))
for k in keys:
    vals = [json.dumps(res[c]['cfg'].get(k, '—'), default=str) for c in cols]
    if k == 'RULES':
        for rk in ('targets', 'total_dd', 'daily_dd', 'min_days', 'profit_days', 'notional_x', 'position_x', 'per_trade_loss'):
            print(f'RULES.{rk} | ' + ' | '.join(json.dumps(res[c]['cfg']['RULES'].get(rk)) for c in cols))
        continue
    print(f'{k} | ' + ' | '.join(vals))
