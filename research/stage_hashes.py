#!/usr/bin/env python3
"""Config hashes (EXPECTED_CONFIG_HASH) for every stage / account size, with the env assembled exactly like ops/stage.sh
(evns/base.env minus local keys + stage profile + stage lines), evaluated by v4/config.effective_config() in a clean process.
Used for PREREG registrations (reg. 4). Run from the repo root: python3 research/stage_hashes.py"""
import os, re, sys, json, subprocess
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCAL = re.compile(r'^(BYBIT_|TG_|STATUS_|STATE_|PORT|DRY_RUN|MODEL_DIR|EXPECTED_CONFIG_HASH)')
PROF = dict(trial=('mubite_challenge.env', 'challenge', 1), phase1=('mubite_challenge.env', 'challenge', 1), phase2=('mubite_challenge.env', 'challenge', 2),
            funded=('mubite_funded.env', 'funded', 1), personal=('personal.env', 'funded', 1))
COMBOS = [('trial', 100000), ('trial', 10000), ('phase1', 10000), ('phase2', 10000), ('phase1', 100000), ('phase2', 100000),
          ('phase1', 200000), ('phase2', 200000), ('funded', 10000), ('funded', 100000), ('funded', 200000), ('personal', 5000)]

def read(p):
    for line in open(os.path.join(ROOT, 'evns', p), encoding='utf-8'):
        line = line.strip()
        if re.match(r'^[A-Z0-9_]+=', line):
            k, v = line.split('=', 1); yield k, v.strip().strip('"\'')

def stage_env(stage, size):
    prof, mode, phase = PROF[stage]
    env = {k: v for k, v in read('base.env') if not LOCAL.match(k) and k not in ('ACCOUNT_SIZE', 'MODE', 'PHASE', 'FIRM')}
    env.update({k: v for k, v in read(prof) if k not in ('MODE', 'PHASE')})
    env['BYBIT_BASE_URL'] = 'https://api.bybit.com' if stage == 'personal' else 'https://api-demo.bybit.com'
    env.update(MODE=mode, PHASE=str(phase), ACCOUNT_SIZE=str(size))
    if stage == 'trial':
        env.update(TARGETS='5', MIN_DAYS='2', PROFIT_DAYS='0')
    env['STATE_FILE'] = f'/opt/propsleeves/state/{stage}_{size}_X.json'; env['PATH'] = os.environ.get('PATH', '/usr/bin:/bin')
    return env

if __name__ == '__main__':
    code = 'import json, config as C; o, h = C.effective_config(); print(json.dumps([h, len(o)]))'
    for stage, size in COMBOS:
        r = subprocess.run([sys.executable, '-c', code], cwd=os.path.join(ROOT, 'v4'), env=stage_env(stage, size), capture_output=True, text=True)
        if r.returncode:
            print(stage, size, 'ERROR', r.stderr[-500:]); continue
        h, n = json.loads(r.stdout.strip().splitlines()[-1])
        print(f'{stage:<9}{size:>8}  {h}  ({n} keys)')
