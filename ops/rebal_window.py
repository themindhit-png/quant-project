#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Wait until NOW is outside the rebalance window of a bot (F8): ops/deploy.sh and ops/stage.sh restart the bot only after this returns, so a restart
never cuts a rebalance in the middle (open PostOnly quotes, half a batch). Same slot logic as ops/retrain.py (rebal_slots / in_rebal_window:
from 3 min before a slot to 30 min after it, slots from the bot's env: reg. 5c = 01:01 and 13:01 UTC).
  python3 ops/rebal_window.py ENV_FILE [--max-wait-min 45] [--check]
Exit 0 = free (after waiting if needed); 3 = still inside the window after --max-wait-min (or --check and inside now)."""
import os, sys, time, argparse
from datetime import datetime, timezone
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import opslib as L
from retrain import rebal_slots, in_rebal_window


def wait_outside(env, max_wait_min=45, now_fn=None, sleep=time.sleep, say=print):
    """True when outside the window (waiting up to max_wait_min), False when still inside after that."""
    now_fn = now_fn or (lambda: datetime.now(timezone.utc))
    slots = rebal_slots(env); t0 = time.time(); said = False
    while in_rebal_window(now_fn(), slots):
        if not said:
            say(f'сейчас окно ребаланса (слоты UTC {[f"{s // 60:02d}:{s % 60:02d}" for s in slots]}, от -3 до +30 мин): жду, перезапуск только после него'); said = True
        if time.time() - t0 >= max_wait_min * 60:
            return False
        sleep(30)
    return True


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('env_file'); ap.add_argument('--max-wait-min', type=float, default=45); ap.add_argument('--check', action='store_true')
    A = ap.parse_args()
    env = L.read_env(A.env_file)
    ok = wait_outside(env, 0 if A.check else A.max_wait_min)
    print('вне окна ребаланса' if ok else 'внутри окна ребаланса')
    sys.exit(0 if ok else 3)
