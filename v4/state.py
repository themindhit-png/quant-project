#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Persistence, logging and Telegram for PROP-SLEEVES v4."""
import os, sys, json, time, threading
from datetime import datetime, timezone
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C

LOCK = threading.Lock()
STATUS = {'version': C.VERSION, 'firm': C.FIRM, 'mode': C.MODE, 'phase': C.PHASE, 'account_size': C.ACCOUNT_SIZE}
_ALERTS = {}


def log(msg):
    print(f'[{datetime.now(timezone.utc):%Y-%m-%dT%H:%M:%SZ}] {msg}', flush=True)


def tg(text):
    if not C.TG_TOKEN or not C.TG_CHAT:
        return
    try:
        requests.post(f'https://api.telegram.org/bot{C.TG_TOKEN}/sendMessage',
                      json={'chat_id': C.TG_CHAT, 'text': text[:4000], 'disable_web_page_preview': True}, timeout=15)
    except Exception as e:
        log(f'TG fail: {e}')


def tg_once(key, text, hours=24.0):
    now = time.time()
    if now - _ALERTS.get(key, 0) >= hours * 3600:
        _ALERTS[key] = now
        tg(text)


class Memory(dict):
    """Persistent bot memory (JSON). Atomic save; conservative defaults on loss."""
    PERSIST = ('initial', 'eq_peak', 'day_date', 'day_anchor', 'day_peak', 'halted_day', 'halt_total', 'phase_base',
               'target_hit', 'trading_days', 'td_date', 'day_pnls', 'day_pnl_dates', 'lev', 'w_prev', 'eq_daily',
               'mult_daily', 'mult_day_sum', 'mult_day_n', 'clip_n', 'reb_n', 'flow_daily', 'flow_today', 'attrib_shadow_ml8',
               'fund_days', 'fund_settlements', 'fund_total', 'ml8_audit', 'ml8_audit_day', 'ml8_changes',
               'attrib_share', 'attrib_px', 'attrib_notional', 'attrib_day', 'sleeve_pnl_day', 'sleeve_pnl_hist', 'ml8_mult', 'ml8_mult_date',
               'cooldown', 'k_avg', 'last_rebal_date', 'last_rebal_slot', 'last_daily_date', 'held_w', 'last_targets', 'k_ref',
               'cycles', 'blocked', 'created', 'abandoned')

    def __init__(self):
        super().__init__()
        self.loaded = False
        self.update(created=time.time(), cycles=0, blocked={}, cooldown={}, last_targets={}, k_ref=None)

    def load(self):
        """Restore order: STATE_FILE -> STATE_BACKUP_FILE -> STATE_JSON_B64 env (paste of the last Telegram backup)."""
        srcs = [(C.STATE_FILE, 'file'), (C.STATE_BACKUP_FILE, 'backup')]
        for path, kind in srcs:
            try:
                if path and os.path.exists(path):
                    with open(path) as f:
                        d = json.load(f)
                    self.update({k: v for k, v in d.items() if k in self.PERSIST}); self.loaded = True
                    log(f'state restored from {path} ({kind}; created {datetime.fromtimestamp(d.get("created", 0), timezone.utc):%Y-%m-%d})')
                    return
            except Exception as e:
                log(f'state load fail ({path}): {e}')
        b64 = os.environ.get('STATE_JSON_B64', '').strip()
        if b64:
            try:
                import base64
                d = json.loads(base64.b64decode(b64).decode())
                self.update({k: v for k, v in d.items() if k in self.PERSIST}); self.loaded = True
                log('state restored from STATE_JSON_B64')
            except Exception as e:
                log(f'state load fail (STATE_JSON_B64): {e}')

    _last_push = 0.0

    def save(self):
        try:
            data = {k: self.get(k) for k in self.PERSIST}
            blob = json.dumps(data, default=float)
            for path in (C.STATE_FILE, C.STATE_BACKUP_FILE):
                if not path:
                    continue
                tmp = path + '.tmp'
                with open(tmp, 'w') as f:
                    f.write(blob)
                os.replace(tmp, path)
            if C.TG_TOKEN and C.TG_CHAT and C.STATE_PUSH_H > 0 and time.time() - Memory._last_push >= C.STATE_PUSH_H * 3600:
                Memory._last_push = time.time()
                threading.Thread(target=self._push, args=(blob,), daemon=True).start()
        except Exception as e:
            log(f'state save fail: {e}')

    @staticmethod
    def _push(blob):
        """Off-box copy of the state (Telegram document); restore via STATE_JSON_B64 if the host loses /tmp."""
        try:
            requests.post(f'https://api.telegram.org/bot{C.TG_TOKEN}/sendDocument', data={'chat_id': C.TG_CHAT, 'caption': f'state backup {datetime.now(timezone.utc):%Y-%m-%d %H:%M}Z'},
                          files={'document': ('state.json', blob.encode())}, timeout=30)
        except Exception as e:
            log(f'state push fail: {e}')


def set_status(**kw):
    with LOCK:
        STATUS.update(kw)


def get_status(authorized):
    with LOCK:
        st = dict(STATUS)
    if not authorized:
        return {k: st.get(k) for k in ('version', 'firm', 'mode', 'phase', 'cycles', 'last_cycle_utc', 'halt_total', 'target_hit', 'ok')}
    return st
