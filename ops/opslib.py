#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared helpers for the VPS services (ops/tg_control.py, ops/daily.py): find the bot units (systemd propsleeves*.service), read their
env files, query /status, build the owner's summary, send Telegram messages. Bot code (v4/) is not imported: these services keep working
when a bot is stopped or broken."""
import os, json, subprocess, time
from datetime import datetime, timezone
import requests

APP = os.environ.get('APP', '/opt/propsleeves')
SERVICE_UNITS = {'propsleeves-tg', 'propsleeves-daily', 'propsleeves-collect'}


def bot_units():
    """Bot units installed on this host: 'propsleeves' (prop account) first, then 'propsleeves-<name>' (own account etc.)."""
    try:
        out = subprocess.run(['systemctl', 'list-unit-files', 'propsleeves*.service', '--no-legend'], capture_output=True, text=True).stdout
    except FileNotFoundError:
        return []
    units = [ln.split()[0].removesuffix('.service') for ln in out.splitlines() if ln.strip()]
    units = [u for u in units if u not in SERVICE_UNITS]
    return sorted(units, key=lambda u: (u != 'propsleeves', u))


def short(unit):
    return 'prop' if unit == 'propsleeves' else unit.removeprefix('propsleeves-')


def unit_for(name):
    """'prop' / '' -> propsleeves; 'personal' -> propsleeves-personal (if installed)."""
    name = (name or 'prop').strip().lower()
    for u in bot_units():
        if short(u) == name:
            return u
    return None


def env_file(unit):
    out = subprocess.run(['systemctl', 'show', '-p', 'EnvironmentFiles', '--value', unit], capture_output=True, text=True).stdout
    return out.split()[0].lstrip('-') if out.strip() else None


def read_env(path):
    env = {}
    if not path or not os.path.exists(path):
        return env
    for line in open(path, encoding='utf-8'):
        line = line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        k, v = line.split('=', 1); k = k.strip().removeprefix('export ').strip()
        env[k] = v.strip().strip('"\'')
    return env


def pause_file(env):
    return env.get('PAUSE_FILE') or (env.get('STATE_FILE', '') + '.pause')


def status(env, timeout=5):
    try:
        r = requests.get(f'http://127.0.0.1:{env.get("PORT", "8080")}/status', headers={'X-Status-Token': env.get('STATUS_TOKEN', '')}, timeout=timeout)
        return r.json()
    except Exception:
        return None


def is_active(unit):
    return subprocess.run(['systemctl', 'is-active', '--quiet', unit]).returncode == 0


def summary(unit):
    """Owner-readable state of one bot (Russian)."""
    env = read_env(env_file(unit)); st = status(env) or {}
    size = float(env.get('ACCOUNT_SIZE') or 0); eq = st.get('equity')
    stage = env.get('MODE', '?') + (f" фаза {env.get('PHASE')}" if env.get('MODE') == 'challenge' else '')
    lines = [f"[{short(unit)}] {env.get('FIRM', '?')} {stage}, счёт {size:,.0f}"]
    if not is_active(unit):
        lines.append('⛔ служба не запущена')
    if os.path.exists(pause_file(env)):
        lines.append('⏸ ПАУЗА (позиции закрыты, бот ждёт /resume)')
    if eq is None:
        lines.append('нет ответа /status (бот запускается или остановлен)')
        return '\n'.join(lines)
    floor = st.get('total_floor')
    lines.append(f"equity {eq:,.2f} ({(eq / size - 1) * 100 if size else 0:+.2f} %)")
    if floor is not None:
        lines.append(f"до пола фирмы {eq - floor:,.0f} USDT")
    if env.get('MODE') == 'challenge':
        lines.append(f"торговых дней {st.get('trading_days', '?')}, прибыльных {st.get('profit_days', '?')}")
    lines.append(f"K {st.get('k', '?')}, позиций {st.get('n_positions', '?')} / целей {st.get('n_targets', '?')}, "
                 f"последний ребаланс {str(st.get('last_cycle_utc', '?'))[:16]} UTC")
    if st.get('errors'):
        lines.append(f"ошибки ордеров: {len(st['errors'])}")
    return '\n'.join(lines)


def tg_conf():
    """Telegram token/chat from the main bot's env (all bots share one Telegram bot)."""
    for u in bot_units():
        env = read_env(env_file(u))
        if env.get('TG_BOT_TOKEN') and env.get('TG_CHAT_ID'):
            return env['TG_BOT_TOKEN'], str(env['TG_CHAT_ID'])
    return None, None


def tg_send(text, token=None, chat=None):
    if token is None:
        token, chat = tg_conf()
    if not token:
        print(text); return
    try:
        requests.post(f'https://api.telegram.org/bot{token}/sendMessage', json={'chat_id': chat, 'text': text[:4000]}, timeout=15)
    except Exception as e:
        print(f'tg fail: {e}')


def now_utc():
    return datetime.now(timezone.utc)
