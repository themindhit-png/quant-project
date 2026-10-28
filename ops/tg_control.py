#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Управление ботами из Telegram (служба propsleeves-tg, ставит ops/install.sh). Принимает команды ТОЛЬКО из чата TG_CHAT_ID:
  /status [personal]   — сводка (без аргумента — все боты)
  /pause  [personal]   — пауза перед выплатой / выводом: бот закрывает все позиции и ждёт (то же, что ops/bot.sh pause)
  /resume [personal]   — снять паузу: изменение баланса учтётся как вывод, бот продолжит
  /help
Без аргумента команды относятся к проп-боту (служба propsleeves)."""
import os, sys, time, json
import requests
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import opslib as L

HELP = ('Команды:\n/status — сводка по всем ботам\n/pause — пауза проп-бота перед выплатой (закрыть всё)\n/resume — снять паузу после выплаты\n'
        'Для своего счёта добавьте слово personal: /pause personal')
OFFSET_FILE = os.path.join(L.APP, 'tg_offset')


def handle(text):
    parts = text.strip().split()
    if not parts:
        return None
    cmd = parts[0].split('@')[0].lower(); arg = parts[1] if len(parts) > 1 else ''
    if cmd in ('/start', '/help'):
        return HELP
    if cmd == '/status':
        units = [L.unit_for(arg)] if arg else L.bot_units()
        units = [u for u in units if u]
        return '\n\n'.join(L.summary(u) for u in units) if units else 'ботов на сервере не найдено'
    if cmd in ('/pause', '/resume'):
        unit = L.unit_for(arg)
        if not unit:
            return f'нет бота «{arg or "prop"}». Есть: {", ".join(L.short(u) for u in L.bot_units())}'
        pf = L.pause_file(L.read_env(L.env_file(unit)))
        if not pf or pf == '.pause':
            return 'в настройках бота нет STATE_FILE — пауза невозможна, напиши в чат проекта'
        if cmd == '/pause':
            open(pf, 'a').close()
            return f'⏸ [{L.short(unit)}] пауза включена. Через 1-2 мин придёт «все позиции закрыты» — после этого запрашивай выплату.'
        if os.path.exists(pf):
            os.remove(pf)
        return f'▶️ [{L.short(unit)}] пауза снята. Изменение баланса учтётся как вывод, бот продолжит торговать.'
    return 'не понял. ' + HELP


def main():
    token, chat = L.tg_conf()
    if not token:
        print('нет TG_BOT_TOKEN / TG_CHAT_ID в env ботов — жду 5 мин'); time.sleep(300); sys.exit(1)
    try:
        offset = int(open(OFFSET_FILE).read().strip())
    except Exception:
        offset = 0
    while True:
        try:
            r = requests.get(f'https://api.telegram.org/bot{token}/getUpdates', params={'offset': offset, 'timeout': 50}, timeout=60).json()
        except Exception as e:
            print(f'getUpdates: {e}'); time.sleep(10); continue
        for upd in r.get('result', []):
            offset = upd['update_id'] + 1
            msg = upd.get('message') or {}
            if str((msg.get('chat') or {}).get('id')) != chat or not msg.get('text'):
                continue                                   # чужие чаты игнорируются
            try:
                reply = handle(msg['text'])
            except Exception as e:
                reply = f'ошибка: {e}'
            if reply:
                L.tg_send(reply, token, chat)
        try:
            with open(OFFSET_FILE, 'w') as f:
                f.write(str(offset))
        except OSError:
            pass


if __name__ == '__main__':
    main()
