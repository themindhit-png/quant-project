#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Раз в сутки (таймер propsleeves-daily, 00:30 UTC): копия файла состояния каждого бота в /opt/propsleeves/backup/
(хранится 30 дней; только состояние, без ключей) и сводка в Telegram."""
import os, sys, glob, shutil, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import opslib as L

KEEP_D = 30


def main():
    bdir = os.path.join(L.APP, 'backup'); os.makedirs(bdir, exist_ok=True)
    day = L.now_utc().strftime('%Y%m%d'); parts = []
    for u in L.bot_units():
        env = L.read_env(L.env_file(u)); sf = env.get('STATE_FILE')
        if sf and os.path.exists(sf):
            shutil.copy2(sf, os.path.join(bdir, f'{L.short(u)}-{day}.json'))
        parts.append(L.summary(u))
    for f in glob.glob(os.path.join(bdir, '*.json')):
        if time.time() - os.path.getmtime(f) > KEEP_D * 86400:
            os.remove(f)
    L.tg_send('📋 Сводка за сутки\n\n' + ('\n\n'.join(parts) if parts else 'ботов не найдено'))


if __name__ == '__main__':
    main()
