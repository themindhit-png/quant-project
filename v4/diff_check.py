#!/usr/bin/env python3
"""Direct comparison of bot combined weights/targets vs the research engine (exp13b P0 functions)
on a series of dates, using the harness fakes. Run: python3 v4/diff_check.py"""
import os, sys, json, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.argv = [sys.argv[0], '--pure', '--ml-panels', '--start', '2022-01-01', '--mode', 'funded']
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)
import harness_v4 as H
from bt import quantile_ls
from strategies import core_signal
import config as C

d = H.d; N = H.N; syms = H.syms; sidx = H.sidx
P3, P7 = H.PANELS[3], H.PANELS[7]
MAJ5 = [s for s in C.MAJORS if s in syms]
jm = np.array([sidx[s] for s in MAJ5])
launch_h = np.array([H.instr_real[s]['launch'] / 3.6e6 if s in H.instr_real and H.instr_real[s].get('launch') else np.nan for s in syms])


def research_w(i):
    age = d.age[i].astype(float)
    age = np.maximum(age, np.where(np.isfinite(launch_h), d.hh[i] - launch_h, -np.inf))
    t = d.t24[i]
    cand = (age >= 24 * 7) & (age <= 24 * 90) & d.valid[i] & np.isfinite(t) & (t >= 2e7) & (d.age[i] >= 72)
    cand[jm] = False
    r24 = d.cff[i] / d.cff[i - 24] - 1; cand &= ~(r24 > 0.30)
    idx = np.flatnonzero(cand); wl = np.zeros(N)
    if len(idx):
        idx = idx[np.argsort(-t[idx])][:30]; ws = min(1.0 / len(idx), 0.05); wl[idx] = -ws; wl[jm] += ws * len(idx) / len(jm)
    d.prev_mask = None
    mc = d.universe(i, min_age_h=720, min_turn=1e7, top_n=100)
    wc = quantile_ls(core_signal(d, i, mc, 336), 0.2)
    d.prev_mask = None
    mm = d.universe(i, min_age_h=720, min_turn=1e7, top_n=150)
    w3 = quantile_ls(np.where(mm & np.isfinite(P3[i]), P3[i], np.nan), 0.3)
    w7 = quantile_ls(np.where(mm & np.isfinite(P7[i]), P7[i], np.nan), 0.3)
    return (wl + wc + w3 + w7) / 4, dict(listing=wl, core=wc, ml3=w3, ml7=w7), dict(core_uni=int(mc.sum()), ml_uni=int(mm.sum()))


api = H.FakeApi(); md = H.FakeMD(api)
bot = H.SimBot(api, md=md, mem=H.botstate.Memory())
bot.md.refresh_instruments(); bot.instr = bot.md.instr
for date in ['2022-02-01', '2022-03-15', '2022-05-01', '2022-09-01', '2023-03-01', '2024-01-01', '2025-01-01', '2026-06-01']:
    i = int(d.idx.searchsorted(pd.Timestamp(date, tz='UTC')))
    H.SIM['i'] = i
    now_ms = int(H.now_dt().timestamp() * 1000)
    tickers = api.tickers()
    cands = [s for s, t in tickers.items() if s in bot.instr and s.endswith('USDT') and s not in C.BLACKLIST
             and not any(p in s for p in C.BLACKLIST_PATTERNS) and t['turnover24h'] >= min(C.UNI_MIN_TURN, C.LIST_MIN_TURN) * 0.8]
    cands.sort(key=lambda s: -tickers[s]['turnover24h'])
    young = [s for s in cands if tickers[s]['turnover24h'] >= C.LIST_MIN_TURN and 0 <= md.age_hours(s, now_ms, 'listing') <= C.LIST_AGE_MAX_D * 24]
    symbols = list(dict.fromkeys(C.MAJORS + cands[:max(C.UNI_TOP_ML + 60, 200)] + young))
    last_closed = md.last_closed_hour_ms(now_ms)
    panel = md.panel([s for s in symbols if s in md.bars], last_closed); panel['delist'] = set()
    ps = panel['symbols']
    age_h = np.array([md.age_hours(s, now_ms, 'listing') for s in ps]); age_ml = np.array([md.age_hours(s, now_ms, 'ml') for s in ps])
    bot.sleeves.uni.prev = {}
    wb, parts_b, info_b = bot.sleeves.combined(panel, age_h, set(), age_ml=age_ml)
    wr, parts_r, info_r = research_w(i)
    # align research vector to panel symbols
    js = np.array([sidx[s] for s in ps])
    wr_p = wr[js]
    missing = np.abs(wr).sum() - np.abs(wr_p).sum()
    print(f'{date}: panel {len(ps)} syms | bot gross {np.abs(wb).sum():.3f} net {wb.sum():+.3f} | research gross {np.abs(wr).sum():.3f} (outside panel {missing:.3f}) | '
          f'max|diff| {np.abs(wb - wr_p).max():.4f} corr {np.corrcoef(wb, wr_p)[0,1]:.3f} | uni bot {info_b} res {info_r}')
    for k in ('listing', 'core', 'ml3', 'ml7'):
        b = parts_b.get(k, np.zeros(len(ps))); r = parts_r[k][js]
        print(f'      {k:8s} bot n {int((b != 0).sum()):3d} gross {np.abs(b).sum():.3f} | res n {int((parts_r[k] != 0).sum()):3d} gross {np.abs(parts_r[k]).sum():.3f} | max|diff| {np.abs(b - r).max():.4f}')
