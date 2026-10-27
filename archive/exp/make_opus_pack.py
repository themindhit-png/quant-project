#!/usr/bin/env python3
"""Build the review packs for Opus 5.
CORE (fits one context, ~150-180 KB): 01 REPORT, 02 OPUS_REPLY, 03 PLAN, 04 CODE core (risk.py, config.py, signals.py,
harness FakeApi/pure excerpt), 05 LOGS key lines only.  FULL: + full main.py/harness_v4.py, research engine and scripts, full logs."""
import os, re, zipfile, glob
ROOT = os.path.dirname(os.path.abspath(__file__))
def read(p):
    try: return open(os.path.join(ROOT, p), encoding='utf-8', errors='replace').read()
    except FileNotFoundError: return f'(missing: {p})\n'
def cat(files, title, note=''):
    parts = [f'# {title}\n\n{note}\n']
    for f in files:
        if isinstance(f, tuple):                      # (path, first_line, last_line) excerpt
            p, a, b = f; lines = read(p).splitlines(); body = '\n'.join(lines[a - 1:b]); f = f'{p} (lines {a}-{b})'
        else: body = read(f).rstrip()
        parts.append(f'\n\n---\n## FILE: {f}\n\n```{"python" if f.endswith(".py") or ".py (" in f else ""}\n{body}\n```\n')
    return ''.join(parts)
KEY = re.compile(r'(Sh\s|Sharpe|phase1|both phases|alive|^===|IC|corr|>>>|^\s+\d{4}: |gross/equity|FATAL|median|config|^\[|^K\d|^N\d|^D\d|^R\d|^T\d|^  [a-z].*\|)')
def log_lines(path, key_only, max_lines):
    txt = [l for l in read(path).splitlines() if not l.startswith('  20')]
    if key_only: txt = [l for l in txt if KEY.search(l)]
    if len(txt) > max_lines: txt = txt[:max_lines // 2] + ['... (truncated) ...'] + txt[-max_lines // 2:]
    return '\n'.join(l[:260] for l in txt)
def logs(files, title, key_only, max_lines):
    parts = [f'# {title}\n\nVerbatim log excerpts (progress lines removed{"; key lines only" if key_only else ""}).\n']
    for f in files:
        for p in sorted(glob.glob(os.path.join(ROOT, f))):
            rel = os.path.relpath(p, ROOT); parts.append(f'\n\n---\n## LOG: {rel}\n\n```\n{log_lines(rel, key_only, max_lines)}\n```\n')
    return ''.join(parts)
LOGS = ['logs/hF2_pure_eq_v3.log', 'logs/exp26.log', 'logs/exp27.log', 'logs/exp28.log', 'logs/exp29.log', 'logs/exp24f_f64.log', 'logs/exp25b_f64.log',
        'logs/exp23a.log', 'logs/exp24a.log', 'logs/exp24b.log', 'logs/exp24c.log', 'logs/exp24d.log', 'logs/exp24e.log', 'logs/exp24f.log', 'logs/exp25.log', 'logs/exp25b.log',
        'logs/hF2_pure_eq_nokill.log', 'logs/hF2_pure_eq_final.log', 'logs/hF2_pure_eq_kill90_v2.log', 'logs/hF2_pure_eq_taker_final.log', 'logs/hF2_pure_eq_lag1_final.log',
        'logs/rollF2_*.log', 'logs/rollD2_*.log',
        'logs/exp21a.log', 'logs/exp21b.log', 'logs/exp21e.log', 'logs/exp21c.log', 'logs/exp21d.log', 'logs/exp22.log', 'logs/exp22b.log', 'logs/exp22c.log', 'logs/exp22d.log', 'logs/exp22e.log',
        'logs/exp20h.log', 'logs/exp20i.log', 'logs/exp6b_noage.log', 'logs/chain3.log', 'logs/h5_maker33.log', 'logs/hd_maker33.log', 'logs/h5_taker.log', 'logs/h5_nb_initial.log', 'logs/h5_nb_equity.log',
        'logs/hd_nb_equity.log', 'logs/hd_swing_funded.log', 'logs/hd_swing_pure.log', 'logs/h5_seed2.log', 'logs/hK14_pure.log', 'logs/hK3_pure.log', 'logs/hF_pure.log', 'logs/hF_pure_s2.log',
        'logs/roll5_flat10.log', 'logs/roll5_bold20.log', 'logs/roll5_funded125.log', 'logs/rollK14_bold20.log', 'logs/rollK14_flat10.log', 'logs/rollK14_funded125.log',
        'logs/rollF_bold20.log', 'logs/rollF_flat10.log', 'logs/rollF_funded125.log', 'logs/rollK3_funded125.log']
def build(name, files):
    out = os.path.join(ROOT, name); os.makedirs(out, exist_ok=True); tot = 0
    for fn, content in files.items():
        p = os.path.join(out, fn); open(p, 'w', encoding='utf-8').write(content); sz = os.path.getsize(p); tot += sz
        print(f'  {fn:<26} {sz/1024:7.1f} KB')
    zp = os.path.join(ROOT, name + '.zip')
    with zipfile.ZipFile(zp, 'w', zipfile.ZIP_DEFLATED) as z:
        for fn in files: z.write(os.path.join(out, fn), fn)
    print(f'{name}: TOTAL {tot/1024:.0f} KB (~{tot/2.6/1000:.0f}k tokens for Russian text) -> {zp} ({os.path.getsize(zp)/1024:.0f} KB)')
core = {
    '00_HOWITWORKS.md': read('HOWITWORKS.md'),
    '01_REPORT.md': read('REPORT.md'), '02_OPUS_REPLY.md': read('OPUS_REPLY.md'), '03_PLAN.md': read('PLAN.md') + '\n\n---\n\n' + read('PREREG.md'),
    '04_CODE_core.md': cat(['v4/risk.py', 'v4/config.py', 'v4/signals.py', ('v4/harness_v4.py', 1, 130), ('v4/main.py', 100, 200)],
                           'Bot v4 — core code for review', 'risk.py (direct vol target, CPPI, floors, caps, notional limits), config.py (firm presets, env), signals.py (sleeves, beta cap, ML8), harness excerpt (FakeApi fills/equity, --pure), main.py rebalance excerpt (guards). Full files in the FULL pack.'),
    '05_LOGS_key.md': logs(LOGS, 'Key log lines (fixed vol target unless stated)', True, 90),
}
full = dict(core)
full['04_CODE_bot_full.md'] = cat(['v4/risk.py', 'v4/config.py', 'v4/signals.py', 'v4/main.py', 'v4/harness_v4.py', 'v4/execution.py', 'v4/README.md'], 'Bot v4 — full code')
full['06_CODE_research.md'] = cat(['bt.py', 'strategies.py', 'betautil.py', 'exp21a_opus_tests.py', 'exp21b_2021.py', 'exp21c_ml_noage.py', 'exp21d_noage_portfolio.py', 'exp21e_lag_fix.py',
                                   'exp22_squeeze.py', 'exp22b_simplify.py', 'exp22c_combos.py', 'exp22d_cost_robust.py', 'exp22e_noage_combos.py', 'exp6b_noage_final.py', 'restart_from_rolling.py',
                                   'exp23a_f_checks.py', 'exp24a_new_alpha.py', 'exp24b_slow_ml.py', 'exp24c_ml8_noage.py', 'exp24d_dynamic.py', 'exp24e_funding_sleeves.py', 'exp24f_final_combos.py',
                                   'exp25_unlocks.py', 'exp25b_unlock_filter.py', 'exp26_opus3.py', 'exp27_firm_rules.py', 'prereg.py'],
                                  'Research engine and the review experiments', 'bt.py run(): rebalance loop, VT direct/recursive, caps, neutrality, dust, costs, stops, attribution.')
full['07_LOGS_full.md'] = logs(LOGS, 'Full logs', False, 400)
full['08_FIRMS_research.md'] = cat(['data/firms/firms_audit.md', 'data/firms/firms_reputation.md', 'data/firms/firms_round2.md', 'data/firms/raw/mubite_challengeRules.txt'],
                                   'Prop-firm research (three passes: rules audit, reputation, primary-source verification)',
                                   'Раунд 1 — правила/страны (Ledger), раунд 1 — репутация/выплаты (Sentinel; часть выводов не подтвердилась первоисточниками — см. раунд 2), раунд 2 — цитаты с сайтов фирм (Beacon).')
print('CORE pack:'); build('opus_pack_core', core)
print('FULL pack:'); build('opus_pack_full', full)
