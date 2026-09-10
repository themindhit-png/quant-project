#!/usr/bin/env python3
"""Experiment 10b: challenge/funded Monte-Carlo with a CPPI risk overlay and payout policies.
Overlay: exposure scale_t = base * min(1, buffer_t / (cppi * dd_allowance)), buffer_t = equity - floor_t
(floor = static or trailing per firm), plus optional daily throttle (halve after -1% day-to-date).
Unlimited time (max 1000 days per phase) -> report P(fail) and time-to-pass distribution.
Funded: payout policy 'all' (withdraw all profit monthly) vs 'keep_buffer' (withdraw only the part
above initial*(1+keep)) ; trailing vs static total DD."""
import sys, os, numpy as np, pandas as pd
from challenge_sim import block_bootstrap_paths, RULES

_cands = ['out_final_eq.csv', 'out_exp9c_eq.csv', 'out_exp9b_eq.csv']
src = sys.argv[1] if len(sys.argv) > 1 else next(c for c in _cands if os.path.exists(c))
eq = pd.read_csv(src, index_col=0, parse_dates=True).iloc[:, 0].dropna()
hr = eq.pct_change().dropna().values
print('source', src, 'hours', len(hr), 'ann ret %.1f%%' % (hr.mean() * 24 * 365 * 100), flush=True)


def phase(path, rule, target, min_days, base, cppi, start_eq=1.0, daily_throttle=True):
    eq = start_eq; peak = start_eq
    dd_amt = start_eq * rule['max_dd']
    floor = start_eq - dd_amt
    day_start = eq; day_peak = eq; daily_amt = start_eq * rule['daily_dd']
    day_pnls = []; cur = 0.0
    for h in range(len(path)):
        if h % 24 == 0 and h > 0:
            day_pnls.append(cur); cur = 0.0; day_start = eq; day_peak = eq
        buf = eq - floor
        k = base * min(1.0, max(0.0, buf) / (cppi * dd_amt)) if cppi > 0 else base
        if daily_throttle and (eq - day_start) / start_eq <= -0.01:
            k *= 0.5
        pnl = eq * path[h] * k
        eq += pnl; cur += pnl
        day_peak = max(day_peak, eq)
        if rule['daily_mode'] == 'trailing':
            if day_peak - eq >= daily_amt: return 'fail_daily', h
        elif day_start - eq >= daily_amt: return 'fail_daily', h
        if rule.get('trailing_total'):
            peak = max(peak, eq); floor = max(floor, peak - dd_amt)
        if eq <= floor: return 'fail_total', h
        if eq >= start_eq * (1 + target) and len(day_pnls) + 1 >= min_days:
            if rule.get('consistency'):
                dp = np.array(day_pnls + [cur]); tot = dp.sum()
                if tot > 0 and dp.max() > rule['consistency'] * tot:
                    continue
            return 'pass', h
    return 'timeout', len(path)


def challenge(rule_name, base, cppi, n_paths=1000, max_days=1000, seed=3, trailing=None):
    rule = dict(RULES[rule_name])
    if trailing is not None: rule['trailing_total'] = trailing
    rng = np.random.default_rng(seed)
    paths = block_bootstrap_paths(hr, n_paths, max_days * 24 * len(rule['targets']), 5, rng)
    out = []
    for p in range(n_paths):
        h0 = 0; res = 'pass'
        for tg, md in zip(rule['targets'], rule['min_days']):
            o, used = phase(paths[p, h0:h0 + max_days * 24], rule, tg, md, base, cppi)
            h0 += used + 1
            if o != 'pass': res = o; break
        out.append((res, h0 / 24))
    df = pd.DataFrame(out, columns=['o', 'days'])
    vc = df.o.value_counts(normalize=True)
    ps = df[df.o == 'pass'].days
    return dict(pass_=vc.get('pass', 0), fail_daily=vc.get('fail_daily', 0), fail_total=vc.get('fail_total', 0),
                timeout=vc.get('timeout', 0), med=ps.median() if len(ps) else np.nan, p75=ps.quantile(.75) if len(ps) else np.nan,
                p90=ps.quantile(.9) if len(ps) else np.nan)


def funded(rule_name, base, cppi, policy='all', keep=0.05, n_paths=600, days=365, seed=5, trailing=None, split=0.9):
    rule = dict(RULES[rule_name])
    if trailing is not None: rule['trailing_total'] = trailing
    rng = np.random.default_rng(seed)
    paths = block_bootstrap_paths(hr, n_paths, days * 24, 5, rng)
    out = []
    for p in range(n_paths):
        eq = 1.0; peak = 1.0; dd_amt = rule['max_dd']; floor = 1 - dd_amt; paid = 0.0; alive = True
        day_start = eq; day_peak = eq; daily_amt = rule['daily_dd']
        for h in range(days * 24):
            if h % 24 == 0 and h > 0:
                day_start = eq; day_peak = eq
                if (h // 24) % 30 == 0:
                    if policy == 'all' and eq > 1.0:
                        paid += (eq - 1.0) * split; eq = 1.0
                    elif policy == 'keep_buffer' and eq > 1.0 + keep:
                        paid += (eq - 1.0 - keep) * split; eq = 1.0 + keep
                    day_start = eq; day_peak = eq
            buf = eq - floor
            k = base * min(1.0, max(0.0, buf) / (cppi * dd_amt)) if cppi > 0 else base
            if (eq - day_start) <= -0.01: k *= 0.5
            eq *= (1 + paths[p, h] * k)
            day_peak = max(day_peak, eq)
            if rule['daily_mode'] == 'trailing':
                if day_peak - eq >= daily_amt: alive = False; break
            elif day_start - eq >= daily_amt: alive = False; break
            if rule.get('trailing_total'):
                peak = max(peak, eq); floor = max(floor, peak - dd_amt)
            if eq <= floor: alive = False; break
        out.append((alive, paid + (max(eq - 1.0, 0) * split if alive else 0), h / 24))
    df = pd.DataFrame(out, columns=['alive', 'paid', 'days'])
    return dict(surv=df.alive.mean(), paid_mean=df.paid.mean(), paid_med=df.paid.median())


rows = []
configs = [('hyro_2step', True, 'Hyro 2-step trailing'), ('hyro_2step', False, 'Hyro 2-step static(swing?)'),
           ('mubite_2step', False, 'Mubite 8%+5% static'), ('mubite_2step_10', False, 'Mubite 10%+5% static'),
           ('hyro_1step', True, 'Hyro 1-step trailing')]
for rn, tr, label in configs:
    for base in (0.75, 1.0, 1.5, 2.0):
        for cppi in (0.0, 0.5):
            c = challenge(rn, base, cppi, trailing=tr)
            f_all = funded(rn, base, cppi, 'all', trailing=tr)
            f_keep = funded(rn, base, cppi, 'keep_buffer', trailing=tr)
            row = dict(firm=label, base=base, cppi=cppi, **c, surv_all=f_all['surv'], paid_all=f_all['paid_mean'],
                       surv_keep=f_keep['surv'], paid_keep=f_keep['paid_mean'])
            rows.append(row)
            print(f'{label:<28} base {base:3.2f} cppi {cppi:3.1f} | pass {c["pass_"]*100:5.1f}% fail_d {c["fail_daily"]*100:4.1f}% fail_t {c["fail_total"]*100:4.1f}% '
                  f'timeout {c["timeout"]*100:4.1f}% | days med {c["med"]:5.0f} p75 {c["p75"]:5.0f} p90 {c["p90"]:5.0f} | funded1y: all surv {f_all["surv"]*100:5.1f}% paid {f_all["paid_mean"]*100:5.1f}% ; '
                  f'keep5% surv {f_keep["surv"]*100:5.1f}% paid {f_keep["paid_mean"]*100:5.1f}%', flush=True)
pd.DataFrame(rows).to_csv('out_exp10b_challenge.csv', index=False)
print('DONE')
