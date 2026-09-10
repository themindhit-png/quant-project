#!/usr/bin/env python3
"""Monte-Carlo simulator of prop-firm challenge / funded outcomes from a strategy's HOURLY
return series (block bootstrap preserves intraday clustering and fat tails).

Firm rule presets (verified Sep 2026):
  hyro_2step : phase targets 10%/5%, max DD 10%? (set), daily DD 5% TRAILING intraday from day's
               peak equity incl. unrealised, amount fixed = 5% of initial; consistency: best day <= 40%
               of total net result (evaluation only); min 5 trading days per phase.
  hyro_1step : target 10%, daily 4%, ...
  mubite_2step: 10% (8% add-on) / 5%, max DD 8% static from initial, daily 5% (static from day start
               balance), min 10+3 days.
"""
import numpy as np, pandas as pd

RULES = {
    'hyro_2step': dict(targets=[0.10, 0.05], max_dd=0.10, daily_dd=0.05, daily_mode='trailing',
                       consistency=0.40, min_days=[5, 5], trailing_total=True),
    # Swing upgrade: per HyroTrader comparison pages "6%/10% max drawdown (trailing or static, optional Swing
    # static upgrade)" -> treat both daily and total DD as static from initial balance. VERIFY with support.
    'hyro_2step_swing': dict(targets=[0.10, 0.05], max_dd=0.10, daily_dd=0.05, daily_mode='static',
                             consistency=0.40, min_days=[5, 5], trailing_total=False),
    'hyro_1step': dict(targets=[0.10], max_dd=0.06, daily_dd=0.04, daily_mode='trailing',
                       consistency=0.40, min_days=[5], trailing_total=True),
    'mubite_2step': dict(targets=[0.08, 0.05], max_dd=0.08, daily_dd=0.05, daily_mode='static',
                         consistency=None, min_days=[10, 3], trailing_total=False),
    'mubite_2step_10': dict(targets=[0.10, 0.05], max_dd=0.08, daily_dd=0.05, daily_mode='static',
                            consistency=None, min_days=[10, 3], trailing_total=False),
}


def block_bootstrap_paths(hr, n_paths, n_hours, block_days=5, rng=None):
    """hr: hourly returns (np.array, fraction). Returns (n_paths, n_hours) sampled by day-blocks."""
    rng = rng or np.random.default_rng(0)
    T = len(hr) // 24 * 24
    days = hr[:T].reshape(-1, 24)
    nd = len(days)
    need_days = int(np.ceil(n_hours / 24)) + block_days
    out = np.empty((n_paths, need_days * 24))
    for p in range(n_paths):
        pos = 0
        while pos < need_days:
            s = rng.integers(0, nd - block_days)
            blk = days[s:s + block_days].reshape(-1)
            k = min(block_days * 24, (need_days - pos) * 24)
            out[p, pos * 24:pos * 24 + k] = blk[:k]
            pos += block_days
    return out[:, :n_hours]


def simulate_phase(path, rule, target, min_days, scale=1.0, start_eq=1.0):
    """Run one phase on an hourly path (returns as fraction of equity, scaled by `scale`).
    Returns (outcome, hours_used, equity_end). outcome in {'pass','fail_daily','fail_total','timeout'}."""
    eq = start_eq
    peak_total = start_eq
    floor_total = start_eq * (1 - rule['max_dd'])
    day_start = eq; day_peak = eq
    daily_amt = start_eq * rule['daily_dd']         # fixed amount from initial (HyroTrader/Mubite)
    day_pnls = []
    cur_day_pnl = 0.0
    n_hours = len(path)
    for h in range(n_hours):
        if h % 24 == 0 and h > 0:
            day_pnls.append(cur_day_pnl); cur_day_pnl = 0.0
            day_start = eq; day_peak = eq
        r = path[h] * scale
        pnl = eq * r
        eq += pnl; cur_day_pnl += pnl
        day_peak = max(day_peak, eq)
        # daily rule
        if rule['daily_mode'] == 'trailing':
            if day_peak - eq >= daily_amt:
                return 'fail_daily', h, eq
        else:
            if day_start - eq >= daily_amt:
                return 'fail_daily', h, eq
        # total rule
        if rule.get('trailing_total'):
            peak_total = max(peak_total, eq)
            floor_total = max(floor_total, peak_total * (1 - rule['max_dd']))  # trailing from equity high
        if eq <= floor_total:
            return 'fail_total', h, eq
        # target
        if eq >= start_eq * (1 + target):
            days_done = len(day_pnls) + 1
            if days_done >= min_days:
                if rule.get('consistency'):
                    dp = np.array(day_pnls + [cur_day_pnl])
                    tot = dp.sum()
                    if tot > 0 and dp.max() > rule['consistency'] * tot:
                        continue          # keep trading until consistency satisfied
                return 'pass', h, eq
    return 'timeout', n_hours, eq


def run_challenge(hr, rule_name, scale=1.0, n_paths=2000, max_days=365, block_days=5, seed=0,
                  trailing_total_override=None):
    rule = dict(RULES[rule_name])
    if trailing_total_override is not None:
        rule['trailing_total'] = trailing_total_override
    rng = np.random.default_rng(seed)
    n_hours = max_days * 24 * len(rule['targets'])
    paths = block_bootstrap_paths(np.asarray(hr, float), n_paths, n_hours, block_days, rng)
    res = []
    for p in range(n_paths):
        h0 = 0; ok = True; outcome = 'pass'
        for ph, (tg, md) in enumerate(zip(rule['targets'], rule['min_days'])):
            seg = paths[p, h0:h0 + max_days * 24]
            o, used, _ = simulate_phase(seg, rule, tg, md, scale)
            h0 += used + 1
            if o != 'pass':
                outcome = o; break
        res.append((outcome, h0 / 24))
    df = pd.DataFrame(res, columns=['outcome', 'days'])
    summ = df.outcome.value_counts(normalize=True).to_dict()
    summ['median_days_pass'] = float(df[df.outcome == 'pass'].days.median()) if (df.outcome == 'pass').any() else np.nan
    summ['p90_days_pass'] = float(df[df.outcome == 'pass'].days.quantile(0.9)) if (df.outcome == 'pass').any() else np.nan
    return summ, df


def run_funded(hr, rule_name, scale=1.0, n_paths=1000, days=365, block_days=5, seed=1,
               payout_every_days=30, split=0.9):
    """Funded account: same DD rules (no target/consistency). Pays out all profit above initial
    every payout_every_days. Returns survival prob and mean payout per year (as fraction of initial)."""
    rule = dict(RULES[rule_name])
    rng = np.random.default_rng(seed)
    paths = block_bootstrap_paths(np.asarray(hr, float), n_paths, days * 24, block_days, rng)
    out = []
    for p in range(n_paths):
        eq = 1.0; paid = 0.0; alive = True
        day_start = eq; day_peak = eq; daily_amt = rule['daily_dd']
        floor_total = 1 - rule['max_dd']; peak_total = 1.0
        for h in range(days * 24):
            if h % 24 == 0 and h > 0:
                day_start = eq; day_peak = eq
                d = h // 24
                if d % payout_every_days == 0 and eq > 1.0:
                    paid += (eq - 1.0) * split; eq = 1.0
                    day_start = eq; day_peak = eq
            r = paths[p, h] * scale
            eq *= (1 + r)
            day_peak = max(day_peak, eq)
            if rule['daily_mode'] == 'trailing':
                if day_peak - eq >= daily_amt: alive = False; break
            elif day_start - eq >= daily_amt: alive = False; break
            if rule.get('trailing_total'):
                peak_total = max(peak_total, eq)
                floor_total = max(floor_total, peak_total * (1 - rule['max_dd']))
            if eq <= floor_total: alive = False; break
        out.append((alive, paid, h / 24))
    df = pd.DataFrame(out, columns=['alive', 'paid', 'days'])
    return dict(survival=df.alive.mean(), mean_paid=df.paid.mean(), median_paid=df.paid.median(),
                mean_life_days=df.days.mean()), df


if __name__ == '__main__':
    # synthetic sanity check: Sharpe 2 strategy, 0.6% daily vol
    rng = np.random.default_rng(0)
    dv = 0.006; mu = 2.0 * dv / np.sqrt(365)
    hr = rng.standard_t(4, size=24 * 365 * 4) * dv / np.sqrt(24) / np.sqrt(2) + mu / 24
    for rn in RULES:
        s, _ = run_challenge(hr, rn, n_paths=500)
        print(rn, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items()})
