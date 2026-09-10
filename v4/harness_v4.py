#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bot-in-the-loop harness for PROP-SLEEVES v4: runs the REAL bot code (signals, risk, execution
planning/ordering, watchdog) against a fake exchange fed by the research panels (Binance hourly data),
daily rebalance at 00:xx UTC + hourly watchdog/scale pass. Compares to the research engine (bt.py).

Usage: python3 v4/harness_v4.py --start 2022-01-01 --end 2026-08-31 [--firm hyrotrader_2step] [--pure]
  --pure : disable firm floors/CPPI (BASE_SCALE 1, CPPI 0, no halts) to compare with bt.py's exp13 MAIN.
"""
import os, sys, time, json, argparse, math
import numpy as np, pandas as pd
from datetime import datetime, timezone, timedelta

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ap = argparse.ArgumentParser()
ap.add_argument('--start', default='2022-01-01'); ap.add_argument('--end', default='2026-08-31 23:00')
ap.add_argument('--firm', default='hyrotrader_2step'); ap.add_argument('--pure', action='store_true')
ap.add_argument('--maker-fill', type=float, default=0.7); ap.add_argument('--slip-scale', type=float, default=1.0)
ap.add_argument('--account', type=float, default=10000.0); ap.add_argument('--seed', type=int, default=1)
ap.add_argument('--mode', default='funded'); ap.add_argument('--verbose', action='store_true')
ap.add_argument('--ml-panels', action='store_true', help='use walk-forward prediction panels (out_exp6_pred_h*.npy) instead of final models (fair OOS)')
ap.add_argument('--panels-prefix', default='out_exp6_pred', help='prefix of prediction panels (e.g. out_exp16_pred, with --panels-suffix _k)')
ap.add_argument('--panels-suffix', default='')
ap.add_argument('--starts', default='', help='comma list of start dates for rolling challenge/funded simulations (one Data load)')
ap.add_argument('--phases', type=int, default=1, help='challenge: run phase 1 then phase 2 from the pass date')
ap.add_argument('--payout-days', type=int, default=0, help='funded: withdraw profit above initial every N days')
ap.add_argument('--keep-buffer', type=float, default=0.0, help='funded: keep this fraction of initial as cushion above initial before paying out')
ap.add_argument('--max-days', type=int, default=0, help='stop a run after N days (0 = to end of data)')
ap.add_argument('--exec-lag-h', type=int, default=0, help='execution delay: decide on bar i-lag (panel, clock), fill at bar i prices (Opus stress)')
A = ap.parse_args()

# ---- environment for the bot BEFORE importing config
os.environ.update(dict(FIRM=A.firm, MODE=A.mode, ACCOUNT_SIZE=str(A.account), BYBIT_API_KEY='x', BYBIT_API_SECRET='x',
                       MAKER_WAIT_S='0', USE_BINANCE_CLOCK='0', STATE_FILE=f'/tmp/harness_v4_{A.firm}_{A.mode}.json',
                       MODEL_DIR=os.path.join(ROOT, 'models'), TG_BOT_TOKEN='', SET_LEVERAGE='10'))
if A.pure:
    os.environ.update(dict(BASE_SCALE='1.0', CPPI_FRAC='0', DAILY_HALT_FRAC='100', DAILY_THROTTLE_PCT='100',
                           TOTAL_DD_PCT='90', DAILY_DD_PCT='90', TARGETS='1000'))
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)
import config as C
import execution, main as botmain, state as botstate
from bt import Data, slip_model, metrics

execution.time.sleep = lambda x: None          # no real waiting in simulation
botstate.tg = lambda *a, **k: None
botmain.tg = lambda *a, **k: None
botmain.tg_once = lambda *a, **k: None
if not A.verbose:
    botmain.log = lambda *a, **k: None
    execution_log = lambda *a, **k: None
else:
    execution_log = print

# ---- research data (with --ml-panels the Data window must match exp6: start 2021-01-01)
d = Data(start='2021-01-01' if A.ml_panels else A.start, end=A.end, min_turn_ever=5e6, load_hl=True, load_premium=True)
d.start_i = int(d.idx.searchsorted(pd.Timestamp(A.start, tz='UTC')))
T, N = d.ret.shape
syms = [str(s) for s in d.cols]
sidx = {s: j for j, s in enumerate(syms)}
instr_real = json.load(open(os.path.join(ROOT, 'instruments.json')))
SLIP = slip_model(scale=A.slip_scale)
rng = np.random.default_rng(A.seed)
SIM = dict(i=d.start_i, dec=d.start_i)          # i: fill/price bar; dec: decision bar (= i - exec lag)


def now_dt():
    return d.idx[SIM['dec']].to_pydatetime() + timedelta(hours=1, minutes=C.REBAL_MINUTE)


class FakeApi:
    """Fills at the close of the current bar; PostOnly fills with prob maker_fill at the touch (no slippage,
    maker fee 2 bps); market fills at close*(1 +- slip) with taker fee 5.5 bps. Funding applied hourly."""
    def __init__(self):
        self.wallet = float(A.account); self.qty = {}; self.entry = {}
        self.stat = dict(maker_tried=0, maker_filled=0, market=0, fees=0.0, slip=0.0, funding=0.0, turnover=0.0)
        self.instr_cache = None

    def px(self, s):
        j = sidx.get(s); v = d.cff[SIM['i'], j] if j is not None else np.nan
        return float(v) if np.isfinite(v) else 0.0

    def upnl(self):
        return sum(q * (self.px(s) - self.entry[s]) for s, q in self.qty.items() if q)

    def equity(self):
        return self.wallet + self.upnl()

    def wallet_info(self):
        return dict(equity=self.equity())

    def positions(self):
        out = {}
        for s, q in self.qty.items():
            if q == 0:
                continue
            m = self.px(s)
            out[s] = dict(qty=q, mark=m, entry=self.entry[s], upnl=q * (m - self.entry[s]), im=abs(q * m) / C.SET_LEVERAGE,
                          legs=[dict(qty=q, positionIdx=0, mark=m, entry=self.entry[s])])
        return out

    def tickers(self):
        i = SIM['i']; out = {}
        for s, j in sidx.items():
            p = d.cff[i, j]
            if not np.isfinite(p) or p <= 0 or not d.valid[i, j]:
                continue
            t = d.t24[i, j]
            out[s] = dict(last=float(p), mark=float(p), index=float(p), bid=float(p), ask=float(p),
                          turnover24h=float(t) if np.isfinite(t) else 0.0, funding=0.0, oi_value=0.0)
        return out

    def instruments(self):
        if self.instr_cache is None:
            out = {}
            for s in syms:
                m = instr_real.get(s)
                p0 = np.nanmedian(np.where(d.valid[:, sidx[s]], d.cff[:, sidx[s]], np.nan))
                step = m['qtyStep'] if m else (10 ** math.floor(math.log10(max(1.0 / max(p0, 1e-9), 1e-9))))
                out[s] = dict(qtyStep=step, minQty=(m['minQty'] if m else step), maxQty=1e12, minNotional=5.0,
                              tickSize=(m['tickSize'] if m else max(p0 * 1e-4, 1e-8)), launch=0, maxLeverage=50.0, fundingInterval=480)
            self.instr_cache = out
        return self.instr_cache

    def announcements_delisting(self):
        return set()

    def transaction_log_sum(self, start_ms):
        return 0.0

    def switch_one_way(self):
        return {}

    def set_leverage(self, *a, **k):
        return {}

    def cancel_all(self, *a, **k):
        return {}

    def open_orders(self):
        return []

    # fills
    def _fill(self, s, side, qty, ro, maker):
        p0 = self.px(s)
        if p0 <= 0 or qty <= 0:
            raise execution.ApiError(110001, 'no price')
        q0 = self.qty.get(s, 0.0)
        if ro:
            qty = min(qty, abs(q0))
            if qty <= 0:
                return
        if maker:
            self.stat['maker_tried'] += 1
            if rng.random() > A.maker_fill:
                return
            self.stat['maker_filled'] += 1
            px, fee_bps, sl = p0, 2.0, 0.0
        else:
            self.stat['market'] += 1
            sl = float(SLIP(d, SIM['i'])[sidx[s]])
            px = p0 * (1 + (sl if side == 'Buy' else -sl) / 1e4); fee_bps = 5.5
        signed = qty if side == 'Buy' else -qty
        e0 = self.entry.get(s, px)
        if q0 == 0 or q0 * signed > 0:
            self.entry[s] = (abs(q0) * e0 + qty * px) / (abs(q0) + qty)
        else:
            closed = min(qty, abs(q0))
            self.wallet += closed * (px - e0) * (1.0 if q0 > 0 else -1.0)
            if qty > abs(q0):
                self.entry[s] = px
        qn = q0 + signed
        self.qty[s] = 0.0 if abs(qn) < 1e-12 else qn
        if self.qty[s] == 0.0:
            self.entry.pop(s, None)
        fee = qty * px * fee_bps / 1e4
        self.wallet -= fee; self.stat['fees'] += fee; self.stat['slip'] += qty * p0 * sl / 1e4; self.stat['turnover'] += qty * px

    def market(self, symbol, qty, side, reduce_only=False, position_idx=0):
        self._fill(symbol, side, float(qty), reduce_only, False); return {}

    def post_only(self, symbol, qty, side, price_str, reduce_only=False, position_idx=0):
        self._fill(symbol, side, float(qty), reduce_only, True); return {}

    def apply_funding(self, i):
        tot = 0.0
        for s, q in self.qty.items():
            if q:
                r = d.fund[i, sidx[s]]
                if r:
                    tot -= q * self.px(s) * float(r)
        self.wallet += tot; self.stat['funding'] += tot

    def apply_close_events(self, i):
        for s, q in list(self.qty.items()):
            if q and d.close_event[i, sidx[s]]:
                m = self.px(s); pnl = q * (m - self.entry[s]) - 0.05 * abs(q * m)
                self.wallet += pnl; self.qty[s] = 0.0; self.entry.pop(s, None)


class FakeMD:
    """Panels straight from the research arrays (no API kline calls)."""
    def __init__(self, api):
        self.api = api; self.instr = {}; self.delist = set(); self.bars = {s: True for s in syms}

    def refresh_instruments(self, force=False):
        self.instr = self.api.instruments()

    def refresh_mcap(self, max_age_s=0):
        """No historical market caps in the simulation -> None, so main() applies the firm's low-cap rule through the
        documented proxy (24h turnover < lowcap_turn OR age < LOWCAP_YOUNG_D) — the same code path as the live bot without API."""
        return None

    def age_hours(self, s, now_ms, mode='listing'):
        """'ml': Binance segment age (as in training); 'listing': earliest listing = max(Binance age, Bybit age)."""
        if s not in sidx:
            return -1.0
        a = float(d.age[SIM['dec'], sidx[s]])
        if mode == 'ml' or os.environ.get('HARNESS_CLOCK', 'two') == 'binance':
            return a
        lh = instr_real.get(s, {}).get('launch')
        if lh:
            a = max(a, float(d.hh[SIM['dec']] - lh / 3.6e6))
        return a

    @staticmethod
    def last_closed_hour_ms(now_ms=None):
        return int(d.idx[SIM['dec']].timestamp() * 1000)

    def refresh(self, symbols, last_closed_ms, sleep=0):
        return [s for s in symbols if s in sidx], [s for s in symbols if s not in sidx]

    def panel(self, symbols, last_closed_ms, need_h=2200):
        i = SIM['dec']; i0 = max(0, i - need_h + 1)
        js = np.array([sidx[s] for s in symbols])
        sl = slice(i0, i + 1)
        valid = d.valid[sl][:, js]
        stale = ~valid[-1]
        data_age = np.array([float(d.age[i, j]) for j in js])
        return dict(grid=d.hh[sl], cff=d.cff[sl][:, js], high=d.high[sl][:, js].astype(np.float64), low=d.low[sl][:, js].astype(np.float64),
                    t24=d.t24[sl][:, js].astype(np.float64), fund=d.fund[sl][:, js].astype(np.float64),
                    prem=d.prem[sl][:, js].astype(np.float64) if d.prem is not None else None, valid=valid, stale=stale,
                    data_age=data_age, symbols=list(symbols))


class SimBot(botmain.Bot):
    def now(self):
        return now_dt()


if A.ml_panels:
    import signals as botsignals
    PANELS = {}
    for hz in (3, 7, 14, 30):
        pth = os.path.join(ROOT, f'{A.panels_prefix}_h{hz}{A.panels_suffix}.npy')
        if os.path.exists(pth) and (hz in (3, 7) or f'ml{hz}' in C.SLEEVES):
            PANELS[hz] = np.load(pth)
    p8 = os.path.join(ROOT, 'out_exp17_pred_8h_p0.npy')
    if os.path.exists(p8):
        PANELS[8] = np.load(p8)
    assert PANELS[3].shape == (T, N), 'prediction panel shape mismatch — rebuild with the same Data window'

    def ml_from_panel(self, panel, mask, age_h, hz):
        P = PANELS[hz]; i = SIM['dec']
        js = np.array([sidx[s] for s in panel['symbols']])
        s = np.where(mask & np.isfinite(P[i, js]), P[i, js], np.nan)
        return botsignals.quantile_ls(s, C.ML_TOP_FRAC) if hz != 8 else self.ml8_weights(s, panel)
    botsignals.Sleeves.ml = ml_from_panel
    if 8 in PANELS:
        botsignals.Sleeves.ml8 = lambda self, panel, mask, age_h, hour_utc: ml_from_panel(self, panel, mask, age_h, 8)
    print('ML sleeve: walk-forward prediction panels (fair OOS)')


def run_once(i0, phase=1, quiet=False, max_days=0):
    """One simulation from bar i0 with a fresh account/state. Returns dict(outcome, days, eq_end, eqs, stats, paid)."""
    C.PHASE = phase
    api = FakeApi(); md = FakeMD(api)
    mem = botstate.Memory(); mem.save = lambda: None
    bot = SimBot(api, md=md, mem=mem)
    bot.md.refresh_instruments(); bot.instr = bot.md.instr
    C.KILL_SWITCH = False
    eq_path = []; ts_path = []; gx_path = []
    SIM['i'] = i0; SIM['dec'] = i0 - A.exec_lag_h
    eq = api.equity(); mem['day_date'] = now_dt().date().isoformat(); mem['day_anchor'] = eq; mem['day_peak'] = eq; mem['eq_peak'] = max(A.account, eq)
    if C.MODE == 'challenge':
        mem['phase_base'] = A.account
    t0 = time.time(); n_reb = 0; paid = 0.0; outcome = 'end'; last_payout_day = None
    for i in range(i0, T - 1):
        SIM['i'] = i; SIM['dec'] = i - A.exec_lag_h
        api.apply_funding(i); api.apply_close_events(i)
        eq = api.equity(); eq_path.append(eq); ts_path.append(d.idx[i])
        if (i - i0) % 24 == 12:                       # daily gross/equity diagnostic (Opus: does the notional cap bind?)
            gx_path.append((d.idx[i], sum(abs(q) * api.px(s) for s, q in api.qty.items() if q) / max(eq, 1e-9)))
        dt = now_dt()
        if A.payout_days and C.MODE == 'funded' and dt.hour == 0 and (i - i0) // 24 > 0 and ((i - i0) // 24) % A.payout_days == 0 and last_payout_day != (i - i0) // 24:
            last_payout_day = (i - i0) // 24
            floor_keep = A.account * (1.0 + A.keep_buffer)
            profit = api.wallet + api.upnl() - floor_keep
            if profit > 50:
                amt = min(profit, api.wallet - floor_keep) if api.wallet > floor_keep else 0.0
                if amt > 0:
                    api.wallet -= amt; paid += amt
                    bot.risk.note_cash_flow(-amt)          # payout is a cash flow, not a loss: anchors/peak and VT NAV return adjust
        try:
            bot.watchdog()
            if mem['halt_total'] or mem['target_hit']:
                pass
            elif ((dt.hour - C.REBAL_HOUR_UTC) % C.REBAL_EVERY_H == 0) and mem.get('last_rebal_slot') != f'{dt.date().isoformat()}T{dt.hour:02d}':
                bot.rebalance(); n_reb += 1
            else:
                bot.scale_pass()
        except Exception as e:
            import traceback; print(f'bot error at {dt}: {e}\n{traceback.format_exc()[:2000]}'); raise
        # firm-side breach check (independent of the bot's own guards): static/trailing floor and daily limit
        if mem['halt_total']:
            outcome = 'halt'; break
        if mem['target_hit']:
            outcome = 'pass'; break
        if max_days and (i - i0) >= max_days * 24:
            outcome = 'timeout'; break
        if not quiet and (i - i0) % (24 * 90) == 0 and i > i0:
            print(f'  {d.idx[i].date()} eq {eq:,.0f} rebalances {n_reb} maker {api.stat["maker_filled"]}/{api.stat["maker_tried"]} market {api.stat["market"]} ({time.time()-t0:.0f}s)', flush=True)
    eqs = pd.Series(eq_path, index=pd.DatetimeIndex(ts_path))
    return dict(outcome=outcome, days=(i - i0) / 24.0, i_end=i, eq_end=api.equity(), eqs=eqs, stats=api.stat, n_reb=n_reb, paid=paid,
                min_eq=float(eqs.min()) if len(eqs) else np.nan, gx=pd.Series([g for _, g in gx_path], index=pd.DatetimeIndex([t for t, _ in gx_path])),
                ml8=dict(changes=mem.get('ml8_changes') or [], audit=mem.get('ml8_audit') or [], mult=mem.get('ml8_mult', 1.0)))


def run():
    if A.starts:
        starts = [s.strip() for s in A.starts.split(',') if s.strip()]
        rows = []
        for s in starts:
            i0 = max(int(d.idx.searchsorted(pd.Timestamp(s, tz='UTC'))), 2300)
            r1 = run_once(i0, phase=1, quiet=True, max_days=A.max_days)
            row = dict(start=s, p1=r1['outcome'], p1_days=round(r1['days']), p1_min_eq=round(r1['min_eq']), p1_end=round(r1['eq_end']))
            if C.MODE == 'challenge' and r1['outcome'] == 'pass' and A.phases >= 2 and r1['i_end'] + 24 < T - 1:
                r2 = run_once(r1['i_end'] + 1, phase=2, quiet=True, max_days=A.max_days)
                row.update(p2=r2['outcome'], p2_days=round(r2['days']), p2_min_eq=round(r2['min_eq']), total_days=round(r1['days'] + r2['days']))
            if C.MODE == 'funded':
                row.update(paid=round(r1['paid']), paid_pct_yr=round(r1['paid'] / A.account / max(r1['days'] / 365.0, 1e-9) * 100, 1))
            row.update(ml8_changes=len(r1['ml8']['changes']), ml8_mult_end=r1['ml8']['mult'])
            rows.append(row)
            print(row, flush=True)
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(ROOT, f'out_harness_rolling_{A.firm}_{A.mode}.csv'), index=False)
        if C.MODE == 'challenge':
            ok = df[df.p1 == 'pass']
            print(f'\nphase1: pass {len(ok)}/{len(df)} | halts {(df.p1 == "halt").sum()} | timeouts {(df.p1 == "timeout").sum()} | '
                  f'days median {ok.p1_days.median():.0f} p75 {ok.p1_days.quantile(.75):.0f} max {ok.p1_days.max():.0f}')
            if 'total_days' in df:
                both = df.dropna(subset=['total_days'])
                print(f'both phases: {len(both)} | total days median {both.total_days.median():.0f} p75 {both.total_days.quantile(.75):.0f} max {both.total_days.max():.0f} | phase2 outcomes {df.p2.value_counts().to_dict() if "p2" in df else {}}')
        else:
            print(f'\nfunded: alive {(df.p1 != "halt").sum()}/{len(df)} | paid %/yr median {df.paid_pct_yr.median():.1f} mean {df.paid_pct_yr.mean():.1f}')
        print('DONE'); return
    i0 = max(d.start_i, 2300)
    r = run_once(i0, phase=int(os.environ.get('PHASE', '1')), max_days=A.max_days)
    if r['outcome'] in ('halt', 'pass'):
        print(f'stopped: outcome={r["outcome"]} after {r["days"]:.0f} days, equity {r["eq_end"]:,.2f}')
    eqs = r['eqs']; api_stat = r['stats']
    m = metrics(eqs, label=f'HARNESS v4 {A.firm} {A.mode} pure={A.pure} maker_fill={A.maker_fill}', verbose=True)
    yrs = m['years']; e0 = A.account
    print(f'  costs %/yr of initial: fees {api_stat["fees"]/e0/yrs*100:.2f} slip {api_stat["slip"]/e0/yrs*100:.2f} funding {api_stat["funding"]/e0/yrs*100:+.2f} | '
          f'turnover {api_stat["turnover"]/e0/yrs:.0f}x | maker filled {api_stat["maker_filled"]}/{api_stat["maker_tried"]} market {api_stat["market"]} | rebalances {r["n_reb"]}'
          + (f' | paid out {r["paid"]:,.0f}' if r['paid'] else ''))
    ml8 = r.get('ml8') or {}
    print(f'  ml8 kill rule: multiplier changes {ml8.get("changes")} | final mult {ml8.get("mult")}')
    print('  ml8 monthly audit (date, net Sharpe 180d, cost/gross, mult): ' + ' '.join(str(a) for a in (ml8.get('audit') or [])[-24:]))
    gx = r.get('gx')
    if gx is not None and len(gx):
        print('  gross/equity by year (mean, p90): ' + '  '.join(f'{y}: {g.mean():.2f} ({g.quantile(.9):.2f})' for y, g in gx.groupby(gx.index.year))
              + f' | equity end/initial {r["eq_end"]/A.account:.2f}x | NOTIONAL_BASE={C.NOTIONAL_BASE}')
    eqs.to_csv(os.path.join(ROOT, f'out_harness_v4_{A.firm}_{A.mode}{"_pure" if A.pure else ""}.csv'))
    print('DONE')


if __name__ == '__main__':
    run()
