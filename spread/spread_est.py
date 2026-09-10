#!/usr/bin/env python3
"""Effective spread by liquidity tier from Bybit public tick trades (public.bybit.com/trading).
For each symbol-date: turnover, trades, effective spread from consecutive opposite-side trades
within 2s (median/p75 bps), tick-size floor (bps), Roll estimator on 1-min prices (bps)."""
import os, io, gzip, json, sys, time
import numpy as np, pandas as pd, requests

OUT = os.path.dirname(os.path.abspath(__file__))
SYMS = """BTCUSDT ETHUSDT SOLUSDT XRPUSDT DOGEUSDT BNBUSDT ADAUSDT AVAXUSDT LINKUSDT LTCUSDT DOTUSDT SUIUSDT
NEARUSDT APTUSDT ARBUSDT OPUSDT FILUSDT ATOMUSDT UNIUSDT INJUSDT TIAUSDT SEIUSDT WLDUSDT 1000PEPEUSDT WIFUSDT
FETUSDT RENDERUSDT AAVEUSDT ONDOUSDT ENAUSDT JUPUSDT STXUSDT ALGOUSDT SANDUSDT MANAUSDT GALAUSDT AXSUSDT CHZUSDT
ENJUSDT GMTUSDT APEUSDT LDOUSDT DYDXUSDT GRTUSDT IMXUSDT RUNEUSDT KAVAUSDT ZILUSDT ONEUSDT IOTAUSDT ANKRUSDT
CELOUSDT SKLUSDT STORJUSDT LRCUSDT ONTUSDT ZRXUSDT BANDUSDT KNCUSDT SXPUSDT RLCUSDT CTSIUSDT HOTUSDT RVNUSDT
SFPUSDT ALICEUSDT BAKEUSDT DODOUSDT CVCUSDT ARPAUSDT HIGHUSDT JASMYUSDT ACHUSDT IDUSDT ARKMUSDT PENDLEUSDT
XVGUSDT CYBERUSDT BIGTIMEUSDT ORDIUSDT MEMEUSDT PYTHUSDT BEAMUSDT STRKUSDT PIXELUSDT PORTALUSDT AEVOUSDT
ETHFIUSDT ZKUSDT IOUSDT NOTUSDT TONUSDT POPCATUSDT NEIROUSDT MOODENGUSDT GOATUSDT PNUTUSDT ACTUSDT
API3USDT AGLDUSDT ALPHAUSDT BELUSDT COTIUSDT DENTUSDT DUSKUSDT FLMUSDT GLMUSDT HFTUSDT IOSTUSDT KEYUSDT
LEVERUSDT LITUSDT MDTUSDT MTLUSDT NKNUSDT OGNUSDT OXTUSDT PERPUSDT QTUMUSDT RDNTUSDT REQUSDT SLPUSDT SNTUSDT
SPELLUSDT STEEMUSDT STGUSDT TRUUSDT UMAUSDT VIDTUSDT WAXPUSDT XEMUSDT""".split()
DATES = ['2026-08-12', '2026-05-20']
OLD = '2024-06-12'
sess = requests.Session()


def fetch(sym, date):
    url = f'https://public.bybit.com/trading/{sym}/{sym}{date}.csv.gz'
    for a in range(3):
        try:
            r = sess.get(url, timeout=180)
            if r.status_code == 200 and len(r.content) > 100:
                return r.content
            if r.status_code == 404:
                return None
        except Exception:
            time.sleep(2)
    return None


def analyse(sym, date, raw):
    df = pd.read_csv(io.BytesIO(raw), compression='gzip',
                     usecols=['timestamp', 'side', 'price', 'foreignNotional'])
    df = df.sort_values('timestamp', kind='mergesort').reset_index(drop=True)
    px = df.price.values.astype(float); ts = df.timestamp.values.astype(float)
    side = (df.side.values == 'Buy')
    notion = df.foreignNotional.values.astype(float)
    n = len(df)
    if n < 100:
        return None
    vwap = notion.sum() / max((notion / px).sum(), 1e-12)
    # consecutive opposite-side pairs within 2 s
    opp = side[1:] != side[:-1]
    close_t = (ts[1:] - ts[:-1]) <= 2.0
    m = opp & close_t
    sp = np.abs(px[1:] - px[:-1])[m] / px[1:][m] * 1e4
    # quoted-spread proxy: only count pairs where buy price > sell price (crossing the spread)
    d = (px[1:] - px[:-1]) * np.where(side[1:], 1, -1)   # buy after sell -> positive if ask>bid
    dd = d[m] / px[1:][m] * 1e4
    dd_pos = dd[dd > 0]
    u = np.unique(px)
    tick = np.min(np.diff(u)) if len(u) > 1 else np.nan
    tick_bps = tick / vwap * 1e4
    # Roll estimator on 1-min last prices
    mins = (ts // 60).astype(int)
    last_idx = np.flatnonzero(np.diff(mins, append=mins[-1] + 1))
    p1 = px[last_idx]
    dp = np.diff(p1)
    cov = np.cov(dp[1:], dp[:-1])[0, 1] if len(dp) > 10 else np.nan
    roll_bps = 2 * np.sqrt(max(0.0, -cov)) / vwap * 1e4 if np.isfinite(cov) else np.nan
    hours = (ts // 3600).astype(int)
    ht = pd.Series(notion).groupby(hours).sum()
    return dict(symbol=sym, date=date, turnover_usd=float(notion.sum()), n_trades=int(n),
                med_trade_usd=float(np.median(notion)), vwap=float(vwap),
                eff_spread_med_bps=float(np.median(sp)) if len(sp) else np.nan,
                eff_spread_p75_bps=float(np.percentile(sp, 75)) if len(sp) else np.nan,
                cross_spread_med_bps=float(np.median(dd_pos)) if len(dd_pos) else np.nan,
                n_pairs=int(m.sum()), tick_bps=float(tick_bps), roll_bps=float(roll_bps),
                hourly_turnover_med=float(ht.median()), hourly_turnover_min=float(ht.min()))


def main():
    instr = json.load(open(os.path.join(OUT, '..', 'instruments.json')))
    syms = [s for s in SYMS if s in instr]
    print(f'{len(syms)} symbols', flush=True)
    rows = []
    outp = os.path.join(OUT, 'spread_by_symbol_day.csv')
    done = set()
    if os.path.exists(outp):
        prev = pd.read_csv(outp); rows = prev.to_dict('records')
        done = set(zip(prev.symbol, prev.date))
    jobs = [(s, d) for s in syms for d in DATES] + [(s, OLD) for s in syms[:20] + syms[-25:]]
    t0 = time.time()
    for k, (s, d) in enumerate(jobs):
        if (s, d) in done:
            continue
        raw = fetch(s, d)
        if raw is None:
            print(f'  {s} {d}: missing', flush=True); continue
        try:
            r = analyse(s, d, raw)
        except Exception as e:
            print(f'  {s} {d}: ERR {e}', flush=True); continue
        del raw
        if r:
            rows.append(r)
            print(f'  {k}/{len(jobs)} {s} {d}: turn ${r["turnover_usd"]/1e6:.1f}M spread {r["eff_spread_med_bps"]:.2f}bp '
                  f'tick {r["tick_bps"]:.2f}bp roll {r["roll_bps"]:.2f}bp ({time.time()-t0:.0f}s)', flush=True)
        if k % 10 == 0:
            pd.DataFrame(rows).to_csv(outp, index=False)
    df = pd.DataFrame(rows); df.to_csv(outp, index=False)
    # tier table
    bins = [0, 1e6, 3e6, 1e7, 5e7, 2e8, 1e9, 1e12]
    labels = ['<1M', '1-3M', '3-10M', '10-50M', '50-200M', '200M-1B', '>1B']
    df['tier'] = pd.cut(df.turnover_usd, bins, labels=labels)
    g = df.groupby('tier', observed=True).agg(n=('symbol', 'count'), spread_med=('eff_spread_med_bps', 'median'),
                                              spread_p75=('eff_spread_p75_bps', 'median'), cross=('cross_spread_med_bps', 'median'),
                                              tick=('tick_bps', 'median'), roll=('roll_bps', 'median'))
    print(g.round(2).to_string())
    ok = df[(df.turnover_usd > 0) & df.eff_spread_med_bps.notna() & (df.eff_spread_med_bps > 0)]
    b, a = np.polyfit(np.log(ok.turnover_usd), np.log(ok.eff_spread_med_bps), 1)
    print(f'log-log fit: spread_bps = exp({a:.3f}) * turnover^{b:.3f}')
    json.dump({'tiers': g.round(3).reset_index().astype(str).to_dict('records'), 'loglog': {'a': a, 'b': b}},
              open(os.path.join(OUT, 'spread_model.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
