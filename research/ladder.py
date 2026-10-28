#!/usr/bin/env python3
"""Phase T: ladder economics on Mubite Two-Step Pro (one account at a time — account rolling is forbidden, so a funded account is closed
before the next challenge starts). Historical paths from research/fastsim.py, risk settings = consensus (challenge risk 1.2, funded risk 1.2).

A strategy = start size, add-ons (DD10 / T8 / split 90), path of target sizes. Own money = the first fee only. Every later fee is paid from
payouts. Funded: payout every 28 days (positions closed, 48 h pause), caps 2.5 -> 5 %, Funded Score 40 %, keep 2 % buffer; trader gets the
split; the challenge fee is refunded with the first payout of that account. Switch to the next size when cash >= its price (+ reserve);
the old account is closed then. A failed challenge: buy the largest affordable size again (else stop). At the last size: scaling plan
+25 % of the start size per 90-day window with >= 10 % profit (incl. payouts), up to CAP. Extra challenge-failure stress: each challenge
also fails with probability Q (Monte Carlo).
Run: python3 research/ladder.py  -> research/phase4_out/T_ladder.csv + table"""
import os, sys, itertools, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fastsim as fs

PRICE = {5000: 65, 10000: 110, 25000: 249, 50000: 419, 100000: 729, 200000: 1099}
ADDON = 0.20
CAP = 300000            # funded cap per person (FAQ: $300k; another FAQ says $400k) — conservative
REVIEW_D, FUNDED_D = 2, 3
ts, R, G, V, hour, day, RH = fs.load_books()
T = R.shape[2]
P_CH = fs.pvec(VT=0.006, BASE=2.0, BASE_P2=2.0, KMAX=3.0, CPPI=1.0, DD_BUF=0.3, THROTTLE=0.5, HALT_FRAC=0.5, FG=0.03, BOOK=0, RAT=0.35)
P_FU = fs.pvec(VT=0.006, BASE=2.0, KMAX=3.0, CPPI=0.75, DD_BUF=0.3, THROTTLE=0.5, HALT_FRAC=0.5, FG=0.03, BOOK=0, KEEP_BUF=0.02)
_cache = {}


def challenge(i0, t1, dd):
    """(passed, i_end) of phase 1 + review + phase 2 from bar i0 (cached)."""
    key = ('c', i0, t1, dd)
    if key not in _cache:
        f = fs.fvec(T1=t1, T2=5, TOTAL_DD=dd, DAILY_DD=5, MIN_D1=10, MIN_D2=3, PROF_D1=3, PROF_D2=2, NOTIONAL_X=100)
        o1, ie1, *_ = fs.sim(R, G, V, RH, hour, day, i0, P_CH, f, 0, 1, 0, 10000.0)
        if o1 != 1:
            _cache[key] = (False if o1 in (2, 3) else None, ie1)
        else:
            i2 = min(ie1 + 24 * REVIEW_D, T - 2)
            o2, ie2, *_ = fs.sim(R, G, V, RH, hour, day, i2, P_CH, f, 0, 2, 0, 10000.0)
            _cache[key] = ((True if o2 == 1 else (False if o2 in (2, 3) else None)), ie2)
    return _cache[key]


def funded(i0, dd):
    """cumulative payouts (fraction of size) per bar from i0, wealth path, death bar or None (cached)."""
    key = ('f', i0, dd)
    if key not in _cache:
        f = fs.fvec(T1=1e9, T2=1e9, TOTAL_DD=dd, DAILY_DD=5, NOTIONAL_X=3, PAY_EVERY=28, FSCORE=40, MIN_PAY=50, PAUSE_H=48, PAY_COST=0.0011)
        rp = np.zeros(T); rw = np.zeros(T)
        o, ie, *_ = fs.sim(R, G, V, RH, hour, day, i0, P_FU, f, 1, 1, 0, 10000.0, np.zeros(1), rw, rp)
        rp[ie + 1:] = rp[ie]; rw[ie + 1:] = rw[ie]
        _cache[key] = (rp / 10000.0, rw / 10000.0, ie if o in (2, 3) else None)
    return _cache[key]


def price(size, addons):
    return PRICE[size] * (1 + ADDON * len(addons))


def run(i0, start_size, path, addons, horizon_d, q, rng, reserve=0.0):
    """One owner history from bar i0. Returns cash (net of own first fee) at horizon, final funded size, days to the last size."""
    t1 = 8 if 'T8' in addons else 10; dd = 10 if 'DD10' in addons else 8; split = 0.9 if 'S90' in addons else 0.8
    i_h = min(i0 + horizon_d * 24, T - 1)
    cash = -price(start_size, addons)                       # own money spent on the first fee
    size = start_size; t = i0; fee = price(size, addons); steps = list(path); reached = None
    while t < i_h:
        ok, ie = challenge(t, t1, dd)
        if ok and rng.random() < q:                          # extra failure stress
            ok = False
        if ok is None or ie >= i_h:                          # still running at the horizon
            return cash, 0, reached
        if not ok:
            aff = [s for s in PRICE if price(s, addons) <= cash]                 # buy again from payouts only
            if not aff:
                return cash, 0, reached
            size = max(aff); fee = price(size, addons); cash -= fee; t = ie + 24; continue
        tf = min(ie + 24 * FUNDED_D, T - 2)
        rp, rw, dead = funded(tf, dd)
        final = not steps or size >= steps[-1]
        nxt = next((s for s in steps if s > size), None)
        if final and reached is None:
            reached = (tf - i0) / 24
        refund_done = False; scale = 1.0; base_size = size; last_scale_t = tf
        # walk the funded account day by day until switch / death / horizon
        for tt in range(tf + 24, i_h + 1, 24):
            if dead is not None and tt >= dead:
                break
            paid_frac = rp[tt]
            if final:                                       # scaling plan at the last size
                if tt - last_scale_t >= 90 * 24 and rw[tt] - rw[tt - 90 * 24] >= 0.10 and base_size * (scale + 0.25) <= CAP:
                    scale += 0.25; last_scale_t = tt
            got = paid_frac * base_size * split * (scale if final else 1.0)
            if got > 0 and not refund_done:
                cash += fee; refund_done = True
            if not final and nxt is not None and cash + got >= price(nxt, addons) + reserve:
                cash += got; size = nxt; fee = price(size, addons); cash -= fee; t = tt; break
        else:
            tt = i_h
            cash += rp[min(tt, T - 1)] * base_size * split * (scale if final else 1.0)
            return cash, int(base_size * scale), reached
        if dead is not None and tt >= dead:                  # account died: keep the cash, buy the largest affordable challenge
            cash += rp[dead] * base_size * split * (scale if final else 1.0)
            aff = [s for s in PRICE if price(s, addons) <= cash]
            if not aff:
                return cash, 0, reached
            size = max(aff); fee = price(size, addons); cash -= fee; t = dead + 24
            continue
    return cash, 0, reached


PATHS = {'direct': [200000], 'via50': [50000, 200000], 'via25_100': [25000, 100000, 200000], 'via100': [100000, 200000],
         'ladder': [25000, 50000, 100000, 200000]}
ADDONS = {'all3': ('DD10', 'T8', 'S90'), 'DD10+S90': ('DD10', 'S90'), 'DD10+T8': ('DD10', 'T8'), 'DD10': ('DD10',), 'none': ()}

if __name__ == '__main__':
    rows = []
    for horizon in (24, 36):
        st = fs.starts_idx(ts, first='2022-01-03', last=str((ts[-1] - pd.Timedelta(days=horizon * 30.5)).date()), every_days=7)
        for (s0, (pn, path), (an, ad), q) in itertools.product((5000, 10000), PATHS.items(), ADDONS.items(), (0.0, 0.05)):
            rng = np.random.default_rng(7)
            nrep = 1 if q == 0 else 20
            res = [run(i0, s0, path, ad, int(horizon * 30.5), q, rng) for i0 in st for _ in range(nrep)]
            cash = np.array([r[0] for r in res]); sz = np.array([r[1] for r in res]); rc = np.array([r[2] if r[2] is not None else np.nan for r in res])
            rows.append(dict(horizon_m=horizon, start=s0, path=pn, addons=an, q=q, n=len(st), cash_mean=round(cash.mean()), cash_p10=round(np.percentile(cash, 10)),
                             cash_p50=round(np.median(cash)), p_lost=round((cash <= 0).mean(), 3), size_end_med=int(np.median(sz)),
                             p_reach=round(np.isfinite(rc).mean(), 3), months_to_last=round(np.nanmedian(rc) / 30.5, 1) if np.isfinite(rc).any() else np.nan))
            print(rows[-1], flush=True)
    df = pd.DataFrame(rows); df.to_csv(os.path.join(fs.ROOT, 'research', 'phase4_out', 'T_ladder.csv'), index=False)
