#!/usr/bin/env python3
"""Phase T: random search over ALL risk settings of the bot with the fast simulator (research/fastsim.py), Mubite Two-Step Pro.
Challenge and funded settings act independently, so they are searched separately:
  challenge: every config x firm variant (target 10/8 %, DD 8/10 %) x weekly starts 2022-01-03..2026-09-07; phase 1 -> 2 days review ->
             phase 2. Metrics: fail (bot halt or firm breach in either phase), pass, days to pass both. TRAIN = starts 2022-2024, TEST = 2025-26.
  funded:    every config x DD 8/10 % x weekly starts 2022-01-03..2025-10-20, 365 days: breach / halt, payouts (14-day cycle, caps 2.5->5 %,
             Funded Score < 40 %, min $50), trader share is applied later.
The search space (list fixed before the run) is in SPACE_CH / SPACE_FU. Output: research/phase4_out/T_grid_<kind>.parquet
Run: python3 research/fastsim_grid.py challenge 6000 ; python3 research/fastsim_grid.py funded 4000"""
import os, sys, time, numpy as np, pandas as pd
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fastsim as fs

SPACE_CH = dict(VT=[0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.010, 0.012], BASE=[1.0, 1.5, 2.0, 2.5, 3.0], P2R=[0.5, 0.75, 1.0, 1.25],
                KMAXR=[1.0, 1.5], CPPI=[0.0, 0.25, 0.5, 0.75, 1.0], KMIN=[0.0, 0.2, 0.3, 0.5], DD_BUF=[0.0, 0.3, 0.5, 0.7],
                THROTTLE=[0.5, 1.0, 2.0, 100.0], HALT_FRAC=[0.4, 0.5, 0.6, 0.8], FG=[0.03, 0.05, 0.1], BOOK=[0, 2], RAT=[0.1, 0.35, 0.7],
                MAXG=[2.0, 3.0])
SPACE_FU = dict(VT=[0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.010], BASE=[0.75, 1.0, 1.25, 1.5, 2.0], KMAXR=[1.0, 1.5],
                CPPI=[0.0, 0.25, 0.5, 0.75, 1.0], KMIN=[0.0, 0.2, 0.3, 0.5], DD_BUF=[0.0, 0.3, 0.5, 0.7], THROTTLE=[0.5, 1.0, 2.0, 100.0],
                HALT_FRAC=[0.4, 0.5, 0.6, 0.8], FG=[0.03, 0.05, 0.1], BOOK=[0, 2], KEEP_BUF=[0.0, 0.01, 0.02, 0.04], MAXG=[2.0, 3.0])
FIRMS_CH = {'T10_DD8': (10, 8), 'T10_DD10': (10, 10), 'T8_DD8': (8, 8), 'T8_DD10': (8, 10)}
REVIEW_D = 2
BOOKS = None


def _init():
    global BOOKS
    BOOKS = fs.load_books()


def to_p(c, kind):
    kw = dict(VT=c['VT'], BASE=c['BASE'], KMAX=c['BASE'] * c['KMAXR'] if c['KMAXR'] > 1 else 0.0, CPPI=c['CPPI'], KMIN=c['KMIN'],
              DD_BUF=c['DD_BUF'], THROTTLE=c['THROTTLE'], HALT_FRAC=c['HALT_FRAC'], FG=c['FG'], BOOK=c['BOOK'], MAXG=c['MAXG'])
    if kind == 'challenge':
        kw.update(BASE_P2=c['BASE'] * c['P2R'], RAT=c['RAT'])
    else:
        kw.update(KEEP_BUF=c['KEEP_BUF'])
    return fs.pvec(**kw)


def eval_ch(args):
    cid, c = args
    ts, R, G, V, hour, day, RH = BOOKS
    p = to_p(c, 'challenge'); T = R.shape[2]
    st = fs.starts_idx(ts, every_days=7); yr = ts[st].year
    out = []
    for fname, (t1, dd) in FIRMS_CH.items():
        f = fs.fvec(T1=t1, T2=5, TOTAL_DD=dd, DAILY_DD=5, MIN_D1=10, MIN_D2=3, PROF_D1=3, PROF_D2=2, NOTIONAL_X=100)
        fail = np.zeros(len(st), bool); ok = np.zeros(len(st), bool); days = np.full(len(st), np.nan); p1f = np.zeros(len(st), bool)
        for j, i0 in enumerate(st):
            o1, ie1, *_ = fs.sim(R, G, V, RH, hour, day, i0, p, f, 0, 1, 0, 10000.0)
            if o1 in (2, 3):
                fail[j] = True; p1f[j] = True; continue
            if o1 != 1:
                continue
            i2 = ie1 + 24 * REVIEW_D
            if i2 >= T - 48:
                continue
            o2, ie2, *_ = fs.sim(R, G, V, RH, hour, day, i2, p, f, 0, 2, 0, 10000.0)
            if o2 in (2, 3):
                fail[j] = True
            elif o2 == 1:
                ok[j] = True; days[j] = (ie2 - i0) / 24.0
        for lab, sel in (('tr', yr < 2025), ('te', yr >= 2025)):
            done = sel & (fail | ok)
            out.append(dict(cid=cid, firm=fname, set=lab, n=int(sel.sum()), n_done=int(done.sum()), fail=float(fail[sel].mean()),
                            fail_p1=float(p1f[sel].mean()), pass_=float(ok[sel].mean()), days_med=float(np.nanmedian(days[sel])) if ok[sel].any() else np.nan,
                            days_p75=float(np.nanpercentile(days[sel], 75)) if ok[sel].any() else np.nan))
    return [dict(r, **c) for r in out]


def eval_fu(args):
    cid, c = args
    ts, R, G, V, hour, day, RH = BOOKS
    p = to_p(c, 'funded')
    st = fs.starts_idx(ts, last='2025-10-20', every_days=7); yr = ts[st].year
    out = []
    for dd in (8, 10):
        f = fs.fvec(T1=1e9, T2=1e9, TOTAL_DD=dd, DAILY_DD=5, NOTIONAL_X=3, PAY_EVERY=14, FSCORE=40, MIN_PAY=50)
        res = np.array([fs.sim(R, G, V, RH, hour, day, i0, p, f, 1, 1, 365 * 24, 10000.0)[:6] for i0 in st])
        o, paid, npay = res[:, 0], res[:, 4], res[:, 5]
        for lab, sel in (('tr', yr < 2025), ('te', yr >= 2025)):
            out.append(dict(cid=cid, dd=dd, set=lab, n=int(sel.sum()), dead=float(np.isin(o[sel], (2, 3)).mean()),
                            breach=float((o[sel] == 3).mean()), paid_mean=float(paid[sel].mean()), paid_p10=float(np.percentile(paid[sel], 10)),
                            npay=float(npay[sel].mean())))
    return [dict(r, **c) for r in out]


def sample(space, n, seed):
    rng = np.random.default_rng(seed)
    return [(k, {name: vals[rng.integers(len(vals))] for name, vals in space.items()}) for k in range(n)]


if __name__ == '__main__':
    kind, n = sys.argv[1], int(sys.argv[2])
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    cfgs = sample(SPACE_CH if kind == 'challenge' else SPACE_FU, n, seed)
    if kind == 'challenge':                                         # anchors: current presets (R3-bis P3/VT 0.6) for reference
        cfgs.append((n, dict(VT=0.006, BASE=2.0, P2R=1.0, KMAXR=1.0, CPPI=0.5, KMIN=0.3, DD_BUF=0.5, THROTTLE=1.0, HALT_FRAC=0.5, FG=0.05,
                             BOOK=0, RAT=0.35, MAXG=2.0)))
    else:
        cfgs.append((n, dict(VT=0.006, BASE=1.25, KMAXR=1.0, CPPI=0.5, KMIN=0.3, DD_BUF=0.5, THROTTLE=1.0, HALT_FRAC=0.5, FG=0.05, BOOK=0,
                             KEEP_BUF=0.0, MAXG=2.0)))
    t0 = time.time(); rows = []
    with Pool(int(os.environ.get('NPROC', '4')), initializer=_init) as pool:
        for k, r in enumerate(pool.imap_unordered(eval_ch if kind == 'challenge' else eval_fu, cfgs, chunksize=4)):
            rows += r
            if k % 200 == 0:
                print(f'{k}/{len(cfgs)} {time.time() - t0:.0f}s', flush=True)
    out = os.path.join(fs.ROOT, 'research', 'phase4_out', f'T_grid_{kind}{os.environ.get("TAG", "")}.parquet')
    pd.DataFrame(rows).to_parquet(out)
    print('saved', out, f'{time.time() - t0:.0f}s')
