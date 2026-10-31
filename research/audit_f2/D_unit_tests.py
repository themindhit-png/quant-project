"""D_ audit: synthetic unit tests of bt.run accounting (funding timing/sign, price P&L timing, forced close shock)."""
import os, sys, numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, ROOT)
import bt
class Fake(bt.Data):
    def __init__(self, T, close, fund):
        self.idx = pd.date_range('2022-01-01', periods=T, freq='h', tz='UTC'); self.cols = np.array(['AUSDT', 'BUSDT'])
        self.cff = close; self.valid = np.isfinite(close); r = np.full_like(close, np.nan); r[1:] = close[1:] / close[:-1] - 1; self.ret = r
        self.fund = fund.astype(np.float32); self.close_event = np.zeros_like(close, bool); self.hh = (self.idx.as_unit('s').asi8 // 3600).astype(np.int64)
        self.t24 = np.full(close.shape, 1e9, np.float32); self.age = np.full(close.shape, 10 ** 6, np.int32); self.start_i = 0; self.high = self.low = self.prem = None; self.prev_mask = None
T = 24 * 40; i0 = 24 * 35
close = np.ones((T, 2)) * 100.0; fund = np.zeros((T, 2))
# signal: long A / short B decided at bar i0 (hh%24==0) -> executed at close[i0]
def sig(dd, i, mask): return np.array([1.0, -1.0])
base = dict(reb_h=24 * 1000, fee_bps=0, slip_bps=0, verbose=False, min_trade=0, pos_cap=1.0, start_eq=10000.0)
# Test 1: funding +1% on A at row i0 (instant the order fills) and +1% at row i0+5 (while held)
f1 = fund.copy(); f1[i0, 0] = 0.01; f1[i0 + 5, 0] = 0.01; f1[i0 + 5, 1] = 0.01
d = Fake(T, close, f1); m, eq = bt.run(d, sig, **dict(base, reb_h=24))
print('T1 funding: end eq', round(eq.iloc[-1], 4), '(expected 10000 - 5000*0.01 [long A pays] + 5000*0.01 [short B receives] = 10000; row i0 not charged)')
f2 = fund.copy(); f2[i0 + 5, 0] = 0.01
d = Fake(T, close, f2); m, eq = bt.run(d, sig, **dict(base, reb_h=24)); print('T1b long pays +rate: end eq', round(eq.iloc[-1], 4), '(expected 9950)')
f3 = fund.copy(); f3[i0, 0] = 0.01
d = Fake(T, close, f3); m, eq = bt.run(d, sig, **dict(base, reb_h=24)); print('T1c funding row == execution bar i0: end eq', round(eq.iloc[-1], 4), '(10000 => position opened at close[i0] does not pay row i0)')
# Test 2: price jump on bar i0 itself (close[i0] up 10%) must NOT be earned; jump on bar i0+1 must be earned
c2 = close.copy(); c2[i0:, 0] = 110.0
d = Fake(T, c2, fund); m, eq = bt.run(d, sig, **dict(base, reb_h=24)); print('T2a jump at bar i0 (decision bar): end eq', round(eq.iloc[-1], 4), '(expected 10000)')
c3 = close.copy(); c3[i0 + 1:, 0] = 110.0
d = Fake(T, c3, fund); m, eq = bt.run(d, sig, **dict(base, reb_h=24)); print('T2b jump at bar i0+1: end eq', round(eq.iloc[-1], 4), '(expected 10500)')
# Test 3: short notional drift: B falls 50% -> short P&L +2500 and notional halves
c4 = close.copy(); c4[i0 + 1:, 1] = 50.0
d = Fake(T, c4, fund); m, eq = bt.run(d, sig, **dict(base, reb_h=24)); print('T3 short gain: end eq', round(eq.iloc[-1], 4), '(expected 12500)')
# Test 4: funding row timing relative to ret: is funding fund[i] charged before or after the bar-i return? (notional check)
c5 = close.copy(); c5[i0 + 5:, 0] = 200.0; f5 = fund.copy(); f5[i0 + 5, 0] = 0.01
d = Fake(T, c5, f5); m, eq = bt.run(d, sig, **dict(base, reb_h=24))
print('T4 funding row k (instant k, price 100) while price doubles on bar k (close[k]=200): end eq', round(eq.iloc[-1], 4), '(notional at instant k = 10000 -> 19900; engine charging notional after bar k = 20000 -> 19800)')
