"""Сравнение предсказаний боевых моделей ml3/ml7 на панелях Binance и на свечах Bybit (дамп VPS) в одни и те же часы.
Запуск из корня: python3 research/ml_inference_check.py"""
import os, sys, gzip, numpy as np, pandas as pd, lightgbm as lgb
from scipy.stats import spearmanr
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'v4'))
from ml_features import compute_features, rank_norm, FEATURES, NEED_H
feats = [f for f in FEATURES if f != 'age_d']
mdl = {hz: lgb.Booster(model_file=os.path.join(ROOT, f'v4/models/ml_h{hz}_noage.txt')) for hz in (3, 7)}
P = lambda n: pd.read_parquet(os.path.join(ROOT, f'data/panels/{n}.parquet'))
B = {k: P(n) for k, n in dict(c='close', h='high', l='low', q='qvol', f='funding', p='premium').items()}
# Bybit из дампа
kl = pd.concat([pd.read_csv(os.path.join(ROOT, 'research/vps_dump/klines', f), compression='gzip') for f in sorted(os.listdir(os.path.join(ROOT, 'research/vps_dump/klines')))])
kl['t'] = pd.to_datetime(kl.ts_ms, unit='ms', utc=True)
Y = {k: kl.pivot_table(index='t', columns='symbol', values=v, aggfunc='last') for k, v in dict(c='close', h='high', l='low', q='turnover', p='prem_close').items()}
fu = pd.read_csv(os.path.join(ROOT, 'research/vps_dump/funding_bybit.csv.gz'))
fu['t'] = pd.to_datetime(fu.ts_ms, unit='ms', utc=True).dt.floor('h')
Y['f'] = fu.pivot_table(index='t', columns='symbol', values='rate', aggfunc='sum')

def preds(D, end, syms, fund_zero=False):
    idx = pd.date_range(end=end, periods=NEED_H, freq='h', tz='UTC')
    g = lambda k: D[k].reindex(index=idx, columns=syms)
    c = g('c'); c = c.ffill()
    q = g('q'); t24 = q.fillna(0).rolling(24).sum().where(q.notna().rolling(24, min_periods=1).max() > 0)
    f = g('f').fillna(0.0)
    F = compute_features(c.values, f.values, t24.values, g('p').values, g('h').values, g('l').values, np.full(len(syms), 1e5), syms.index('BTCUSDT'))
    X = np.column_stack([F[k] for k in feats]).astype(np.float32)
    ok = np.isfinite(c.values[-1]) & (t24.values[-1] > 5e6)
    out = {}
    for hz, m in mdl.items():
        s = np.full(len(syms), np.nan); ii = np.flatnonzero(ok)
        s[ii] = m.predict(rank_norm(X[ii]).astype(np.float64)); out[hz] = s
    return out, F, ok

common = sorted(set(B['c'].columns) & set(Y['c'].columns))
print('общих символов', len(common))
rows = []
for end in pd.to_datetime(['2026-09-15 00:00', '2026-09-20 08:00', '2026-09-25 16:00', '2026-09-29 00:00'], utc=True):
    pb, Fb, okb = preds(B, end, common); py, Fy, oky = preds(Y, end, common)
    both = okb & oky
    r = dict(end=end, n=int(both.sum()), n_bin=int(okb.sum()), n_byb=int(oky.sum()))
    for hz in (3, 7):
        r[f'sp_h{hz}'] = spearmanr(pb[hz][both], py[hz][both]).correlation
    for k in ('r24', 'r168', 'r720', 'fund24', 'fund168', 'prem24', 'vol168', 'logturn', 'hl168', 'beta720'):
        a, b = Fb[k][both], Fy[k][both]; m = np.isfinite(a) & np.isfinite(b)
        r['f_' + k] = spearmanr(a[m], b[m]).correlation if m.sum() > 10 else np.nan
    rows.append(r)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
print(pd.DataFrame(rows).round(3).T)

# --- абляция: какие входы Bybit ломают предсказание (подмена входа Bybit на Binance по одному)
import warnings; warnings.filterwarnings('ignore')
print('\nАбляция: Spearman(pred Binance, pred Bybit-с-подменой), среднее по 4 датам, h3/h7; и перекрытие топ/низ-20% лонг-шорт')
def sw(keys):
    Z = {k: (B[k] if k in keys else Y[k]) for k in Y}
    return Z
ends = pd.to_datetime(['2026-09-15 00:00', '2026-09-20 08:00', '2026-09-25 16:00', '2026-09-29 00:00'], utc=True)
for name, keys in [('как есть', []), ('funding<-Binance', ['f']), ('premium<-Binance', ['p']), ('turnover<-Binance', ['q']), ('f+p', ['f', 'p']), ('f+p+q', ['f', 'p', 'q'])]:
    s3, s7, ov = [], [], []
    for end in ends:
        pb, _, okb = preds(B, end, common); py, _, oky = preds(sw(keys), end, common)
        both = okb & oky
        s3.append(spearmanr(pb[3][both], py[3][both]).correlation); s7.append(spearmanr(pb[7][both], py[7][both]).correlation)
        a, b = pb[7][both], py[7][both]; n = max(int(len(a) * .2), 1)
        top = lambda x: set(np.argsort(x)[-n:]) | set(np.argsort(x)[:n]); ov.append(len(top(a) & top(b)) / len(top(a)))
    print(f'{name:20s} h3 {np.mean(s3):.3f}  h7 {np.mean(s7):.3f}  перекрытие L/S-корзин h7 {np.mean(ov):.2f}')

# --- IC в сентябре: предсказания (Binance-входы vs Bybit-входы) против будущей доходности по ценам Bybit
print('\nIC сентября (Spearman pred vs fwd-доходность Bybit), ends каждые 8ч 11.09-26.09; h3 -> fwd72ч, h7 -> fwd168ч(до 23.09)')
res = {k: [] for k in ('b3', 'y3', 'b7', 'y7')}
for end in pd.date_range('2026-09-11 00:00', '2026-09-27 00:00', freq='8h', tz='UTC'):
    pb, _, okb = preds(B, end, common); py, _, oky = preds(Y, end, common)
    for hz, fw in ((3, 72), (7, 168)):
        e2 = end + pd.Timedelta(hours=fw)
        if e2 > Y['c'].index[-1] or e2 > pd.Timestamp('2026-09-30 23:00', tz='UTC'): continue
        r = (Y['c'].reindex(columns=common).loc[e2] / Y['c'].reindex(columns=common).loc[end] - 1).values
        m = okb & oky & np.isfinite(r)
        if m.sum() < 30: continue
        res[f'b{hz}'].append(spearmanr(pb[hz][m], r[m]).correlation); res[f'y{hz}'].append(spearmanr(py[hz][m], r[m]).correlation)
for hz in (3, 7):
    b, y = np.array(res[f'b{hz}']), np.array(res[f'y{hz}'])
    print(f'h{hz}: n={len(b)} IC Binance-входы {b.mean():+.3f} (t {b.mean()/b.std()*len(b)**.5:+.1f})  IC Bybit-входы {y.mean():+.3f} (t {y.mean()/y.std()*len(y)**.5:+.1f})')
