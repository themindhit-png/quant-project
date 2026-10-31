#!/usr/bin/env python3
"""Tables for docs/REVERSE_2020_21.md from research/phase4_out/{REV,FWD}_*.csv (research/rev_book.py) and R4_E_V2 / R4_new_sleeves.
Run: python3 research/rev_report.py > logs/rev_report.txt"""
import math, os, numpy as np, pandas as pd
P = 'research/phase4_out/'
def rd(f): return pd.read_csv(P + f, index_col=0, parse_dates=True)
def sh(x): x = x.dropna(); return x.mean() / x.std(ddof=1) * math.sqrt(365) if len(x) > 20 and x.std() > 0 else float('nan')
def tot(x): return (1 + x.dropna()).prod() - 1
def mdd(x): e = (1 + x.dropna()).cumprod(); e = pd.concat([pd.Series([1.0]), e]); return (e / e.cummax() - 1).min()
def beta(y, f):
    j = pd.concat([y, f], axis=1).dropna()
    if len(j) < 20: return float('nan'), float('nan')
    a, b = j.iloc[:, 0].values, j.iloc[:, 1].values; bb = np.cov(a, b)[0, 1] / b.var(ddof=1); res = a - a.mean() - bb * (b - b.mean())
    se = math.sqrt(res.var(ddof=2) / (b.var(ddof=1) * (len(a) - 1))); return bb, bb / se
def p(v, k=1): return 'nan' if not np.isfinite(v) else f'{v*100:+.{k}f}'
fac = rd('REV_factor.csv'); F = fac['alt_minus_btc']; B = fac['btc']
rb = rd('REV_book.csv'); a, a15 = rb['F2V2_x1'], rb['F2V2_x1.5']
fw = rd('R4_E_V2.csv'); f1, f15 = fw['lag1_x1'], fw['lag1_x1.5']
W = {'2020 (06-12)': ('2020-06-01', '2020-12-31'), '2021': ('2021-01-01', '2021-12-31'), 'альт-сезон 1: 2021-01..05': ('2021-01-01', '2021-05-31'),
     'альт-сезон 2: 2021-08..11': ('2021-08-01', '2021-11-30'), 'весь реверс 2020-06..2021-12': ('2020-06-01', '2021-12-31')}
print('## Книга F2+V2 (lag 1 ч)\n\n| период | Sharpe ×1 | Sharpe ×1.5 | доходность ×1, % | MDD, % | бета к «альты−BTC» (t) | бета к BTC | альты−BTC за период, % | BTC, % |\n|---|---|---|---|---|---|---|---|---|')
for k, (s, e) in W.items():
    x = a[s:e]; bt_, t_ = beta(x, F[s:e]); bb, _ = beta(x, B[s:e])
    print(f'| {k} | {sh(x):.2f} | {sh(a15[s:e]):.2f} | {p(tot(x))} | {p(mdd(x))} | {bt_:+.3f} ({t_:+.1f}) | {bb:+.3f} | {p(((1+F[s:e]).prod()-1))} | {p(tot(B[s:e]),0)} |')
for k, (s, e) in {'2022-01..2026-10 (R4_E_V2)': ('2022-01-01', '2026-10-26')}.items():
    x = f1[s:e]; bt_, t_ = beta(x, F[s:e]); bb, _ = beta(x, B[s:e])
    print(f'| {k} | {sh(x):.2f} | {sh(f15[s:e]):.2f} | {p(tot(x))} | {p(mdd(x))} | {bt_:+.3f} ({t_:+.1f}) | {bb:+.3f} | {p(((1+F[s:e]).prod()-1))} | {p(tot(B[s:e]),0)} |')
# months where alts beat BTC, both periods
for nm, x in (('реверс', a['2020-06-01':]), ('2022-26', f1)):
    mm = pd.concat([x, F], axis=1, join='inner'); mm.columns = ['b', 'f']; g = mm.groupby(mm.index.to_period('M')).apply(lambda z: pd.Series({'b': (1+z.b).prod()-1, 'f': (1+z.f).prod()-1}))
    up, dn = g[g.f > 0.05], g[g.f < -0.05]
    print(f'\n{nm}: месяцы альты−BTC > +5%: {len(up)} шт., книга в среднем {p(up.b.mean(),2)}%/мес; < −5%: {len(dn)} шт., {p(dn.b.mean(),2)}%/мес; остальные {p(g[(g.f.abs()<=0.05)].b.mean(),2)}%/мес')
print('\n## По кварталам (реверс, ×1)\n\n| квартал | книга, % | Sharpe | альты−BTC, % | BTC, % |\n|---|---|---|---|---|')
for q, x in a['2020-06-01':].groupby(a['2020-06-01':].index.to_period('Q')):
    s, e = x.index[0], x.index[-1]
    print(f'| {q} | {p(tot(x))} | {sh(x):.2f} | {p((1+F[s:e]).prod()-1)} | {p(tot(B[s:e]),0)} |')
print('\nХудшие дни реверса: ' + ', '.join(f'{d.date()} {v*100:+.2f}% (альты−BTC {F.get(d, np.nan)*100:+.1f}%)' for d, v in a.nsmallest(8).items()))
print('Лучшие дни реверса: ' + ', '.join(f'{d.date()} {v*100:+.2f}%' for d, v in a.nlargest(5).items()))
print('Худшие дни 2022-26: ' + ', '.join(f'{d.date()} {v*100:+.2f}%' for d, v in f1.nsmallest(5).items()))
# sleeves
rs = rd('REV_sleeves.csv'); ns = rd('R4_new_sleeves.csv')
print('\n## Рукава отдельно (lag 1 ч, ×1), Sharpe\n\n| рукав | 2020-06..12 | 2021 | АС1 21-01..05 | АС2 21-08..11 | реверс всего | бета к «альты−BTC» реверс | 2022-26 (R4) | бета 2022-26 |\n|---|---|---|---|---|---|---|---|---|')
for c in rs.columns:
    x = rs[c]; y = ns[c] if c in ns else pd.Series(dtype=float)
    print(f'| {c} | {sh(x["2020-06-01":"2020-12-31"]):.2f} | {sh(x["2021"]):.2f} | {sh(x["2021-01-01":"2021-05-31"]):.2f} | {sh(x["2021-08-01":"2021-11-30"]):.2f} | {sh(x):.2f} | {beta(x, F)[0]:+.3f} | {sh(y):.2f} | {beta(y, F)[0]:+.3f} |')
print('\nДоходность рукавов, % (реверс): ' + '; '.join(f'{c} 2020 {p(tot(rs[c]["2020"]))} / 2021 {p(tot(rs[c]["2021"]))}' for c in rs.columns))
# guards
for tag, base1, base15, s0, e0 in (('REV', a, a15, '2020-06-01', '2021-12-31'), ('FWD', None, None, '2022-01-01', '2026-10-26')):
    if not os.path.exists(P + f'{tag}_guards.csv'): continue
    g = rd(f'{tag}_guards.csv')
    if tag == 'FWD': bk = rd('FWD_book.csv'); base1, base15 = bk['F2V2_x1'], bk['F2V2_x1.5']
    win = {'REV': [('всё', s0, e0), ('2020', '2020-06-01', '2020-12-31'), ('2021', '2021-01-01', '2021-12-31'), ('АС1', '2021-01-01', '2021-05-31'), ('АС2', '2021-08-01', '2021-11-30')],
           'FWD': [('всё', s0, e0), ('DEV 22-01..26-02', '2022-01-01', '2026-02-28'), ('HOLD 26-03..10', '2026-03-01', '2026-10-26')] + [(str(y), f'{y}-01-01', f'{y}-12-31') for y in range(2022, 2027)]}[tag]
    print(f'\n## Фильтры {tag} (Sharpe ×1 / ×1.5; доходность ×1 %; MDD %)\n\n| вариант | ' + ' | '.join(w[0] for w in win) + ' |\n|' + '---|' * (len(win) + 1))
    rows = [('без фильтра', base1, base15)] + [(c[:-3], g[c], g[c[:-3] + '_x1.5']) for c in g.columns if c.endswith('_x1')]
    for nm, x1, x15 in rows:
        print(f'| {nm} | ' + ' | '.join(f'{sh(x1[s:e]):.2f} / {sh(x15[s:e]):.2f}; {p(tot(x1[s:e]),0)}; {p(mdd(x1[s:e]),1)}' for _, s, e in win) + ' |')
fl = fac[['btc30', 'alt30_minus_btc30']]
for nm, (s, e) in (('реверс', ('2020-06-01', '2021-12-31')), ('2022-26', ('2022-01-01', '2026-10-26'))):
    z = fl[s:e]; print(f'\nДоля дней с флагом ({nm}): btc15 {(z.btc30 > 0.15).mean()*100:.0f}%, alt10 {(z.alt30_minus_btc30 > 0.10).mean()*100:.0f}%')
