#!/usr/bin/env python3
"""Audit E: summary table of all runs (project engine P vs independent E and its variants) -> logs/E_table.md"""
import os, pickle, numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); os.chdir(ROOT)
def daily(e): return e.groupby(e.index.floor('D')).last().pct_change().dropna()
def sh(r): return r.mean()/r.std()*np.sqrt(365)
def mdd(e): return (e/e.cummax()-1).min()
P={n:pickle.load(open(f'logs/E_proj_{n}.pkl','rb')) for n in ['core_raw','fcarry_raw','core_full','fcarry_full','combo_full']}
rows=[]
def add(label, ea, ref):
    ra=daily(ea); rb=daily(P[ref]['eq']); j=ra.index.intersection(rb.index)
    y={yy:(1+g).prod()-1 for yy,g in ra.groupby(ra.index.year)}
    cagr=(ea.iloc[-1]/ea.iloc[0])**(365.25/(ea.index[-1]-ea.index[0]).days)-1
    rows.append([label, f'{ra[j].corr(rb[j]):.4f}', f'{sh(ra):.2f}', f'{cagr*100:.1f}', f'{mdd(ea)*100:.1f}']+[f'{y[k]*100:.1f}' for k in sorted(y)]+[f'{ea.iloc[-1]-P[ref]["eq"].iloc[-1]:+.2f}'])
for n in P:
    add('P (bt.run) '+n, P[n]['eq'], n); add('E indep '+n, pickle.load(open(f'logs/E_indep_{n}.pkl','rb'))['eq'], n)
for n in ['fcarry_raw','fcarry_full','combo_full']:
    add('E tie=numpy-default '+n, pickle.load(open(f'logs/E_indep_{n}_npd.pkl','rb'))['eq'], n)
    for s in [1,2,3,5]: add(f'E tie=random seed{s} '+n, pickle.load(open(f'logs/E_indep_{n}_rnd{s}.pkl','rb'))['eq'], n)
for n in ['core_full','fcarry_full','combo_full']: add('E FREEZE '+n, pickle.load(open(f'logs/E_indep_{n}_frz.pkl','rb'))['eq'], n)
add('E remask combo_full', pickle.load(open('logs/E_indep_combo_full_remask.pkl','rb'))['eq'], 'combo_full')
add('E VT-paper combo_full', pickle.load(open('logs/E_indep_combo_full_vtp.pkl','rb'))['eq'], 'combo_full')
hdr=['run','corr daily vs P','Sharpe','CAGR %','MDD %','2022 %','2023 %','2024 %','2025 %','2026 % (to 08-31)','end $ − end $ P']
md='| '+' | '.join(hdr)+' |\n|'+'---|'*len(hdr)+'\n'+'\n'.join('| '+' | '.join(r)+' |' for r in rows)+'\n'
open('logs/E_table.md','w').write(md); print(md)
