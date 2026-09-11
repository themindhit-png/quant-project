#!/usr/bin/env python3
import json, lightgbm as lgb
d = json.load(open('models/spec_noage.json'))
print('spec_noage: feats', len(d['features']), '| dropped', d['dropped'],
      '| train_end', d['train_end'], '| env', d['env'])
for hz in (3, 7):
    b = lgb.Booster(model_file=f'models/ml_h{hz}_noage.txt')
    print(f'  ml_h{hz}_noage.txt: num_feature =', b.num_feature(), '| trees =', b.num_trees())
d8 = json.load(open('models/spec_8h.json'))
print('spec_8h: feats', len(d8['features']), '| horizon', d8['horizon_h'],
      '| grid', d8['grid'], '| train_end', d8['train_end'])
b8 = lgb.Booster(model_file='models/ml_8h.txt')
print('  ml_8h.txt: num_feature =', b8.num_feature(), '| trees =', b8.num_trees())
