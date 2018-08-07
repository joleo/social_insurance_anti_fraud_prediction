# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     feature_integreate3
   Description :
   Author :       Administrator
   date：          2018/7/12 0012
-------------------------------------------------
   Change Activity:
                   2018/7/12 0012:
-------------------------------------------------
"""
__author__ = 'Administrator'
import pandas as pd
from feature_extraction.feature_extraction import *

fe = FeatureExtraction()
values = ['PERSONID']
for i in [1, 3, 6, 11, 19, 24, 26, 46]:
    values.append('FTR' + str(i))

train_path = 'data/feature/train/'
test_path = 'data/feature/test/'
train_id = pd.read_csv('data/train_id.csv', sep='\t')
train_set = pd.read_csv('data/train.csv', sep='\t')
train = train_id.merge(train_set, on='PERSONID', how='left')[values]
test =  pd.read_csv('data/test.csv', sep='\t')[values]

# col = [col for col in values if col not in ['PERSONID']]
# train_data = train.groupby('PERSONID', as_index=False)[col].first()
# test_data = test.groupby('PERSONID', as_index=False)[col].first()
# train_data.to_csv(train_path + 'cat_feature.csv')
# test_data.to_csv(test_path + 'cat_feature.csv')




# # print(test_data.head())
df = 'test'

if df == 'train':
    cat_feature = pd.read_csv(train_path + 'cat_feature.csv')
    train = pd.read_csv(train_path + 'train.csv')
    train = cat_feature.merge(train, on='PERSONID', how='left')
    train.to_csv(train_path + 'train.csv', index=False)
else:
    cat_feature = pd.read_csv(test_path + 'cat_feature.csv')
    test = pd.read_csv(test_path + 'test.csv')
    test = cat_feature.merge(test, on='PERSONID', how='left')
    test.to_csv(test_path + 'test.csv', index=False)