# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     feature_integrate
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

path = 'data/'
# train = pd.read_csv(path + 'train_1.csv',encoding='utf-8')

train_id = pd.read_csv('data/train_id.csv', sep='\t')
train_set = pd.read_csv('data/train.csv', sep='\t')
train = train_id.merge(train_set, on='PERSONID', how='left')
test =  pd.read_csv(path + 'test.csv', sep='\t')
fe = FeatureExtraction()

# df = 'test'
values = []
count_names = ['PERSONID']
median_names = ['PERSONID']
mean_names = ['PERSONID']
max_names = ['PERSONID']
min_names = ['PERSONID']
std_names = ['PERSONID']
for i in range(51):
    if i not in [1,3,6,11,19,24,26,46]:
        values.append('FTR' + str(i))
for i in range(51):
    if i not in [1, 3, 6, 11, 19, 24, 26, 46]:
        count_names.append('FTR_count' + str(i))
for i in range(51):
    if i not in [1, 3, 6, 11, 19, 24, 26, 46]:
        median_names.append('FTR_median' + str(i))
for i in range(51):
    if i not in [1, 3, 6, 11, 19, 24, 26, 46]:
        mean_names.append('FTR_mean' + str(i))
for i in range(51):
    if i not in [1, 3, 6, 11, 19, 24, 26, 46]:
        max_names.append('FTR_max' + str(i))
for i in range(51):
    if i not in [1, 3, 6, 11, 19, 24, 26, 46]:
        min_names.append('FTR_min' + str(i))
for i in range(51):
    if i not in [1, 3, 6, 11, 19, 24, 26, 46]:
        std_names.append('FTR_std' + str(i))

df = 'train' # train
if df == 'test':
    # gen_merge_count  = fe.gen_merge_count(test, columns='PERSONID', value=values, name=count_names)
    # gen_merge_median = fe.gen_merge_median(test, columns='PERSONID', value=values, name=median_names)
    # gen_merge_mean = fe.gen_merge_mean(test, columns='PERSONID', value=values, name=mean_names)
    # gen_merge_max = fe.gen_merge_max(test, columns='PERSONID', value=values, name=max_names)
    # gen_merge_min = fe.gen_merge_min(test, columns='PERSONID', value=values, name=min_names)
    # gen_merge_std = fe.gen_merge_std(test, columns='PERSONID', value=values, name=std_names)
    # gen_time_feature = fe.gen_time_feature(test, columns='PERSONID')
    # gen_time_feature2 = fe.gen_time_feature2(test)
    # gen_time_feature3 = fe.gen_time_feature3(test)
    gen_diff_feature = fe.gen_diff_feature(test,value=values )

    # print(gen_merge_mean.head())
    # gen_merge_count.to_csv('data/feature/test/gen_merge_count.csv', index=False)
    # gen_merge_median.to_csv('data/feature/test/gen_merge_median.csv', index=False)
    # gen_merge_mean.to_csv('data/feature/test/gen_merge_mean.csv', index=False)
    # gen_merge_max.to_csv('data/feature/test/gen_merge_max.csv', index=False)
    # gen_merge_min.to_csv('data/feature/test/gen_merge_min.csv', index=False)
    # gen_merge_std.to_csv('data/feature/test/gen_merge_std.csv', index=False)
    # gen_time_feature.to_csv('data/feature/test/gen_time_feature.csv', index=False)
    # gen_time_feature2.to_csv('data/feature/test/gen_time_feature2.csv', index=False)
    # gen_time_feature3.to_csv('data/feature/test/gen_time_feature3.csv', index=False)
    gen_diff_feature.to_csv('data/feature/test/gen_diff_feature.csv', index=False)

else:
    # gen_merge_count = fe.gen_merge_count(train, columns='PERSONID', value=values, name=median_names)
    # gen_merge_median = fe.gen_merge_median(train, columns='PERSONID', value=values, name=median_names)
    # gen_merge_mean = fe.gen_merge_mean(train, columns='PERSONID', value=values, name=mean_names)
    # gen_merge_max = fe.gen_merge_max(train, columns='PERSONID', value=values, name=max_names)
    # gen_merge_min = fe.gen_merge_min(train, columns='PERSONID', value=values, name=min_names)
    # gen_merge_std = fe.gen_merge_std(train, columns='PERSONID', value=values, name=std_names)
    # gen_time_feature = fe.gen_time_feature(train, columns='PERSONID')
    # gen_time_feature2 = fe.gen_time_feature2(train)
    # gen_time_feature3 = fe.gen_time_feature3(train)
    gen_diff_feature = fe.gen_diff_feature(train,value=values )

    # gen_merge_count.to_csv('data/feature/train/gen_merge_count.csv', index=False)
    # gen_merge_median.to_csv('data/feature/train/gen_merge_median.csv', index=False)
    # gen_merge_mean.to_csv('data/feature/train/gen_merge_mean.csv', index=False)
    # gen_merge_max.to_csv('data/feature/train/gen_merge_max.csv', index=False)
    # gen_merge_min.to_csv('data/feature/train/gen_merge_min.csv', index=False)
    # gen_merge_std.to_csv('data/feature/train/gen_merge_std.csv', index=False)
    # gen_time_feature.to_csv('data/feature/train/gen_time_feature.csv', index=False)
    # gen_time_feature2.to_csv('data/feature/train/gen_time_feature2.csv', index=False)
    # gen_time_feature3.to_csv('data/feature/train/gen_time_feature3.csv', index=False)
    gen_diff_feature.to_csv('data/feature/train/gen_diff_feature.csv', index=False)



