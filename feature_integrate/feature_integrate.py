# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     feature_integrate2
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

train_path = 'data/feature/train/'
test_path = 'data/feature/test/'

def train_set():
    # gen_merge_count = pd.read_csv(train_path + 'gen_merge_count.csv')
    gen_merge_median = pd.read_csv(train_path + 'gen_merge_median.csv')
    # gen_merge_mean = pd.read_csv(train_path + 'gen_merge_mean.csv')
    gen_merge_max = pd.read_csv(train_path + 'gen_merge_max.csv')
    gen_merge_min = pd.read_csv(train_path + 'gen_merge_min.csv')
    gen_merge_std = pd.read_csv(train_path + 'gen_merge_std.csv')
    cat_feature = pd.read_csv(train_path + 'cat_feature.csv')
    gen_time_feature = pd.read_csv(train_path + 'gen_time_feature.csv')
    gen_time_feature2 = pd.read_csv(train_path + 'gen_time_feature2.csv')
    gen_time_feature3 = pd.read_csv(train_path + 'gen_time_feature3.csv')
    gen_diff_feature = pd.read_csv(train_path + 'gen_diff_feature.csv')

    # merge
    # train = gen_merge_count.merge(gen_merge_median, on='PERSONID', how='left')
    # train = gen_merge_median.merge(gen_merge_mean, on='PERSONID', how='left')
    train = gen_merge_median.merge(gen_merge_max, on='PERSONID', how='left')
    train = train.merge(gen_merge_min, on='PERSONID', how='left')
    train = train.merge(gen_merge_std, on='PERSONID', how='left')
    train = train.merge(cat_feature, on='PERSONID', how='left')
    # train = train.merge(gen_time_feature, on='PERSONID', how='left')
    train = train.merge(gen_time_feature2, on='PERSONID', how='left')
    train = train.merge(gen_time_feature3, on='PERSONID', how='left')
    # train = train.merge(gen_diff_feature, on='PERSONID', how='left')

    return train

def test_set():
    # gen_merge_count = pd.read_csv(test_path + 'gen_merge_count.csv')
    gen_merge_median = pd.read_csv(test_path + 'gen_merge_median.csv')
    # gen_merge_mean = pd.read_csv(test_path + 'gen_merge_mean.csv')
    gen_merge_max = pd.read_csv(test_path + 'gen_merge_max.csv')
    gen_merge_min = pd.read_csv(test_path + 'gen_merge_min.csv')
    gen_merge_std = pd.read_csv(test_path + 'gen_merge_std.csv')
    cat_feature = pd.read_csv(test_path + 'cat_feature.csv')
    gen_time_feature = pd.read_csv(test_path + 'gen_time_feature.csv')
    gen_time_feature2 = pd.read_csv(test_path + 'gen_time_feature2.csv')
    gen_time_feature3 = pd.read_csv(test_path + 'gen_time_feature3.csv')
    gen_diff_feature = pd.read_csv(test_path + 'gen_diff_feature.csv')

    # merge
    # test = gen_merge_count.merge(gen_merge_median, on='PERSONID', how='left')
    # test = gen_merge_median.merge(gen_merge_mean, on='PERSONID', how='left')
    test = gen_merge_median.merge(gen_merge_max, on='PERSONID', how='left')
    test = test.merge(gen_merge_min, on='PERSONID', how='left')
    test = test.merge(gen_merge_std, on='PERSONID', how='left')
    test = test.merge(cat_feature, on='PERSONID', how='left')
    # test = test.merge(gen_time_feature, on='PERSONID', how='left')
    test = test.merge(gen_time_feature2, on='PERSONID', how='left')
    test = test.merge(gen_time_feature3, on='PERSONID', how='left')
    # test = test.merge(gen_diff_feature, on='PERSONID', how='left')

    return test