# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     feature_integrate2
   Description :
   Author :       Administrator
   date：          2018/7/13 0013
-------------------------------------------------
   Change Activity:
                   2018/7/13 0013:
-------------------------------------------------
"""
__author__ = 'Administrator'
import pandas as pd

train_path = '../data/feature/train/'
test_path = '../data/feature/test/'

def train():
    # 加载特征
    gen_access_num = pd.read_csv(train_path + 'gen_access_num.csv')
    gen_ftr51_stat = pd.read_csv(train_path + 'gen_ftr51_stat.csv')
    gen_ftr_stat = pd.read_csv(train_path + 'gen_ftr_stat.csv')
    gen_ftr_cat = pd.read_csv(train_path + 'gen_ftr_cat.csv')
    user_next_time_stat = pd.read_csv(train_path + 'user_next_time_stat.csv')
    gen_ftr_nunique = pd.read_csv(train_path + 'gen_ftr_nunique.csv')
    ftr51_unique_rate = pd.read_csv(train_path + 'ftr51_unique_rate.csv')
    # gen_mod_stat = pd.read_csv(train_path + 'gen_mod_stat.csv')

    train = gen_access_num.merge(gen_ftr51_stat, on='PERSONID', how='left')
    train = train.merge(gen_ftr_stat, on='PERSONID', how='left')
    # train = train.merge(gen_ftr_cat, on='PERSONID', how='left')
    # train = train.merge(gen_mod_stat, on='PERSONID', how='left')
    train = train.merge(user_next_time_stat, on='PERSONID', how='left')
    train = train.merge(gen_ftr_nunique, on='PERSONID', how='left')
    train = train.merge(ftr51_unique_rate, on='PERSONID', how='left')

    return train

def test():
    gen_access_num = pd.read_csv(test_path + 'gen_access_num.csv')
    gen_ftr51_stat = pd.read_csv(test_path + 'gen_ftr51_stat.csv')
    gen_ftr_stat = pd.read_csv(test_path + 'gen_ftr_stat.csv')
    gen_ftr_cat = pd.read_csv(test_path + 'gen_ftr_cat.csv')
    user_next_time_stat = pd.read_csv(test_path + 'user_next_time_stat.csv')
    gen_ftr_nunique = pd.read_csv(test_path + 'gen_ftr_nunique.csv')
    ftr51_unique_rate = pd.read_csv(test_path + 'ftr51_unique_rate.csv')
    # gen_mod_stat = pd.read_csv(test_path + 'gen_mod_stat.csv')

    test = gen_access_num.merge(gen_ftr51_stat, on='PERSONID', how='left')
    test = test.merge(gen_ftr_stat, on='PERSONID', how='left')
    # test = test.merge(gen_ftr_cat, on='PERSONID', how='left')
    # test = test.merge(gen_mod_stat, on='PERSONID', how='left')
    test = test.merge(user_next_time_stat, on='PERSONID', how='left')
    test = test.merge(gen_ftr_nunique, on='PERSONID', how='left')
    test = test.merge(ftr51_unique_rate, on='PERSONID', how='left')

    return test
