# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     gen_ftr_feature
   Description :
   Author :       Administrator
   date：          2018/7/17 0017
-------------------------------------------------
   Change Activity:
                   2018/7/17 0017:
-------------------------------------------------
"""
__author__ = 'Administrator'
import pandas as pd

def gen_ftr_stat(df):
    ftr_n = [5, 7, 8, 9, 10, 12, 14, 16, 21, 28, 29, 30, 33, 36, 39, 41, 43, 47, 0,17,18,23,32,34,35,42,44,48]
    # FTR+月命名
    ftr_mean_stat_col = ['PERSONID']
    for i in ftr_n:
        ftr_mean_stat_col.append('ftr_mean' + str(i))
    values = []
    # 相似7、39 |  28 30  33 36 41  | 9  43
    for i in ftr_n:
        values.append('FTR' + str(i))
    ftr_mean_stat = df.groupby(['PERSONID'])[values].mean().reset_index()
    ftr_mean_stat.columns = ftr_mean_stat_col

    ftr_std_stat_col = ['PERSONID']
    for i in ftr_n:
        ftr_std_stat_col.append('ftr_std' + str(i))
    ftr_std_stat = df.groupby(['PERSONID'])[values].std().reset_index()
    ftr_std_stat.columns = ftr_std_stat_col

    ftr_max_stat_col = ['PERSONID']
    for i in ftr_n:
        ftr_max_stat_col.append('ftr_max' + str(i))
    ftr_max_stat = df.groupby(['PERSONID'])[values].max().reset_index()
    ftr_max_stat.columns = ftr_max_stat_col

    ftr_skew_stat_col = ['PERSONID']
    for i in ftr_n:
        ftr_skew_stat_col.append('ftr_skew' + str(i))
    ftr_skew_stat = df.groupby(['PERSONID'])[values].skew().reset_index()
    ftr_skew_stat.columns = ftr_skew_stat_col

    # 强持
    ftr_sum_stat_col = ['PERSONID']
    for i in ftr_n:
        ftr_sum_stat_col.append('ftr_sum' + str(i))
    ftr_sum_stat = df.groupby(['PERSONID'])[values].sum().reset_index()
    ftr_sum_stat.columns = ftr_sum_stat_col

    gen_ftr_stat = ftr_mean_stat.merge(ftr_std_stat, on='PERSONID', how='left')
    gen_ftr_stat = gen_ftr_stat.merge(ftr_max_stat, on='PERSONID', how='left')
    gen_ftr_stat = gen_ftr_stat.merge(ftr_skew_stat, on='PERSONID', how='left')
    gen_ftr_stat = gen_ftr_stat.merge(ftr_sum_stat, on='PERSONID', how='left')
    gen_ftr_stat.fillna(0, inplace=True)

    return gen_ftr_stat

def gen_ftr_cat(df):
    ftr_n = [1, 3, 6, 11, 19, 24, 26, 46]
    values = ['FTR1', 'FTR3', 'FTR6', 'FTR11', 'FTR19', 'FTR24', 'FTR26', 'FTR46']
    gen_ftr_catt_col = ['PERSONID']
    for i in ftr_n:
        gen_ftr_catt_col.append('ftr_cat' + str(i))
    gen_ftr_cat = df.groupby(['PERSONID'])[values].first().reset_index()
    gen_ftr_cat.columns=gen_ftr_catt_col

    return gen_ftr_cat

def gen_ftr_nunique(df):
    #
    ftr_n = [0, 5, 7, 8, 9, 10, 12, 14, 16, 21, 28, 29, 47,51]
    values = []
    # 相似7、39 |  28 30  33 36 41  | 9  43
    for i in ftr_n:
        values.append('FTR' + str(i))
    gen_ftr_col = ['PERSONID']
    for i in ftr_n:
        gen_ftr_col.append('ftr_nunique' + str(i))
    ftr_nunique = df.groupby(['PERSONID'])[values].nunique().reset_index()
    ftr_nunique.columns = gen_ftr_col#['PERSONID','ftr0_nunique','ftr51_nunique']

    ftr_rate = ['PERSONID']
    for i in ftr_n:
        ftr_rate.append('ftr_count' + str(i))
    ftr_count_rate = df.groupby(['PERSONID'])[values].count().reset_index()
    ftr_count_rate.columns = ftr_rate

    # ftr_n = [0, 5, 7, 8, 9, 10, 12, 14, 16, 21, 28, 29, 47,51]
    ftr_count_rate['ftr_rate0'] =  ftr_nunique['ftr_nunique0'] / ftr_count_rate['ftr_count0']
    ftr_count_rate['ftr_rate5'] =  ftr_nunique['ftr_nunique5'] / ftr_count_rate['ftr_count5']
    ftr_count_rate['ftr_rate7'] =  ftr_nunique['ftr_nunique7'] / ftr_count_rate['ftr_count7']
    ftr_count_rate['ftr_rate8'] =  ftr_nunique['ftr_nunique8'] / ftr_count_rate['ftr_count8']
    ftr_count_rate['ftr_rate9'] =  ftr_nunique['ftr_nunique9'] / ftr_count_rate['ftr_count9']
    ftr_count_rate['ftr_rate10'] =  ftr_nunique['ftr_nunique10'] / ftr_count_rate['ftr_count10']
    ftr_count_rate['ftr_rate12'] =  ftr_nunique['ftr_nunique12'] / ftr_count_rate['ftr_count12']
    ftr_count_rate['ftr_rate14'] =  ftr_nunique['ftr_nunique14'] / ftr_count_rate['ftr_count14']
    ftr_count_rate['ftr_rate16'] =  ftr_nunique['ftr_nunique16'] / ftr_count_rate['ftr_count16']
    ftr_count_rate['ftr_rate21'] =  ftr_nunique['ftr_nunique21'] / ftr_count_rate['ftr_count21']
    ftr_count_rate['ftr_rate28'] =  ftr_nunique['ftr_nunique28'] / ftr_count_rate['ftr_count28']
    ftr_count_rate['ftr_rate29'] =  ftr_nunique['ftr_nunique29'] / ftr_count_rate['ftr_count29']
    ftr_count_rate['ftr_rate47'] =  ftr_nunique['ftr_nunique47'] / ftr_count_rate['ftr_count47']
    ftr_count_rate['ftr_rate51'] =  ftr_nunique['ftr_nunique51'] / ftr_count_rate['ftr_count51']

    # Length(unique(FTR51)) / length(FTR)

    gen_ftr_nunique = ftr_nunique.merge(ftr_count_rate, on='PERSONID', how='left')

    return gen_ftr_nunique

def gen_ftr_sim(df):
    # ftr_n = [0,17,18,23,32,34,35,42,44,48]
    # FTR0 ,'FTR17'、,'FTR18'、,'FTR0'，,'FTR23','FTR32','FTR34','FTR35','FTR42','FTR44','FTR48'
    # gen_ftr_sim = ['PERSONID']
    # for i in ftr_n:
    #     gen_ftr_sim.append('ftr_mean' + str(i))
    df['ftr_sim'] = df['FTR0'] + df['FTR17'] + df['FTR18'] + df['FTR23'] + df['FTR32'] + df['FTR34'] + df['FTR35']
    + df['FTR42']+ df['FTR44'] + df['FTR48']
    ftr_sim_stat = df.groupby(['PERSONID'])['ftr_sim'].agg({'ftr_sim_mean': 'mean','ftr_sim_std': 'std',
                                                            'ftr_sim_std': 'std','ftr_sim_max': 'max',
                                                            'ftr_sim_skew': 'skew','ftr_sim_sum': 'sum'}).reset_index()

    # dup_data = df[['PERSONID','FTR0']].drop_duplicates(['PERSONID', 'FTR0'], keep='first', inplace=False)

    ftr_sim_stat.fillna(0, inplace=True)
    return ftr_sim_stat

import numpy as np
def gen_mod_stat(df):
    ftr_n = [5, 7, 8, 9, 10, 12, 14, 16, 21, 28, 29, 30, 33, 36, 39, 41, 43, 47, 0, 17, 18, 23, 32, 34, 35, 42, 44, 48]

    ftr_mod_stat_col = ['PERSONID']
    values = []
    for i in ftr_n:
        values.append('FTR' + str(i))
    for i in ftr_n:
        ftr_mod_stat_col.append('ftr_mod' + str(i))
    ftr_mod_stat = df.groupby('PERSONID')[values].apply(lambda x: x.mode()).reset_index()
    ftr_mod_stat = ftr_mod_stat.drop_duplicates(subset = ['PERSONID'], keep = 'last', inplace=False)
    # ftr_mod_stat.columns = ftr_mod_stat_col


    return ftr_mod_stat


# if __name__ == '__main__':
#     test = pd.read_csv('../data/train.csv')
#     gen_ftr = gen_mod_stat(test)
#
#     print(gen_ftr.shape)
#     # print(gen_ftr.head())#to_csv('../data/gen_ftr.csv', index=False))
#     gen_ftr.to_csv('../data/gen_mod_stat.csv', index=False)
#     # print(gen_ftr['PERSONID'].value_counts())