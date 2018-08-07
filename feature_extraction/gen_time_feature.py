# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     gen_time_feature
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
import numpy as np
import datetime
# from utils import *
"""
提取：时间统计特征
1、统计信息
2、假期信息
"""
def user_next_time_stat(df):
    # 计算用户下一次的时间差特征
    time_data = df.loc[:, ['PERSONID', 'CREATETIME']]
    import time

    time_data['timestamp'] = time_data['CREATETIME'].map(lambda x: time.mktime(time.strptime(x, "%Y-%m-%d")))
    time_sort_dup_data = time_data.drop_duplicates(['PERSONID', 'timestamp'], keep='first', inplace=False)
    del time_data
    time_sort_dup_data.loc[:, 'next_time'] = time_sort_dup_data.groupby(['PERSONID'])['timestamp'].diff(-1).apply(
        np.abs)  # 对时间求差分
    time_sort_dup_data.loc[:, 'diff_day'] = time_sort_dup_data['next_time'].map(lambda x: x / (24 * 60 * 60)) # 天级别

    # 用户花的时间平均值、方差、最大最小值
    diff_day_stat = time_sort_dup_data.groupby(['PERSONID'], as_index=False)['diff_day'].agg({
        'diff_day_mean': np.mean,
        'diff_day_std': np.std,
        'diff_day_min': np.min,
        'diff_day_max': np.max,
        'diff_day_sum': np.sum # 累加特征
    })
    del time_sort_dup_data

    # 平均时间行为时间间隔



    # 在这个差值内，用户一些异常行为或者偏好




    diff_day_stat.fillna(0)
    return diff_day_stat

def gen_window_stat(df):
    # 窗口特征用不了
    return

def gen_access_num(df):
    # 按照时间(月、日)来统计applyno次数,39个特征
    # 时间维度
    df.loc[:, 'month'] = df['CREATETIME'].map(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d').month)
    df.loc[:, 'day'] = df['CREATETIME'].map(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d').day)

    # 按月
    time_mon_count = df.groupby(['PERSONID', 'month'])['FTR51'].count().reset_index()
    time_mon_count.columns = ['PERSONID', 'month','time_mon_count']

    time_evd_mon_count = pd.pivot_table(time_mon_count, index='PERSONID', columns='month', values='time_mon_count',
                                fill_value=0).reset_index()
    time_evd_mon_count.columns = ['PERSONID','m_count1','m_count2','m_count3','m_count4','m_count5','m_count6',
                                  'm_count7','m_count8','m_count9','m_count10','m_count11','m_count12']

    #按天
    time_day_count = df.groupby(['PERSONID', 'day'])['FTR51'].count().reset_index()
    time_day_count.columns = ['PERSONID', 'day', 'time_day_count']
    del df

    time_evd_day_count = pd.pivot_table(time_day_count, index='PERSONID', columns='day', values='time_day_count',
                                        fill_value=0).reset_index()
    # 为什么25到29没有数据？
    time_evd_day_count.columns = ['PERSONID', 'd_count1', 'd_count2', 'd_count3', 'd_count4', 'd_count5', 'd_count6',
                                  'd_count7', 'd_count8', 'd_count9', 'd_count10', 'd_count11', 'd_count12','d_count13',
                                  'd_count14', 'd_count15','d_count16','d_count17','d_count18','d_count19','d_count20',
                                  'd_count21','d_count22','d_count23','d_count24','d_count30','d_count31']
    # 按非工作日


    # 按周


    gen_access_num = time_evd_mon_count.merge(time_evd_day_count, on='PERSONID', how='left')
    del time_evd_mon_count,time_evd_day_count
    gen_access_num.fillna(0, inplace=True)
    return gen_access_num

def gen_holoday_stat(df):
    # 统计每个月假期次数 / 每个月总次数 占比
    pass




def gen_ftr_mean_stat(df):

    df['month'] = df['CREATETIME'].map(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d').month)
    ftr_n = [5,6,7,8,9,10,12,14,16,21,28,29,30,33,36,39,41,43,47]
    # 均值: 229
    # FTR+月命名
    ftr_mean_stat_col = ['PERSONID','month']
    for i in ftr_n:
        ftr_mean_stat_col.append('ftr_mean' + str(i))
    # FTRX命名
    ftr_mean_col = ['PERSONID']
    # ftr_m = [5,6]
    for j in ftr_n:
        for i in range(12):
            ftr_mean_col.append('ftr' + str(j) + '_mean' + str(i))
    # values
    ftr_values = []
    for i in ftr_n:
        ftr_values.append('ftr_mean' + str(i))

    # if i not in [1, 3, 6, 11, 19, 24, 26, 46]:
    #FTR0 ,'FTR17'、,'FTR18'、,'FTR0'，,'FTR23','FTR32','FTR34','FTR35','FTR42','FTR44','FTR48'相似
    values =['FTR5','FTR6','FTR7','FTR8','FTR9','FTR10','FTR12','FTR14','FTR16','FTR21'
        ,'FTR28','FTR29','FTR30','FTR33','FTR36','FTR39','FTR41','FTR43','FTR47']
    ftr_mean_stat = df.groupby(['PERSONID','month'])[values].mean().reset_index()
    ftr_mean_stat.columns = ftr_mean_stat_col

    ftr5_mon_mean_stat = pd.pivot_table(ftr_mean_stat, index='PERSONID', columns='month',
                                        values=ftr_values,
                                        fill_value=0).reset_index()
    ftr5_mon_mean_stat.columns = ftr_mean_col

    return ftr5_mon_mean_stat


def gen_ftr_median_stat(df):
    df['month'] = df['CREATETIME'].map(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d').month)
    ftr_n = [5, 6, 7, 8, 9, 10, 12, 14, 16, 21, 28, 29, 30, 33, 36, 39, 41, 43, 47]
    # 均值: 229
    # FTR+月命名
    ftr_median_stat_col = ['PERSONID', 'month']
    for i in ftr_n:
        ftr_median_stat_col.append('ftr_median' + str(i))
    # FTRX命名
    ftr_median_col = ['PERSONID']
    # ftr_m = [5,6]
    for j in ftr_n:
        for i in range(12):
            ftr_median_col.append('ftr' + str(j) + '_median' + str(i))
    # values
    ftr_values = []
    for i in ftr_n:
        ftr_values.append('ftr_median' + str(i))

    # if i not in [1, 3, 6, 11, 19, 24, 26, 46]
    # FTR0 ,'FTR17'、,'FTR18'、,'FTR0'，,'FTR23','FTR32','FTR34','FTR35','FTR42','FTR44','FTR48'相似
    values = ['FTR5', 'FTR6', 'FTR7', 'FTR8', 'FTR9', 'FTR10', 'FTR12', 'FTR14', 'FTR16', 'FTR21'
        , 'FTR28', 'FTR29', 'FTR30', 'FTR33', 'FTR36', 'FTR39', 'FTR41', 'FTR43', 'FTR47']
    ftr_median_stat = df.groupby(['PERSONID', 'month'])[values].median().reset_index()
    ftr_median_stat.columns = ftr_median_stat_col

    ftr5_mon_median_stat = pd.pivot_table(ftr_median_stat, index='PERSONID', columns='month',
                                          values=ftr_values,
                                          fill_value=0).reset_index()
    ftr5_mon_median_stat.columns = ftr_median_col

    return ftr5_mon_median_stat

def gen_ftr_std_stat(df):
    df['month'] = df['CREATETIME'].map(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d').month)
    ftr_n = [5, 6, 7, 8, 9, 10, 12, 14, 16, 21, 28, 29, 30, 33, 36, 39, 41, 43, 47]
    # 均值: 229
    # FTR+月命名
    ftr_std_stat_col = ['PERSONID', 'month']
    for i in ftr_n:
        ftr_std_stat_col.append('ftr_std' + str(i))
    # FTRX命名
    ftr_std_col = ['PERSONID']
    # ftr_m = [5,6]
    for j in ftr_n:
        for i in range(12):
            ftr_std_col.append('ftr' + str(j) + '_std' + str(i))
    # values
    ftr_values = []
    for i in ftr_n:
        ftr_values.append('ftr_std' + str(i))

    # if i not in [1, 3, 6, 11, 19, 24, 26, 46]:
    # FTR0 ,'FTR17'、,'FTR18'、,'FTR0'，,'FTR23','FTR32','FTR34','FTR35','FTR42','FTR44','FTR48'相似
    values = ['FTR5', 'FTR6', 'FTR7', 'FTR8', 'FTR9', 'FTR10', 'FTR12', 'FTR14', 'FTR16', 'FTR21'
        , 'FTR28', 'FTR29', 'FTR30', 'FTR33', 'FTR36', 'FTR39', 'FTR41', 'FTR43', 'FTR47']
    ftr_std_stat = df.groupby(['PERSONID', 'month'])[values].std().reset_index()
    ftr_std_stat.columns = ftr_std_stat_col

    ftr5_mon_std_stat = pd.pivot_table(ftr_std_stat, index='PERSONID', columns='month',
                                       values=ftr_values,
                                       fill_value=0).reset_index()
    ftr5_mon_std_stat.columns = ftr_std_col

    return ftr5_mon_std_stat


def gen_ftr_min_stat(df):
    df['month'] = df['CREATETIME'].map(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d').month)
    ftr_n = [5, 6, 7, 8, 9, 10, 12, 14, 16, 21, 28, 29, 30, 33, 36, 39, 41, 43, 47]
    # 均值: 229
    # FTR+月命名
    ftr_min_stat_col = ['PERSONID', 'month']
    for i in ftr_n:
        ftr_min_stat_col.append('ftr_min' + str(i))
    # FTRX命名
    ftr_min_col = ['PERSONID']
    # ftr_m = [5,6]
    for j in ftr_n:
        for i in range(12):
            ftr_min_col.append('ftr' + str(j) + '_min' + str(i))
    # values
    ftr_values = []
    for i in ftr_n:
        ftr_values.append('ftr_min' + str(i))

    # if i not in [1, 3, 6, 11, 19, 24, 26, 46]:
    # FTR0 ,'FTR17'、,'FTR18'、,'FTR0'，,'FTR23','FTR32','FTR34','FTR35','FTR42','FTR44','FTR48'相似
    values = ['FTR5', 'FTR6', 'FTR7', 'FTR8', 'FTR9', 'FTR10', 'FTR12', 'FTR14', 'FTR16', 'FTR21'
        , 'FTR28', 'FTR29', 'FTR30', 'FTR33', 'FTR36', 'FTR39', 'FTR41', 'FTR43', 'FTR47']
    ftr_min_stat = df.groupby(['PERSONID', 'month'])[values].min().reset_index()
    ftr_min_stat.columns = ftr_min_stat_col

    ftr5_mon_min_stat = pd.pivot_table(ftr_min_stat, index='PERSONID', columns='month',
                                       values=ftr_values,
                                       fill_value=0).reset_index()
    ftr5_mon_min_stat.columns = ftr_min_col

    return ftr5_mon_min_stat


def gen_ftr_max_stat(df):
    df['month'] = df['CREATETIME'].map(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d').month)
    ftr_n = [5, 6, 7, 8, 9, 10, 12, 14, 16, 21, 28, 29, 30, 33, 36, 39, 41, 43, 47]
    # 均值: 229
    # FTR+月命名
    ftr_max_stat_col = ['PERSONID', 'month']
    for i in ftr_n:
        ftr_max_stat_col.append('ftr_max' + str(i))
    # FTRX命名
    ftr_max_col = ['PERSONID']
    # ftr_m = [5,6]
    for j in ftr_n:
        for i in range(12):
            ftr_max_col.append('ftr' + str(j) + '_max' + str(i))
    # values
    ftr_values = []
    for i in ftr_n:
        ftr_values.append('ftr_max' + str(i))

    # if i not in [1, 3, 6, 11, 19, 24, 26, 46]:
    # FTR0 ,'FTR17'、,'FTR18'、,'FTR0'，,'FTR23','FTR32','FTR34','FTR35','FTR42','FTR44','FTR48'相似
    values = ['FTR5', 'FTR6', 'FTR7', 'FTR8', 'FTR9', 'FTR10', 'FTR12', 'FTR14', 'FTR16', 'FTR21'
        , 'FTR28', 'FTR29', 'FTR30', 'FTR33', 'FTR36', 'FTR39', 'FTR41', 'FTR43', 'FTR47']
    ftr_max_stat = df.groupby(['PERSONID', 'month'])[values].max().reset_index()
    ftr_max_stat.columns = ftr_max_stat_col

    ftr5_mon_max_stat = pd.pivot_table(ftr_max_stat, index='PERSONID', columns='month',
                                       values=ftr_values,
                                       fill_value=0).reset_index()
    ftr5_mon_max_stat.columns = ftr_max_col

    return ftr5_mon_max_stat

holoday = []

if __name__ == '__main__':
    # test = pd.read_csv('../data/test.csv')
    train = pd.read_csv('../data/train.csv')
    # gen_access_num = gen_access_num(test)
    gen_access_num2 = gen_access_num(train)

    # gen_ftr_stat = gen_ftr_stat(test)
    # print(gen_access_num.shape)
    # print(gen_access_num.to_csv('../data/day1.csv', index=False))
    print(gen_access_num2.to_csv('../data/day2.csv', index=False))
