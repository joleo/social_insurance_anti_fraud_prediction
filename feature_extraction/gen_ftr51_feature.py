# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     gen_ftr512vec_feature
   Description :
   Author :       Administrator
   date：          2018/7/14 0014
-------------------------------------------------
   Change Activity:
                   2018/7/14 0014:
-------------------------------------------------
"""
__author__ = 'Administrator'
import pandas as pd
import datetime

def gen_ftr51_day_stat(df):
    df.loc[:, 'day'] = df['CREATETIME'].map(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d').day)

    # 天级别
    sum_day_col = ['PERSONID']
    day_n = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 30, 31]
    df['ftr51_num'] = df['FTR51'].map(lambda x: x.count(',')+1)

    for i in day_n:
        sum_day_col.append('sum_day_col' + str(i))
    day_sum = df.groupby(['PERSONID', 'day'])['ftr51_num'].sum().reset_index()
    ev_day_sum = pd.pivot_table(day_sum, index='PERSONID', columns='day', values='ftr51_num',
                                fill_value=0).reset_index()
    ev_day_sum.columns = sum_day_col

    return ev_day_sum


def gen_ftr51_stat(df):
    df.loc[:, 'month'] = df['CREATETIME'].map(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d').month)

    # 统计药品个数、均值、方差等信息
    df['ftr51_num'] = df['FTR51'].map(lambda x: x.count(',')+1)
    ftr51_stat = df.groupby(['PERSONID'])['ftr51_num'].agg({'ftr51_mean':'mean','ftr51_median':'median', 'ftr51_std':'std',
                                              'ftr51_max':'max', 'ftr51_min':'min','ftr51_sum':'sum'}).reset_index() # 加sum是否会有提升
    ftr51_stat.fillna(0)

    time_mon_count = df.groupby(['PERSONID', 'month'])['APPLYNO'].count().reset_index()
    time_mon_count.columns = ['PERSONID', 'month','time_mon_count']
    mon_sum = df.groupby(['PERSONID', 'month'])['ftr51_num'].sum().reset_index()


    # 月级别的药品数量
    ev_mon_sum = pd.pivot_table(mon_sum, index='PERSONID', columns='month', values='ftr51_num',
                                fill_value=0).reset_index()
    ev_mon_sum.columns = ['PERSONID', 'm_ev_mon_num1', 'm_ev_mon_num2', 'm_ev_mon_num3', 'm_ev_mon_num4', 'm_ev_mon_num5', 'm_ev_mon_num6',
                          'm_ev_mon_num7', 'm_ev_mon_num8', 'm_ev_mon_num9', 'm_ev_mon_num10', 'm_ev_mon_num11', 'm_ev_mon_num12']

    # 天级别
    # sum_day_col = ['PERSONID']
    # for i in day_n:
    #     sum_day_col.append('sum_day_col' + str(i))
    # day_sum = df.groupby(['PERSONID', 'day'])['ftr51_num'].sum().reset_index()
    # ev_day_sum = pd.pivot_table(day_sum, index='PERSONID', columns='day', values='ftr51_num',
    #                             fill_value=0).reset_index()
    # ev_day_sum.columns = sum_day_col
    # 统计常用药个数


    # 统计月级别药品平均数
    mon_count = df.groupby(['PERSONID', 'month'])['FTR51'].count().reset_index()
    mon_sum['mon_avg'] = mon_sum['ftr51_num'] / mon_count['FTR51']
    del mon_count
    ev_mon_avg = pd.pivot_table(mon_sum, index='PERSONID', columns='month', values='mon_avg',
                                        fill_value=0).reset_index()
    ev_mon_avg.columns = ['PERSONID', 'm_avg_n1', 'm_avg_n2', 'm_avg_n3', 'm_avg_n4', 'm_avg_n5', 'm_avg_n6',
                                  'm_avg_n7', 'm_avg_n8', 'm_avg_n9', 'm_avg_n10', 'm_avg_n11', 'm_avg_n12']
    del mon_sum
    day_n = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 30, 31]

    # 统计天级别药品平均数
    ev_day_col = ['PERSONID']
    for i in day_n:
        ev_day_col.append('ev_day_avg' + str(i))
    day_count = df.groupby(['PERSONID', 'day'])['FTR51'].count().reset_index()
    day_sum = df.groupby(['PERSONID', 'day'])['ftr51_num'].sum().reset_index()

    day_sum['day_avg'] = day_sum['ftr51_num'] / day_count['FTR51']
    del day_count
    ev_day_avg = pd.pivot_table(day_sum, index='PERSONID', columns='day', values='day_avg',
                                fill_value=0).reset_index()
    ev_day_avg.columns = ev_day_col


    gen_ftr51_stat = ftr51_stat.merge(ev_mon_avg, on='PERSONID', how='left')
    # gen_ftr51_stat = gen_ftr51_stat.merge(mon_access_rate, on='PERSONID', how='left') # 降分
    gen_ftr51_stat = gen_ftr51_stat.merge(ev_mon_sum, on='PERSONID', how='left')
    # gen_ftr51_stat = gen_ftr51_stat.merge(ev_day_sum, on='PERSONID', how='left')
    gen_ftr51_stat = gen_ftr51_stat.merge(ev_day_avg, on='PERSONID', how='left')
    gen_ftr51_stat.fillna(0)

    return gen_ftr51_stat

def ftr51_unique_rate(df):
    # Length(unique(FTR51))/length(FTR)
    # ftr51_unique = df.groupby(['PERSONID'])['FTR51'].unique().reset_index()
    # ftr51_unique.columns = ['PERSONID','ftr51_unique']
    # ftr51_unique.loc[:, 'ftr51_len'] = ftr51_unique['ftr51_unique'].map(lambda x:len(x))
    # ftr51_size = df.groupby(['PERSONID'])['FTR51'].size().reset_index()
    # ftr51_size.columns = ['PERSONID','ftr51_size']
    # ftr51_unique.loc[:, 'ftr51_rate'] = ftr51_unique['ftr51_len'] / ftr51_size['ftr51_size']
    df.loc[:, 'month'] = df['CREATETIME'].map(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d').month)

    ftr51_unique = df.groupby(['PERSONID', 'month'])['FTR51'].unique().reset_index()
    ftr51_unique.columns = ['PERSONID', 'month', 'ftr51_unique']
    ftr51_unique.loc[:, 'ftr51_len'] = ftr51_unique['ftr51_unique'].map(lambda x: len(x))

    ftr51_size = df.groupby(['PERSONID', 'month'])['FTR51'].size().reset_index()
    ftr51_size.columns = ['PERSONID', 'month', 'ftr51_size']
    ftr51_unique.loc[:, 'ftr51_rate'] = ftr51_unique['ftr51_len'] / ftr51_size['ftr51_size']
    ftr51_unique_rate = pd.pivot_table(ftr51_unique, index='PERSONID', columns='month', values='ftr51_rate',
                                  fill_value=0).reset_index()
    ftr51_unique_rate.columns = ['PERSONID', 'm_unique1', 'm_unique2', 'm_unique3', 'm_unique4', 'm_unique5', 'm_unique6',
                            'm_unique7', 'm_unique8', 'm_unique9', 'm_unique10', 'm_unique11', 'm_unique12']
    del ftr51_unique,ftr51_size

    return ftr51_unique_rate

def gen_ftr51_len(df):

    # FTR51长度统计信息
    df['month'] = df['CREATETIME'].map(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d').month)
    df['day'] = df['CREATETIME'].map(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d').day)

    # month
    df.loc[:, 'ftr51_len'] = df.FTR51.map(lambda x: len(x))
    ftr51_month_size = df.groupby(['PERSONID', 'month'])['ftr51_len'].agg(
        {"ftr51_sum": "sum", "ftr51_std":"std", "ftr51_mean":"mean","ftr51_min":"min","ftr51_max":"max"}).reset_index()
    ftr51_all_month_num = pd.pivot_table(ftr51_month_size, index='PERSONID', columns='month',
                                            values='ftr51_sum',fill_value=0).reset_index()
    ftr51_all_month_num.rename(columns = {1:'m_s1', 2:'m_s2', 3:'m_s3', 4:'m_s4', 5:'m_s5', 6:'m_s6', 7:'m_s7', 8:'m_s8',
                                        9:'m_s9', 10:'m_s10', 11:'m_s11', 12:'m_s12'}, inplace = True)

    ftr51_all_month_std = pd.pivot_table(ftr51_month_size, index='PERSONID', columns='month',
                                         values='ftr51_std', fill_value=0).reset_index()
    ftr51_all_month_std.rename(columns={1:'m_std1', 2:'m_std2', 3:'m_std3', 4:'m_std4', 5:'m_std5', 6:'m_std6', 7:'m_std7', 8:'m_std8',
                                        9:'m_std9', 10:'m_std10', 11:'m_std11', 12:'m_std12'}, inplace=True)
    ftr51_all_month_mean = pd.pivot_table(ftr51_month_size, index='PERSONID', columns='month',
                                         values='ftr51_mean', fill_value=0).reset_index()
    ftr51_all_month_mean.rename(columns={1:'m_mean1', 2:'m_mean2', 3:'m_mean3', 4:'m_mean4', 5:'m_mean5', 6:'m_mean6', 7:'m_mean7', 8:'m_mean8',
                                        9:'m_mean9', 10:'m_mean10', 11:'m_mean11', 12:'m_mean12'}, inplace=True)
    ftr51_all_month_min = pd.pivot_table(ftr51_month_size, index='PERSONID', columns='month',
                                          values='ftr51_min', fill_value=0).reset_index()
    ftr51_all_month_min.rename(columns={1:'m_min1', 2:'m_min2', 3:'m_min3', 4:'m_min4', 5:'m_min5', 6:'m_min6', 7:'m_min7', 8:'m_min8',
                                        9:'m_min9', 10:'m_min10', 11:'m_min11', 12:'m_min12'}, inplace=True)
    ftr51_all_month_max = pd.pivot_table(ftr51_month_size, index='PERSONID', columns='month',
                                         values='ftr51_max', fill_value=0).reset_index()
    ftr51_all_month_max.rename(columns={1:'m_max1', 2:'m_max2', 3:'m_max3', 4:'m_max4', 5:'m_max5', 6:'m_max6', 7:'m_max7', 8:'m_max8',
                                        9:'m_max9', 10:'m_max10', 11:'m_max11', 12:'m_max12'}, inplace=True)
    gen_time_feature3 = ftr51_all_month_num.merge(ftr51_all_month_std, on='PERSONID',how='left')
    gen_time_feature3 = gen_time_feature3.merge(ftr51_all_month_mean, on='PERSONID',how='left')
    gen_time_feature3 = gen_time_feature3.merge(ftr51_all_month_min, on='PERSONID',how='left')
    gen_time_feature3 = gen_time_feature3.merge(ftr51_all_month_max, on='PERSONID',how='left')
    del ftr51_month_size

    return gen_time_feature3


def ftr51_tfidf_fea(df):
    # train_path = '../data/train_texts2.csv'
    # test_path = '../data/testtexts2.csv'

    # train_texts2 = pd.read_csv(train_path)
    # testtexts2 = pd.read_csv(test_path)

    df['ftr51_w_num'] = df['word'].map(lambda x: str(x).count(" ")+1)

    return df




# if __name__ == '__main__':
    # test = pd.read_csv('../data/test.csv')
    # # user_next_time_stat = user_next_time_stat(test)
    # # print(user_next_time_stat.head().T)
    # gen_ftr_stat = gen_ftr51_stat(test)
    # # print(gen_ftr_stat.head())
    # print(gen_ftr_stat.columns)
    # print(gen_ftr_stat.shape)
    #
    # gen_ftr_stat.to_csv('../data/feature/test/gen_ftr51_stat6.csv', index=False)
    # train_path = '../data/test_B.csv'
    # test_path = '../data/testtexts2.csv'
    # train_texts2 = pd.read_csv(train_path)
    # ftr51_tfidf_fea = ftr51_tfidf_fea(train_texts2)
    # print(train_texts2.columns.tolist())
    # print(train_texts2.shape)
    # ftr51_tfidf_fea.head(100).to_csv('../data/ftr51_tfidf_fea.csv', index=False)

