# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     feature_integrate3
   Description :
   Author :       Administrator
   date：          2018/7/16 0016
-------------------------------------------------
   Change Activity:
                   2018/7/16 0016:
-------------------------------------------------
"""
__author__ = 'Administrator'
from feature_extraction.gen_time_feature import  *
from feature_extraction.gen_ftr51_feature import  *
from feature_extraction.gen_ftr_feature import *
import pandas as pd

flag = 'gen_feature' # gen_feature
if flag == 'gen_data':
    # 训练样本 ：(1368146, 55)，总计15000个不同的用户，日期为2015-3-1至2016-2-23，不均匀分布,正负bi：14230:770
    # data_path = "./data/train.tsv"
    # data_id = "./data/train_id.tsv"
    data_path = "../data/train.tsv"
    data_id = "../data/train_id.tsv"
    dat = pd.read_csv(data_path, sep='\t')
    dat_id = pd.read_csv(data_id,sep="\t")

    train_data  = pd.merge(dat,dat_id, how="inner", on="PERSONID")
    print(train_data.shape)

    # # 删除每一行中空值超过40个的行 少了20万
    # # [False if  train_data.loc[i,:].values.tolist().count(0) > 40  else True for i in train_data.index]
    # ind = [False if i > 40 else True for i in (train_data == 0).astype(int).sum(axis=1)]
    # train_data = train_data.loc[ind,:]
    # 需要预测的样本 ： (232502, 55)，总计2500个用户,日期为2015-3-1至2016-2-23，不均匀分布
    data_path = "../data/test_B.tsv"
    test_data = pd.read_csv(data_path, sep='\t') # error_bad_lines=False,
    test_data['LABEL'] = 2

    # 合并训练集和测试集
    data = pd.concat([train_data,test_data], axis =0)
    print(data.shape)
    data = data.reset_index(drop=True)

    # numeric_fea = ['FTR0', 'FTR1', 'FTR2', 'FTR3',
    #                'FTR4', 'FTR5', 'FTR6', 'FTR7',
    #                'FTR8', 'FTR9', 'FTR10', 'FTR11',
    #                'FTR12', 'FTR13', 'FTR14', 'FTR15',
    #                'FTR16', 'FTR17', 'FTR18', 'FTR19',
    #                'FTR20', 'FTR21', 'FTR22', 'FTR23',
    #                'FTR24', 'FTR25', 'FTR26', 'FTR27',
    #                'FTR28', 'FTR29', 'FTR30', 'FTR31', 'FTR32',
    #                'FTR33', 'FTR34', 'FTR35', 'FTR36', 'FTR37',
    #                'FTR38', 'FTR39', 'FTR40', 'FTR41', 'FTR42', 'FTR43',
    #                'FTR44', 'FTR45', 'FTR46', 'FTR47', 'FTR48', 'FTR49',
    #                'FTR50']
    # for num_fea in numeric_fea:
    #     minn = data[num_fea].quantile(0.01)
    #     maxx = data[num_fea].quantile(0.99)
    #     data.loc[data[num_fea] < minn, num_fea] = minn
    #     data.loc[data[num_fea] > maxx, num_fea] = maxx
    #     # log处理
    #     data[num_fea] = np.log(data[num_fea] + 1)

    data.to_csv('../data/all_data.csv', index=False)
else:
    train = pd.read_csv('../data/train.tsv', sep='\t', encoding='utf-8')


# 时间统计特征
# gen_access_num_train = gen_access_num(train)
#
# # 药物特征
# gen_ftr51_stat_train = gen_ftr51_stat(train)
# #
# # 时间差值特征
# user_next_time_stat_train = user_next_time_stat(train)
#
# # FTR统计值
# gen_ftr_stat_train = gen_ftr_stat(train)
#
# # FTR分类值
# gen_ftr_cat_train = gen_ftr_cat(train)
#
# # gen_ftr51_nunique
# gen_ftr51_nunique_train = gen_ftr_nunique(train)
#
# # ftr51_unique_rate
# ftr51_unique_rate_train = ftr51_unique_rate(train)
#
# gen_ftr51_len_train = gen_ftr51_len(train)

gen_mod_stat = gen_mod_stat(train)

# 无用
# gen_ftr51_day_stat = gen_ftr51_day_stat(train)

# gen_access_num_train.to_csv('../data/feature/train/gen_access_num.csv', index=False)
# gen_ftr51_stat_train.to_csv('../data/feature/train/gen_ftr51_stat.csv', index=False)
# user_next_time_stat_train.to_csv('../data/feature/train/user_next_time_stat.csv', index=False)
# gen_ftr_stat_train.to_csv('../data/feature/train/gen_ftr_stat.csv', index=False)
# gen_ftr_cat_train.to_csv('../data/feature/train/gen_ftr_cat.csv', index=False)
# gen_ftr51_nunique_train.to_csv('../data/feature/train/gen_ftr_nunique.csv', index=False)
# ftr51_unique_rate_train.to_csv('../data/feature/train/ftr51_unique_rate.csv', index=False)
# gen_ftr51_len_train.to_csv('../data/feature/train/gen_ftr51_len.csv', index=False)
gen_mod_stat.to_csv('../data/feature/train/gen_mod_stat.csv', index=False)

# gen_access_num_train.to_csv('../data/feature/test/gen_access_num.csv', index=False)
# gen_ftr51_stat_train.to_csv('../data/feature/test/gen_ftr51_stat.csv', index=False)
# user_next_time_stat_train.to_csv('../data/feature/test/user_next_time_stat.csv', index=False)
# gen_ftr_stat_train.to_csv('../data/feature/test/gen_ftr_stat.csv', index=False)
# gen_ftr_cat_train.to_csv('../data/feature/test/gen_ftr_cat.csv', index=False)
# gen_ftr51_nunique_train.to_csv('../data/feature/test/gen_ftr_nunique.csv', index=False)
# ftr51_unique_rate_train.to_csv('../data/feature/test/ftr51_unique_rate.csv', index=False)
# gen_ftr51_len_train.to_csv('../data/feature/test/gen_ftr51_len.csv', index=False)
# gen_mod_stat.to_csv('../data/feature/test/gen_mod_stat.csv', index=False)





# train_derive = pd.read_csv('../data/train_derive.csv')
# train_derive[:15000].to_csv('../data/feature/train/train_derive.csv', index=False)
# train_derive[15000:].to_csv('../data/feature/test/test_derive.csv', index=False)
# print(train_derive.shape)
# print(train_derive.columns.tolist())


# 预处理 sum
def transf_data(x):
    if x >= 2:
        return np.ceil(np.log(x*x))
    elif(x>=0 and x <2):
        return np.ceil(x)
    elif x < -2:
        return - np.ceil(np.log(x*x))
    elif (x >=-2 and x < 0):
        return - np.ceil(x)
    else:
        return 0