# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     eda
   Description :
   Author :       Administrator
   date：          2018/7/21 0021
-------------------------------------------------
   Change Activity:
                   2018/7/21 0021:
-------------------------------------------------
"""
__author__ = 'Administrator'
import pandas as pd
train_path = '../data/feature/train/'
test_path = '../data/feature/test/'
from feature_integrate.feature_integrate2 import *


# train = pd.read_csv('../data/test_feature.csv')
# train = pd.read_csv('../data/test_feature.csv')
# import seaborn as sns
# import matplotlib.pyplot as plt
#
# dfData = train.corr()
# plt.subplots(figsize=(200, 200))  # 设置画面大小
# sns.heatmap(dfData, annot=True, vmax=1, square=True, cmap="Blues")
# plt.savefig('./test_hot.png')
# plt.show()

# fig, ax = plt.subplots(figsize=(30, 30))
# # 二维的数组的热力图，横轴和数轴的ticklabels要加上去的话，既可以通过将array转换成有column
# # 和index的DataFrame直接绘图生成，也可以后续再加上去。后面加上去的话，更灵活，包括可设置labels大小方向等。
# sns.heatmap(pd.DataFrame(np.round(a, 2), columns=['a', 'b', 'c'], index=range(1, 5)),
#             annot=True, vmax=1, vmin=0, xticklabels=True, yticklabels=True, square=True, cmap="YlGnBu")
# # sns.heatmap(np.round(a,2), annot=True, vmax=1,vmin = 0, xticklabels= True, yticklabels= True,
# #            square=True, cmap="YlGnBu")
# ax.set_title('二维数组热力图', fontsize=18)
# ax.set_ylabel('数字', fontsize=18)
# ax.set_xlabel('字母', fontsize=18)  # 横变成y轴，跟矩阵原始的布局情况是一样的
# train = train()
train = test()
# 处理冗余特征
ftr_sum_col = ['ftr_sum47','ftr_sum0','ftr_sum17','ftr_sum18','ftr_sum23','ftr_sum32','ftr_sum34','ftr_sum35','ftr_sum42','ftr_sum44','ftr_sum48']
train['ftr_sum_c1'] = train['ftr_sum47']+train['ftr_sum0']+train['ftr_sum17']+train['ftr_sum18']+train['ftr_sum23']+train['ftr_sum32']+train['ftr_sum34']+train['ftr_sum35']+train['ftr_sum42']+train['ftr_sum44']+train['ftr_sum48']
ftr_count_col = ['ftr_count0', 'ftr_count5', 'ftr_count7', 'ftr_count8', 'ftr_count9', 'ftr_count10', 'ftr_count12', 'ftr_count14' ,'ftr_count16', 'ftr_count21', 'ftr_count28', 'ftr_count29', 'ftr_count47', 'ftr_count51']
train['ftr_count_c1'] = train['ftr_count0']+ train['ftr_count5']+ train['ftr_count7']+ train['ftr_count8']+ train['ftr_count9']+ train['ftr_count10']+ train['ftr_count12']+ train['ftr_count14']+ train['ftr_count16']+ train['ftr_count21']+ train['ftr_count28']+ train['ftr_count29']+ train['ftr_count47']+ train['ftr_count51']
m_count_col = ['m_count1', 'm_count2']
train['m_count1_c1'] = train['m_count1'] + train['m_count2']
m_count_col2 = ['m_count9',  'm_count10',  'm_count11', 'm_count12']
FTR_col = ['FTR5','FTR8','FTR10','FTR21','FTR29','FTR0','FTR17','FTR18','FTR23','FTR32','FTR35','FTR44','FTR48']

train['m_count1_c2'] = train['m_count9'] + train['m_count10'] + train['m_count11'] + train['m_count12']
drop_col = ftr_sum_col + ftr_count_col + m_count_col + m_count_col2# + FTR_col
train.drop(drop_col, axis=1, inplace=True)
# train.to_csv('../data/train_feature.csv', index=False)
train.to_csv('../data/test_feature.csv', index=False)

print(train.shape)
print(train.columns.tolist())


