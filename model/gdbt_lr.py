# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     gdbt_lr
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
from sklearn import ensemble
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve

path = '../data/'
values = ['LABEL']
for i in range(51):
    # if i not in [1,3,6,11,19,24,26,46]:
    values.append('FTR' + str(i))
test_value = []
for i in range(51):
    # if i not in [1,3,6,11,19,24,26,46]:
    test_value.append('FTR' + str(i))

# 弱分类器的数目
n_estimator = 10
train_id = pd.read_csv('../data/train_id.csv', sep='\t')
train_set = pd.read_csv('../data/train.csv', sep='\t')
train = train_id.merge(train_set, on='PERSONID', how='left')[values]
test =  pd.read_csv(path + 'test.csv', sep='\t')[test_value]
test['LABEL'] = -1

X = train.values
y = train['LABEL'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
X_train, X_train_lr, y_train, y_train_lr = train_test_split(X_train, y_train, test_size=0.3)

# 调用GBDT分类模型。
grd = GradientBoostingClassifier(n_estimators=n_estimator)
# 调用one-hot编码。
grd_enc = OneHotEncoder()
# 调用LR分类模型。
grd_lm = LogisticRegression()

'''使用X_train训练GBDT模型，后面用此模型构造特征'''
grd.fit(X_train, y_train)

# fit one-hot编码器
grd_enc.fit(grd.apply(X_train)[:, :, 0])

''' 
使用训练好的GBDT模型构建特征，然后将特征经过one-hot编码作为新的特征输入到LR模型训练。
'''
grd_lm.fit(grd_enc.transform(grd.apply(X_train_lr)[:, :, 0]), y_train_lr)
# 用训练好的LR模型多X_test做预测
y_pred_grd_lm = grd_lm.predict_proba(grd_enc.transform(grd.apply(X_test)[:, :, 0]))[:, 1]
# 根据预测结果输出
fpr_grd_lm, tpr_grd_lm, _ = roc_curve(y_test, y_pred_grd_lm)


# fit one-hot编码器
test_oh = grd_enc.transform(grd.apply(test)[:, :, 0])
y_pred = grd_lm.predict_proba(test_oh)[:, 1]
