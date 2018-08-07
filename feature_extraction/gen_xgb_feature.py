# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     gen_xgb_feature
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
from sklearn.model_selection import train_test_split
from sklearn import metrics
from  sklearn.datasets  import  make_hastie_10_2
from xgboost.sklearn import XGBClassifier

path = '../data/'
values = ['LABEL']
for i in range(51):
        values.append('f' + str(i))
test_value = []
for i in range(51):
    test_value.append('f' + str(i))

train_id = pd.read_csv('../data/train_id.csv', sep='\t')
train_set = pd.read_csv('../data/train.csv', sep='\t')
train = train_id.merge(train_set, on='PERSONID', how='left')[values]
test =  pd.read_csv(path + 'test.csv', sep='\t')[test_value]
# test['LABEL'] = -1

X = train.values
y = train['LABEL'].values

##载入示例数据 10维度
# X, y = make_hastie_10_2(random_state=0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)##test_size测试集合所占比例
##X_train_1用于生成模型  X_train_2用于和新特征组成新训练集合
X_train_1, X_train_2, y_train_1, y_train_2 = train_test_split(X_train, y_train, test_size=0.6, random_state=0)

##合并维度
import numpy as np
def mergeToOne(X,X2):
    X3=[]
    for i in range(X.shape[0]):
        tmp=np.array([list(X[i]),list(X2[i])])
        X3.append(list(np.hstack(tmp)))
    X3=np.array(X3)
    return X3

clf = XGBClassifier(
 learning_rate =0.3, #默认0.3
 n_estimators=30, #树的个数
 max_depth=3,
 min_child_weight=1,
 gamma=0.5,
 subsample=0.6,
 colsample_bytree=0.6,
 objective= 'binary:logistic', #逻辑回归损失函数
 nthread=4,  #cpu线程数
 scale_pos_weight=1,
 reg_alpha=1e-05,
 reg_lambda=1,
 seed=27)  #随机种子

clf.fit(X_train_1, y_train_1)
new_feature= clf.apply(X_train_2) # X_train_2用于和新特征组成新训练集合

# 新特征
X_train_new2=mergeToOne(X_train_2,new_feature)
new_feature_test= clf.apply(X_test)
X_test_new=mergeToOne(X_test,new_feature_test)


# model = XGBClassifier(
#  learning_rate =0.1,
#  n_estimators=1000,
#  max_depth=3,
#  min_child_weight=1,
#  gamma=0.5,
#  subsample=0.6,
#  colsample_bytree=0.6,
#  objective= 'binary:logistic',
#  nthread=4,
#  scale_pos_weight=1,
#  reg_alpha=1e-05,
#  reg_lambda=1,
#  seed=27)
#
# model.fit(X_train_new2, y_train_2)
# y_pre= model.predict(X_test_new)
# y_pro= model.predict_proba(X_test_new)[:,1]
# print("AUC Score : %f" % metrics.roc_auc_score(y_test, y_pro) )
# print("Accuracy : %.4g" % metrics.accuracy_score(y_test, y_pre) )

new_test_feature= clf.apply(test.values) # X_train_2用于和新特征组成新训练集合
# 新特征
test_new2=mergeToOne(test.values,new_test_feature)

train_pred = pd.DataFrame(X_train_new2)
train_pred.to_csv('../data/X_train_new2.csv', index=False)

test_pred = pd.DataFrame(test_new2)
test_pred.to_csv('../data/test_new2.csv', index=False)