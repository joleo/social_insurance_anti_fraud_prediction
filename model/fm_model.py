# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     fm_model
   Description :
   Author :       Administrator
   date：          2018/8/5 0005
-------------------------------------------------
   Change Activity:
                   2018/8/5 0005:
-------------------------------------------------
"""
__author__ = 'Administrator'
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
import pylibfm
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
import  scipy.sparse
from sklearn.metrics import log_loss

X, y = make_classification(n_samples=1000,n_features=100, n_clusters_per_class=1)  # 1000个样本，100个特征，默认2分类

# 直接转化为稀疏矩阵，对有标称属性的数据集不能处理。
# X = scipy.sparse.csr_matrix(X)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# 由于大部分情况下，数据特征都是标称属性，所以需要先转化为字典，再转化稀疏矩阵。（转化为系数矩阵的过程中标称数据自动one-hot编码，数值属性保留）
data = [ {v: k for k, v in dict(zip(i, range(len(i)))).items()}  for i in X]  # 对每个样本转化为一个字典，key为特征索引（0-99），value为特征取值
X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.1, random_state=42)

v = DictVectorizer()
X_train = v.fit_transform(X_train)  # 转化为稀疏矩阵的形式，fm算法只能是被这种格式
X_test = v.transform(X_test)  # 转化为稀疏矩阵的形式，fm算法只能是被这种格式
# print(X_train.toarray())   # 打印二维矩阵形式



# 建模、训练、预测、评估
fm = pylibfm.FM(num_factors=50, num_iter=10, verbose=True, task="classification", initial_learning_rate=0.0001, learning_rate_schedule="optimal")
fm.fit(X_train,y_train)
y_pred_pro = fm.predict(X_test)  # 预测正样本概率
print("fm算法 验证集log损失: %.4f" % log_loss(y_test,y_pred_pro))


lr = LogisticRegression(verbose=True)
lr.fit(X_train,y_train)
y_pred_pro = lr.predict(X_test)  # 预测正样本概率
print("逻辑回归 验证集log损失: %.4f" % log_loss(y_test,y_pred_pro))
#