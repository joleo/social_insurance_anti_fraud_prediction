# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     lr_model
   Description :
   Author :       Administrator
   date：          2018/7/14 0014
-------------------------------------------------
   Change Activity:
                   2018/7/14 0014:
-------------------------------------------------
"""
__author__ = 'Administrator'
import pandas as pd, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import metrics,cross_validation
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

import time
t1=time.time()
path = '../data/'
# train = pd.read_csv(path + 'train.csv')
# test = pd.read_csv(path + 'test.csv')

train_path = '../data/feature/train/'
test_path = '../data/feature/test/'
train = pd.read_csv(train_path + 'train_derive.csv')
test = pd.read_csv(test_path + 'test_derive.csv')
test = test.fillna(0)

train_id = pd.read_csv('../data/train_id.tsv', sep='\t')
train = train.merge(train_id, on='PERSONID', how='left')
train = train.fillna(0)
train_y = train.pop('LABEL')
test_userid = test.pop('PERSONID')
train_userid = train.pop('PERSONID')

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=2017)
clf = LogisticRegression(C=6, dual=True)
n = 0
cvxx=[]
for index_train, index_eval in cv.split(train,train_y):
    x_train, x_eval = train[index_train], train[index_eval]
    y_train, y_eval = train_y[index_train], train_y[index_eval]

    print(x_train.shape)
    probas = clf.fit(x_train, train_y)
    testpreds=clf.predict_proba(x_eval.values)[:, 1]

    # auc = roc_auc_score(y_eval, testpreds)
    cvxx.append(roc_auc_score(y_eval, testpreds))
    # print(auc)
    preds=clf.predict_proba(test.values)[:, 1]
    if n > 0:
        totalpreds = totalpreds + preds
    else:
        totalpreds = preds
    # gbm.save_model('lgb_model_fold_{}.txt'.format(n), num_iteration=gbm.best_iteration)
    n += 1

totalpreds = totalpreds / n
print('lgb best score', np.mean(cvxx))

#保存概率文件
res = pd.DataFrame()
res['PERSONID'] = list(test_userid.values)
res['Pre'] = preds
res.to_csv('../data/submit/prob_lr_baseline.csv',index=None, sep='\t')