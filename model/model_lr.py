# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     model_lr
   Description :
   Author :       Administrator
   date：          2018/7/17 0017
-------------------------------------------------
   Change Activity:
                   2018/7/17 0017:
-------------------------------------------------
"""
__author__ = 'Administrator'
import time
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from feature_integrate.feature_integrate2 import *
from sklearn.linear_model import LogisticRegression

seed=1024
np.random.seed(seed)
time_date = time.strftime('%Y-%m-%d', time.localtime(time.time()))


train = train()
test = test()
train_id = pd.read_csv('../data/train_id.csv', sep='\t')
train = train.merge(train_id, on='PERSONID', how='left')
train_y = train.pop('LABEL')

print('the number of feature: ' + str(train.shape))
print('the number of feature: ' + str(test.shape))

train_userid = train.pop('PERSONID')
col = train.columns
train_x = train[col].values
test_userid = test.pop('PERSONID')

# cv = StratifiedKFold(train_y, n_folds=5)
# cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=2017)
from sklearn.cross_validation import KFold
kf = KFold(n=train.shape[0], n_folds=5, shuffle=True, random_state=2017)

clf = LogisticRegression(C=6, dual=True)
for train_index,test_index in kf:
    print(len(train_index), test_index)
    probas = clf.fit(train_x[train_index], train_y[train_index])
    auc = roc_auc_score(train_y[test_index], probas[:, 1])
    print(auc)

    # clf.fit(trn_term_doc, train_y)
    preds=clf.predict_proba(test)[:, 1]
    #保存概率文件
    res = pd.DataFrame()
    res['PERSONID'] = list(test_userid.values)
    res['Pre'] = preds
    res.to_csv('../data/submit/prob_lr_baseline.csv',index=None, sep='\t')