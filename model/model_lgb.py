# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     model_lgb
   Description :
   Author :       Administrator
   date：          2018/7/23 0023
-------------------------------------------------
   Change Activity:
                   2018/7/23 0023:
-------------------------------------------------
"""
__author__ = 'Administrator'
import time
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
# from feature_integrate.feature_integrate import *
from feature_integrate.feature_integrate2 import *
# from feature_integrate.feature_integrate3 import *

seed=1024
np.random.seed(seed)
time_date = time.strftime('%Y-%m-%d', time.localtime(time.time()))


# train = train()
# test = test()
train = pd.read_csv('../data/train_feature.csv')
test = pd.read_csv('../data/test_feature.csv')

# 0.927
# feature_name = ['PERSONID','ftr_skew18', 'ftr_mean0', 'ftr_std10', 'ftr_max33', 'ev_day_avg31', 'ftr_mean16', 'd_count1', 'ftr_skew10', 'ftr_nunique51', 'ftr51_std', 'm_count9', 'm_avg_n10', 'ftr_rate28', 'ftr_std12', 'ftr_std48', 'ev_day_avg18', 'ftr_mean42', 'ftr_skew8', 'ftr_mean21', 'ftr_std34', 'ftr_sum5', 'ftr_mean36', 'diff_day_max', 'ftr_skew12', 'ftr_std16', 'ftr_sum42', 'ev_day_avg23', 'm_avg_n1', 'ftr_skew17', 'm_ev_mon_num1', 'ftr_skew36', 'ftr_rate14', 'ftr_mean48', 'ftr_rate8', 'ftr_sum35', 'ftr_mean35', 'ftr_sum28', 'ftr_mean18', 'ftr_std44', 'ftr_skew21', 'ev_day_avg4', 'ftr_std23', 'ftr_skew42', 'ftr_rate51', 'ftr_max43', 'ftr_rate7', 'ev_day_avg1', 'ftr_skew28', 'ftr_nunique28', 'ev_day_avg2', 'ftr_mean23', 'ftr_rate9', 'ftr_mean32', 'm_avg_n12', 'ev_day_avg13', 'ftr_skew41', 'ftr_rate21', 'd_count5', 'ftr_max17', 'ftr_std42', 'ftr_rate12', 'ftr_skew43', 'ftr_sum8', 'ev_day_avg3', 'ftr_max5', 'm_avg_n11', 'ftr_rate10', 'ftr_max36', 'ftr_sum36', 'ftr_nunique0', 'ftr_max9', 'ftr_skew32', 'ftr_sum30', 'ftr_skew48', 'ftr_rate0', 'diff_day_std', 'ftr_skew35', 'm_avg_n9', 'm_ev_mon_num2', 'm_ev_mon_num10', 'm_count2', 'ftr51_mean', 'ftr_max30', 'ftr_mean44', 'ev_day_avg8', 'ftr_skew9', 'ftr_rate5', 'ftr_skew30', 'ftr_sum9', 'm_ev_mon_num9', 'ftr_skew34', 'm_ev_mon_num11', 'ftr_sum43', 'ftr_skew16', 'ftr_mean34', 'ftr_sum41', 'ftr_skew5', 'diff_day_mean', 'ftr_sum33', 'ftr51_sum']
# 100之后，0.9282027178724299
# feature_name = ['PERSONID','ftr_max10', 'ev_day_avg21', 'ftr_mean29', 'ftr_sum47', 'd_count2', 'ftr_sum48', 'ftr_std47', 'ftr_std30', 'ftr_nunique47', 'ftr_nunique5', 'ftr_sum34', 'ftr_max44', 'm_count10', 'ftr_rate47', 'ftr_std9', 'ftr_sum17', 'ftr_std0', 'ftr_nunique0', 'ftr_mean17', 'ftr_std36', 'm_ev_mon_num4', 'ev_day_avg9', 'm_avg_n7', 'ftr_std5', 'ftr_mean16', 'ftr_mean47', 'diff_day_min', 'ftr_max21', 'm_ev_mon_num3', 'ftr_skew10', 'ftr_mean28', 'ftr_rate16', 'ftr_nunique16', 'ftr_std33', 'ftr_sum16', 'ftr51_max', 'ftr_mean30', 'ftr_std21', 'ftr_sum12', 'd_count23', 'ftr_std18', 'm_ev_mon_num7', 'ftr_std29', 'ftr_mean10', 'ftr_mean5', 'ev_day_avg6', 'ftr_mean21', 'ftr_sum0', 'ev_day_avg19', 'ftr_max28', 'ftr_nunique9', 'ev_day_avg15', 'ftr_max41', 'ftr_skew8', 'ftr_sum5', 'ftr_std41', 'ftr_max23', 'ftr_skew28', 'ftr_std32', 'ftr_max29', 'ftr_std17', 'ftr_mean33', 'ftr_skew18', 'ev_day_avg10', 'ev_day_avg16', 'ev_day_avg14', 'ftr_sum29', 'ftr_mean41', 'ftr_skew23', 'ftr_skew0', 'm_count12', 'ftr_std23', 'd_count3', 'ftr_rate29', 'm_unique4', 'ftr_std35', 'ev_day_avg7', 'ftr_rate21', 'm_count9', 'ftr_mean36', 'ftr_sum44', 'ftr_skew33', 'd_count1', 'ftr_sum10', 'ftr_max42', 'ftr_std8', 'm_count1', 'm_avg_n3', 'ftr_skew44', 'ftr_std16', 'ftr_max12', 'ev_day_avg18', 'ftr_std12', 'ftr_mean42', 'ftr_mean0', 'ftr_max8', 'ftr_std43', 'ftr_max0', 'ftr_mean43', 'ev_day_avg23', 'm_avg_n1', 'ftr_mean23', 'ftr_mean12', 'ftr_sum14', 'ftr_std10', 'm_unique12', 'ftr_sum35', 'm_unique9', 'm_ev_mon_num1', 'ftr_std34', 'ftr_max18', 'ftr_mean9', 'ftr_max5', 'ftr_max33', 'ftr51_std', 'ftr_rate28', 'ftr_std42', 'ev_day_avg12', 'ftr_mean14', 'ftr_sum42', 'ftr_max32', 'ftr_rate10', 'ftr_std44', 'ev_day_avg5', 'ftr_rate7', 'ftr_sum28', 'ftr_nunique51', 'ftr_nunique28', 'ftr_mean18', 'm_count2', 'ev_day_avg1', 'm_avg_n10', 'm_avg_n11', 'ftr_skew21', 'ev_day_avg31', 'm_avg_n2', 'ftr_skew41', 'ev_day_avg11', 'm_ev_mon_num12', 'ftr_skew29', 'm_unique10', 'ftr_rate12', 'ftr_mean8', 'm_avg_n6', 'ftr_rate8', 'diff_day_sum', 'ftr_rate0', 'ftr_skew36', 'ftr_mean35', 'ftr_skew17', 'ftr_max36', 'ev_day_avg2', 'm_unique11', 'ftr_skew12', 'ftr_mean48', 'ftr_mean32', 'ftr_mean34', 'ftr_sum8', 'ftr_std48', 'ftr_skew42', 'ftr_skew35', 'ftr_rate9', 'ftr_rate5', 'ev_day_avg4', 'ftr_skew48', 'ftr_skew32', 'm_avg_n9', 'ftr_rate51', 'ftr_skew47', 'ftr_max9', 'ftr_sum36', 'ev_day_avg13', 'ftr_max43', 'ftr_mean44', 'ftr_max30', 'ftr_skew43', 'ftr_skew9', 'd_count5', 'ftr_sum30', 'm_unique2', 'ftr_rate14', 'ftr_max17', 'ftr_sum9', 'ftr_skew16', 'ftr_sum43', 'diff_day_std', 'm_avg_n12', 'ev_day_avg3', 'diff_day_max', 'ftr_sum41', 'ev_day_avg8', 'ftr_skew30', 'm_ev_mon_num11', 'm_ev_mon_num2', 'm_ev_mon_num10', 'ftr51_mean', 'ftr_skew34', 'ftr_skew5', 'diff_day_mean', 'm_ev_mon_num9', 'ftr_sum33', 'ftr51_sum']
# 0.932
# feature_name = ['PERSONID','ftr_sum14', 'ftr_std10', 'm_unique12', 'ftr_sum35', 'm_unique9', 'm_ev_mon_num1', 'ftr_std34', 'ftr_max18', 'ftr_mean9', 'ftr_max5', 'ftr_max33', 'ftr51_std', 'ftr_rate28', 'ftr_std42', 'ev_day_avg12', 'ftr_mean14', 'ftr_sum42', 'ftr_max32', 'ftr_rate10', 'ftr_std44', 'ev_day_avg5', 'ftr_rate7', 'ftr_sum28', 'ftr_nunique51', 'ftr_nunique28', 'ftr_mean18', 'm_count2', 'ev_day_avg1', 'm_avg_n10', 'm_avg_n11', 'ftr_skew21', 'ev_day_avg31', 'm_avg_n2', 'ftr_skew41', 'ev_day_avg11', 'm_ev_mon_num12', 'ftr_skew29', 'm_unique10', 'ftr_rate12', 'ftr_mean8', 'm_avg_n6', 'ftr_rate8', 'diff_day_sum', 'ftr_rate0', 'ftr_skew36', 'ftr_mean35', 'ftr_skew17', 'ftr_max36', 'ev_day_avg2', 'm_unique11', 'ftr_skew12', 'ftr_mean48', 'ftr_mean32', 'ftr_mean34', 'ftr_sum8', 'ftr_std48', 'ftr_skew42', 'ftr_skew35', 'ftr_rate9', 'ftr_rate5', 'ev_day_avg4', 'ftr_skew48', 'ftr_skew32', 'm_avg_n9', 'ftr_rate51', 'ftr_skew47', 'ftr_max9', 'ftr_sum36', 'ev_day_avg13', 'ftr_max43', 'ftr_mean44', 'ftr_max30', 'ftr_skew43', 'ftr_skew9', 'd_count5', 'ftr_sum30', 'm_unique2', 'ftr_rate14', 'ftr_max17', 'ftr_sum9', 'ftr_skew16', 'ftr_sum43', 'diff_day_std', 'm_avg_n12', 'ev_day_avg3', 'diff_day_max', 'ftr_sum41', 'ev_day_avg8', 'ftr_skew30', 'm_ev_mon_num11', 'm_ev_mon_num2', 'm_ev_mon_num10', 'ftr51_mean', 'ftr_skew34', 'ftr_skew5', 'diff_day_mean', 'm_ev_mon_num9', 'ftr_sum33', 'ftr51_sum']

##########################################
# feature_name = ['PERSONID','ftr51_sum', 'ftr_sum33', 'm_ev_mon_num9', 'diff_day_mean', 'm_ev_mon_num11', 'ftr_skew5', 'ftr_skew34', 'm_ev_mon_num2', 'ftr51_mean', 'm_ev_mon_num10', 'ftr_sum41', 'ftr_skew30', 'm_count1_c1', 'd_count5', 'ftr_skew16', 'ev_day_avg8', 'ftr_sum9', 'ftr_max17', 'm_unique2', 'ftr_skew9', 'ftr_mean34', 'ftr_sum36', 'ftr_rate9', 'diff_day_std', 'ftr_max43', 'ev_day_avg3', 'ftr_sum30', 'diff_day_max', 'ftr_skew47', 'ev_day_avg31', 'ftr_mean32', 'ftr_mean44', 'ftr_skew43', 'ftr_rate5', 'ftr_sum43', 'ftr_skew48', 'ftr_rate8', 'ftr_skew32', 'ev_day_avg4', 'm_avg_n12', 'ftr_skew41', 'ftr_max9', 'ftr_rate14', 'ftr_rate10', 'ftr_skew35', 'ftr_mean48', 'ftr_max30', 'm_avg_n9', 'ev_day_avg13', 'ftr_mean35', 'ftr_rate0', 'ev_day_avg1', 'm_avg_n11', 'm_avg_n2', 'ftr_rate7', 'm_unique10', 'ftr_skew17', 'ftr_rate21', 'm_ev_mon_num12', 'ftr_mean18', 'ftr_std42', 'ev_day_avg2', 'ftr_rate51', 'm_avg_n10', 'ftr_skew42', 'ftr_mean9', 'ev_day_avg18', 'ftr_std12', 'ftr_std35', 'ftr_max0', 'ev_day_avg23', 'm_unique11', 'ftr_skew18', 'ftr_std48', 'ftr_max5', 'ftr_sum8', 'ftr_max36', 'ftr51_std', 'm_avg_n3', 'ftr_skew12', 'ftr_mean0', 'm_unique1', 'ftr_rate28', 'ftr_sum28', 'ftr_mean8', 'ftr_skew21', 'ftr_rate29', 'ftr_std43', 'ev_day_avg11', 'ftr_nunique28', 'ftr_max42', 'ftr_skew36', 'm_avg_n6', 'ftr_nunique0', 'm_avg_n1', 'ftr_sum12', 'ev_day_avg12', 'ftr_mean42', 'ftr_rate12', 'ftr_skew28']

feature_name = ['ftr51_min', 'ftr_max35', 'ftr_max34','ftr_std7']
features = [x for x in train.columns if x not in feature_name]
train = train[features]
test = test[features]

train_id = pd.read_csv('../data/train_id.csv', sep='\t')
train = train.merge(train_id, on='PERSONID', how='left')
train_y = train.pop('LABEL')

print('the number of feature: ' + str(train.shape))
print('the number of feature: ' + str(test.shape))

train_userid = train.pop('PERSONID')
col = train.columns
train_x = train[col].values
test_userid = test.pop('PERSONID')

##########################################################交叉预测，实际上是stacking第一层做的操作
# 利用不同折数加参数，特征，样本（随机数种子）扰动，再加权平均得到最终成绩
model = lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=29, max_depth=-1, learning_rate=0.1, n_estimators=10000,
                           max_bin=425, subsample_for_bin=50000, objective='binary', min_split_gain=0,
                           min_child_weight=5, min_child_samples=10, subsample=1, subsample_freq=1,
                           colsample_bytree=1, reg_alpha=3, reg_lambda=5, seed=1000, nthread=-1, silent=True)

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2018)
n = 0
cv = []
for index_train, index_eval in kf.split(train,train_y):

    x_train, x_eval = train.iloc[index_train], train.iloc[index_eval]
    y_train, y_eval = train_y[index_train], train_y[index_eval]

    model.fit(x_train, y_train, eval_metric='auc',
              eval_set=[(x_train, y_train), (x_eval, y_eval)],
              early_stopping_rounds=100)
    #cv.append(roc_auc_score(y_eval, y_pred))

    preds = model.predict_proba(test.values, num_iteration=-1)[:, 1]
    if n > 0:
        totalpreds = totalpreds + preds
    else:
        totalpreds = preds
    # gbm.save_model('lgb_model_fold_{}.txt'.format(n), num_iteration=gbm.best_iteration)
    n += 1

totalpreds = totalpreds / n

#print('lgb best score', np.mean(cv))