# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     gen_ididf_feature
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
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.externals import joblib

# common_path = r'~/Documents/Study/Python/merchants_bank'
common_path = r'/home/ad/Documents/merchants_bank'

# train flg
train_flg_path = common_path + r'/data/corpus/output/train_flg.csv'

# train tfidf
train_tfidf_path = common_path + r'/data/feature/4_train_tfidf.csv'

train_tfidf_stack_path = common_path + r'/data/feature/5_train_tfidf_stack.csv'

# test tfidf
test_tfidf_path = common_path + r'/data/feature/4_test_tfidf.csv'
test_tfidf_stack_path = common_path + r'/data/feature/5_test_tfidf_stack.csv'

# 堆叠模型的保存
lr_model_path = common_path + r'/data/model/LR_model.m'
svm_model_path = common_path + r'/data/model/svm_model.m'
randomf_model_path = common_path + r'/data/model/randomf_model.m'
bayes_model_path = common_path + r'/data/model/bayes_model.m'


def build_model_save(X, Y):
    print('x shape', X.shape)
    print('Y shape ', Y.shape)
    # 堆叠特征
    # 用逻辑回归堆叠
    print('logisticRegression begin ')
    model_lr = LogisticRegression()
    model_lr.fit(X, Y)
    joblib.dump(model_lr, lr_model_path)

    # 用svm堆叠
    print('svm begin ')
    model_svm = SVC(kernel='sigmoid', probability=True)
    model_svm.fit(X, Y)
    joblib.dump(model_svm, svm_model_path)

    # 用随机深林堆叠
    print('randomforest begin ')
    model_rf = RandomForestClassifier(n_estimators=100, max_features='auto', max_depth=6, n_jobs=-1)
    model_rf.fit(X, Y)
    joblib.dump(model_rf, randomf_model_path)

    # 高斯贝叶斯堆叠
    print('gaussian begin ')
    model_bayes = GaussianNB()
    model_bayes.fit(X, Y)
    joblib.dump(model_bayes, bayes_model_path)


def stacking_feat(tfidf_data):
    print('tfidf shape ', tfidf_data.shape)
    # 读取模型
    model_lr = joblib.load(lr_model_path)
    model_svm = joblib.load(svm_model_path)
    model_rf = joblib.load(randomf_model_path)
    model_bayes = joblib.load(bayes_model_path)

    y1_lr = model_lr.predict_proba(tfidf_data)
    y1_lr_df = pd.DataFrame(y1_lr, columns=['lr0', 'lr1'])

    y1_svm = model_svm.predict_proba(tfidf_data)
    y1_svm_df = pd.DataFrame(y1_svm, columns=['svm0', 'svm1'])

    y1_rf = model_rf.predict_proba(tfidf_data)
    y1_rf_df = pd.DataFrame(y1_rf, columns=['rf0', 'rf1'])

    y1_bayes = model_bayes.predict_proba(tfidf_data)
    y1_bayes_df = pd.DataFrame(y1_bayes, columns=['bayes0', 'bayes1'])

    return pd.concat([y1_lr_df, y1_svm_df, y1_rf_df, y1_bayes_df], axis=1)


if __name__ == '__main__':
    print('train')
    train_tfidf_data = pd.read_csv(train_tfidf_path)
    train_usrid = train_tfidf_data['USRID']
    # print(train_usrid)
    train_flg_data = pd.read_csv(train_flg_path)
    x_y = pd.merge(train_tfidf_data, train_flg_data, on='USRID', how='left')

    X = x_y.drop(['USRID', 'FLAG'], axis=1).values
    Y = x_y['FLAG'].values
    # 训练堆叠模型
    build_model_save(X, Y)
    # 获取堆叠后的特征
    train_stacking_tfidf_df = stacking_feat(X)
    print(train_stacking_tfidf_df.shape)
    train_stacking_tfidf_df.to_csv(train_tfidf_stack_path, index=0)

    print('test')
    test_tfidf_data = pd.read_csv(test_tfidf_path)
    test_usrid = test_tfidf_data['USRID']
    test_tfidf_data.pop('USRID')
    test_stacking_tfidf_df = stacking_feat(test_tfidf_data)

    test_stacking_tfidf_df.to_csv(test_tfidf_stack_path, index=0)

    train_usrid = pd.DataFrame(train_usrid, columns=['USRID'])
    train_stacking_tfidf_df = pd.concat([train_stacking_tfidf_df, train_usrid], axis=1)
    train_stacking_tfidf_df.to_csv(train_tfidf_stack_path, index=0)

    test_usrid = pd.DataFrame(test_usrid, columns=['USRID'])
    test_stacking_tfidf_df = pd.concat([test_stacking_tfidf_df, test_usrid], axis=1)
    test_stacking_tfidf_df.to_csv(test_tfidf_stack_path, index=0)



