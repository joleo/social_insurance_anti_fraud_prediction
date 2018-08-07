# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     gen_tfidf_feature
   Description :
   Author :       Administrator
   date：          2018/7/14 0014
-------------------------------------------------
   Change Activity:
                   2018/7/14 0014:
-------------------------------------------------
"""
__author__ = 'Administrator'
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

# class TfIdfFeature(object):
# 根据特征个数生成特征列名,作为输出csv文件的头
# 输入：size，输出：size个特征名
def get_featrue_name(size):
    print('正在生成特征列名')
    names = []
    for i in range(size):
        names.append('tfidf_' + str(i))
    return names

def handle_ftr(data):
    # 合并列
    evt_tfidf_df = data.groupby('PERSONID', as_index=False)['FTR51'].agg({'TFIDF': lambda x: ' '.join(x)})
    return evt_tfidf_df

def tfidf_handle_data(data):
    print('tfidf特征化evt')
    tfv = TfidfVectorizer(min_df=3, max_df=0.95, use_idf=1, smooth_idf=1, sublinear_tf=1, stop_words='english')
    # tfv = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.9, use_idf=1, smooth_idf=1, sublinear_tf=1)
    tfv.fit(data)
    result = tfv.transform(data)
    return result

if __name__ == '__main__':
    print('train')
    path = '../data/'
    train_set = pd.read_csv('../data/train2.csv', sep='\t')
    train_id  = pd.read_csv('../data/train_id.csv', sep='\t')
    train = train_id.merge(train_set, on='PERSONID', how='left')
    # 获取训练集的数据
    train_ftr51_list_data = handle_ftr(train)
    # 获取测试集的数据
    print('test')
    test = pd.read_csv(path + 'test2.csv', sep='\t')
    test_ftr51_list_data = handle_ftr(test)
    train_tfidf_path = '../data/'
    test_tfidf_path = '../data/'
    train_ftr51_list_data.to_csv(train_tfidf_path + 'tfidf_train_data_df.csv' , index=0)
    # test_ftr51_list_data.to_csv(test_tfidf_path + 'tfidf_test_data_df.csv', index=0)

    # # 记录训练个数
    # train_n = train_ftr51_list_data.shape[0]
    #
    # # 合并训练集和测试集
    # ftr51_list_data = pd.concat([train_ftr51_list_data,test_ftr51_list_data],axis=0, ignore_index=True)
    #
    # # 通过tfidf对evt特征化
    # tfidf_data = tfidf_handle_data(ftr51_list_data['TFIDF'])
    # # 将结果转为dataframe
    # tfidf_columns = get_featrue_name(tfidf_data.shape[1])
    # tfidf_data_df = pd.DataFrame(tfidf_data.todense(), columns=tfidf_columns)
    # # 添加USRID
    # usrid_df = pd.DataFrame(ftr51_list_data['PERSONID'],columns=['PERSONID'])
    # # 重置index否则合并时会报错
    # tfidf_data_df = tfidf_data_df.reset_index(drop=True)
    # usrid_df = usrid_df.reset_index(drop=True)
    # # 合并USRID和tfidf
    # tfidf_data_df = pd.concat([usrid_df, tfidf_data_df], axis=1)
    # # 保存文件
    # print('保存文件')
    # train_tfidf_path = '../data/feature/train/'
    # test_tfidf_path = '../data/feature/test/'
    # tfidf_data_df[:train_n].to_csv(train_tfidf_path + 'tfidf_data_df.csv' , index=0)
    # tfidf_data_df[train_n:].to_csv(test_tfidf_path + 'tfidf_data_df.csv', index=0)
    # print('结束')