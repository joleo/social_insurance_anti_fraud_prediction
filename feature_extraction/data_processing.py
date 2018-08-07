# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     data_processing
   Description :
   Author :       Administrator
   date：          2018/7/13 0013
-------------------------------------------------
   Change Activity:
                   2018/7/13 0013:
-------------------------------------------------
"""
__author__ = 'Administrator'
import nltk
# nltk.download('punkt')
import jieba.analyse
import pandas as pd
from nltk.tokenize import word_tokenize

def pre_process_cn(courses, low_freq_filter=True):
    """
     简化的 中文+英文 预处理
        1.去掉停用词
        2.去掉标点符号
        3.处理为词干
        4.去掉低频词
    """


    texts_tokenized = []
    for document in courses:  #########document是courses中的每个元素
        texts_tokenized_tmp = []
        for word in word_tokenize(document):  ############word为document中每个单词
            texts_tokenized_tmp += jieba.analyse.extract_tags(word,
                                                              10)  #########texts_tokenized_tmp为list，里的元素为每个document打散成为
        texts_tokenized.append(texts_tokenized_tmp)  ##########texts_tokenized为列表的列表

    texts_filtered_stopwords = texts_tokenized

    # 去除标点符号
    english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%']
    texts_filtered = [[word for word in document if not word in english_punctuations] for document in
                      texts_filtered_stopwords]  ######去除texts_tokenized中的标点符号

    # # 词干化
    # from nltk.stem.lancaster import LancasterStemmer
    # st = LancasterStemmer()
    # texts_stemmed = [[st.stem(word) for word in docment] for docment in
    #                  texts_filtered]  #####将texts_tokenized每个单词更改为其词根形式

    # ##########去除过低频词
    # if low_freq_filter:
    #     all_stems = sum(texts_stemmed, [])  #######texts_stemmed中词根组成的list，可以有重复...实验了一下sum函数，把二维list改为一维list，具体可以实验
    #     stems_once = set(
    #         stem for stem in set(all_stems) if all_stems.count(stem) == 1)  #######texts_stemmed中所有不重复的元素组成stems_once集合
    #     texts = [[stem for stem in text if stem not in stems_once] for text in
    #              texts_stemmed]  #######将stems_once之外的其他stem改为初始文章形式，即若stems_once中的stem属于某篇文章，则将该stem放到该文章所在的list
    # else:
    #     texts = texts_stemmed
    return texts_filtered


def removeSpecificAndPutMedian(data, first=98, second=96):
    # 异常值插入均值
    New = []
    med = data.median()
    for val in data:
        if ((val == first) | (val == second)):
            New.append(med)
        else:
            New.append(val)

    return New


def del_sign(courses):
    texts_tokenized = []
    for document in courses:  #########document是courses中的每个元素
        texts_tokenized_tmp = []
        for word in word_tokenize(document):  ############word为document中每个单词
            texts_tokenized_tmp += jieba.analyse.extract_tags(word,
                                                              10)  #########texts_tokenized_tmp为list，里的元素为每个document打散成为
        texts_tokenized.append(texts_tokenized_tmp)  ##########texts_tokenized为列表的列表

    texts_filtered_stopwords = texts_tokenized

    # 去除标点符号
    english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%']
    texts_filtered = [[word for word in document if not word in english_punctuations] for document in
                      texts_filtered_stopwords]  ######去除texts_tokenized中的标点符号

def get_featrue_name(size):
    print('正在生成特征列名')
    names = []
    for i in range(size):
        names.append('word_' + str(i))
    return names


def process_set(df):
    train_set = pd.read_csv('../data/train.csv', sep='\t')
    train_id = pd.read_csv('../data/train_id.csv', sep='\t')
    train = train_id.merge(train_set, on='PERSONID', how='left')
    test = pd.read_csv('../data/test.csv', sep='\t')

    sort_test = test.sort_values(by=['PERSONID', 'CREATETIME'])
    sort_train = train.sort_values(by=['PERSONID', 'CREATETIME'])

    sort_test.to_csv('../data/test.csv', index=False)
    sort_train.to_csv('../data/train.csv', index=False)


if __name__ == '__main__':
    path = '../data/'
    # train = pd.read_csv(path + 'ftr51_train_word.csv')
    # test = pd.read_csv(path + 'ftr51_test_word.csv')

    # texts = pre_process_cn(list(train.TFIDF))
    # train_texts = pd.DataFrame(texts)
    # featrue_name = get_featrue_name(train_texts.shape[1])
    # train_texts.columns = featrue_name
    # train_texts.to_csv('../data/train_texts.csv', index=False)

    # texts = pre_process_cn(list(test.TFIDF))
    # test_texts = pd.DataFrame(texts)
    # get_featrue_name = get_featrue_name(test_texts.shape[1])
    # test_texts.columns = get_featrue_name
    # test_texts.to_csv('../data/test_texts.csv', index=False)


    # 取2000维度
    # test_texts = pd.read_csv('../data/train_texts.csv')
    # test_texts = test_texts.iloc[:, 0:2500]
    # test_texts.to_csv('../data/train_2500dim_texts.csv', index=False)

    # 转换为1列
    # test_texts = pd.read_csv('../data/test_2000dim_texts.csv', low_memory=False)
    # test_texts.to_csv('../data/test_texts2.csv', index=False, sep=' ')

    # test_texts = pd.read_csv('../data/test_texts2.csv')
    # test_texts = test_texts.iloc[:, 0:1000]
    # test_texts.to_csv('../data/train_1000dim_texts.csv', index=False)

    train_path = '../data/feature/train/'


    # 加载特征
    gen_access_num = pd.read_csv(train_path + 'gen_access_num.csv')
    gen_ftr51_stat = pd.read_csv(train_path + 'gen_ftr51_stat.csv')
    gen_ftr_stat = pd.read_csv(train_path + 'gen_ftr_stat4.csv')# 添加FRT_SUM是强持
    gen_ftr_cat = pd.read_csv(train_path + 'gen_ftr_cat.csv') # 没有效果
    # gen_ftr_sim = pd.read_csv(train_path + 'gen_ftr_sim2.csv') # ftr_sim_sum 万分位提升，可以删除
    access_day_num = pd.read_csv(train_path + 'access_day_num.csv')
    user_next_time_stat = pd.read_csv(train_path + 'user_next_time_stat2.csv')
    gen_ftr_nunique = pd.read_csv(train_path + 'gen_ftr_nunique2.csv')
    # gen_medicine_price = pd.read_csv(train_path + 'gen_medicine_price.csv')
    ftr51_unique_rate = pd.read_csv(train_path + 'ftr51_unique_rate.csv')
    gen_ftr51_len = pd.read_csv(train_path + 'gen_ftr51_len.csv')

    train = gen_access_num.merge(gen_ftr51_stat, on='PERSONID', how='left')
    train = train.merge(gen_ftr_stat, on='PERSONID', how='left')
    # train =train.merge(gen_ftr_sim, on='PERSONID', how='left') # 去除掉，LGB   0.9273055826815491
    train = train.merge(gen_ftr_cat, on='PERSONID', how='left')
    # train = train.merge(access_day_num, on='PERSONID', how='left')
    train = train.merge(user_next_time_stat, on='PERSONID', how='left')
    train = train.merge(gen_ftr_nunique, on='PERSONID', how='left')
    # train = train.merge(gen_medicine_price, on='PERSONID', how='left')
    train = train.merge(ftr51_unique_rate, on='PERSONID', how='left')
    train = train.merge(gen_ftr51_len, on='PERSONID', how='left')
    train.to_csv('../data/train_B.csv', index=False)
