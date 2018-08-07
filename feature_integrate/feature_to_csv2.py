# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     feature_to_csv2
   Description :
   Author :       Administrator
   date：          2018/7/13 0013
-------------------------------------------------
   Change Activity:
                   2018/7/13 0013:
-------------------------------------------------
"""
__author__ = 'Administrator'
import pandas as pd

# feature_importance = pd.read_csv('../data/feature_importance/fi.csv')
# feature_importance = pd.read_csv('../data/model/feat_importance7.csv')
data_path = "../data/test_B.tsv"
test_data = pd.read_csv(data_path, error_bad_lines=False, sep='\t')
# print(test_data[test_data['PERSONID']=='3e09d3c6f779cc2f6d438636ed5cc235'].head())
print(test_data.groupby('PERSONID')['FTR0'].first().reset_index().head(30))
print(test_data.groupby('PERSONID')['FTR0'].first().reset_index().shape)

# print(train.columns.tolist())