# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     utils
   Description :
   Author :       Administrator
   date：          2018/7/12 0012
-------------------------------------------------
   Change Activity:
                   2018/7/12 0012:
-------------------------------------------------
"""
__author__ = 'Administrator'
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn import preprocessing


def gen_merge_count(self, df, columns, value, name):
    gen_merge_count = pd.DataFrame(df.groupby(columns)[value].count()).reset_index()
    gen_merge_count.columns = name
    return gen_merge_count


def gen_merge_median(self, df, columns, value, name):
    gen_merge_median = pd.DataFrame(df.groupby(columns)[value].median()).reset_index()
    gen_merge_median.columns = name
    return gen_merge_median


def gen_merge_mean(self, df, columns, value, name):
    gen_merge_mean = pd.DataFrame(df.groupby(columns)[value].mean()).reset_index()
    gen_merge_mean.columns = name
    return gen_merge_mean


def gen_merge_sum(self, df, columns, value, name):
    gen_merge_sum = pd.DataFrame(df.groupby(columns)[value].sum()).reset_index()
    gen_merge_sum.columns = name
    return gen_merge_sum


def gen_merge_max(self, df, columns, value, name):
    gen_merge_max = pd.DataFrame(df.groupby(columns)[value].max()).reset_index()
    gen_merge_max.columns = name
    return gen_merge_max


def gen_merge_min(self, df, columns, value, name):
    gen_merge_min = pd.DataFrame(df.groupby(columns)[value].min()).reset_index()
    gen_merge_min.columns = name
    return gen_merge_min


def gen_merge_std(self, df, columns, value, name):
    gen_merge_std = pd.DataFrame(df.groupby(columns)[value].std()).reset_index()
    gen_merge_std.columns = name
    return gen_merge_std
