# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     ffm_model
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

np.random.seed(0)
import math
from logistic import Logistic


# 该类表示一个样本的一个特征
class FFM_Node(object):
    '''
    通常x是高维稀疏向量，所以用链表来表示一个x，链表上的每个节点是个3元组(j,f,v)，表示一个样本x的一个非0特征
    '''
    __slots__ = ['j', 'f', 'v']  # 按元组（而不是字典）的方式来存储类的成员属性

    def __init__(self, j, f, v):
        '''
        :param j: Feature index (0 to n-1)
        :param f: Field index (0 to m-1)
        :param v: value
        '''
        self.j = j
        self.f = f
        self.v = v


class FFM(object):
    def __init__(self, m, n, k, eta, lambd):
        # m 域个数，n特征个数，k隐变量维度，eta学习速率，lambd正则系数
        self.m = m
        self.n = n
        self.k = k
        # 超参数
        self.eta = eta
        self.lambd = lambd
        # 初始化三维权重矩阵w~U(0,1/sqrt(k))
        self.w = np.random.rand(n, m, k) / math.sqrt(k)
        # 初始化累积梯度平方和为，AdaGrad时要用到，防止除0异常
        self.G = np.ones(shape=(n, m, k), dtype=np.float64)
        self.log = Logistic()

    # 特征组合式的线性加权求和
    def phi(self, node_list):
        #  node_list: 一个样本，用链表存储x中的非0值
        z = 0.0
        for a in range(len(node_list)):
            node1 = node_list[a]
            j1 = node1.j
            f1 = node1.f
            v1 = node1.v
            for b in range(a + 1, len(node_list)):
                node2 = node_list[b]
                j2 = node2.j
                f2 = node2.f
                v2 = node2.v
                w1 = self.w[j1, f2]
                w2 = self.w[j2, f1]
                z += np.dot(w1, w2) * v1 * v2  # 域感知的因子分解法FFM
        return z

    # 输入x，预测y的值
    def predict(self, node_list):
        # node_list: 用链表存储x中的非0值
        z = self.phi(node_list)
        y = self.log.decide_by_tanh(z)
        return y

    # 根据一个样本来更新模型参数
    def sgd(self, node_list, y):
        # node_list: 用链表存储x中的非0值。 y: 正样本1，负样本-1
        kappa = -y / (1 + math.exp(y * self.phi(node_list)))
        for a in range(len(node_list)):
            node1 = node_list[a]
            j1 = node1.j
            f1 = node1.f
            v1 = node1.v
            for b in range(a + 1, len(node_list)):
                node2 = node_list[b]
                j2 = node2.j
                f2 = node2.f
                v2 = node2.v
                c = kappa * v1 * v2
                # self.w[j1,f2]和self.w[j2,f1]是向量，导致g_j1_f2和g_j2_f1也是向量
                g_j1_f2 = self.lambd * self.w[j1, f2] + c * self.w[j2, f1]
                g_j2_f1 = self.lambd * self.w[j2, f1] + c * self.w[j1, f2]
                # 计算各个维度上的梯度累积平方和
                self.G[j1, f2] += g_j1_f2 ** 2  # 所有G肯定是大于0的正数，因为初始化时G都为1
                self.G[j2, f1] += g_j2_f1 ** 2
                # AdaGrad
                self.w[j1, f2] -= self.eta / np.sqrt(self.G[j1, f2]) * g_j1_f2  # sqrt(G)作为分母，所以G必须是大于0的正数
                self.w[j2, f1] -= self.eta / np.sqrt(self.G[j2, f1]) * g_j2_f1  # math.sqrt()只能接收一个数字作为参数，而numpy.sqrt()可以接收一个array作为参数，表示对array中的每个元素分别开方

    # 根据一堆样本训练模型
    def train(self, sample_generator, max_echo, max_r2):
        '''
        :param sample_generator: 样本生成器，每次yield (node_list, y)，node_list中存储的是x的非0值。通常x要事先做好归一化，即模长为1，这样精度会略微高一点
        :param max_echo: 最大迭代次数
        :param max_r2: 拟合系数r2达到阈值时即可终止学习
        :return:
        '''
        for itr in range(max_echo):
            print("echo", itr)
            y_sum = 0.0
            y_square_sum = 0.0
            err_square_sum = 0.0  # 误差平方和
            population = 0  # 样本总数
            for node_list, y in sample_generator:
                self.sgd(node_list, y)
                y = 0.0 if y == -1 else y  # 真实的y取值为{-1,1}，而预测的y位于(0,1)，计算拟合效果时需要进行统一
                y_hat = self.predict(node_list)
                y_sum += y
                y_square_sum += y ** 2
                err_square_sum += (y - y_hat) ** 2
                population += 1
            var_y = y_square_sum - y_sum * y_sum / population  # y的方差
            r2 = 1 - err_square_sum / var_y
            print("r2=",r2)
            if r2 > max_r2:  # r2值越大说明拟合得越好
                print('r2 have reach', r2)
                break

    # 序列化模型，保存到文件
    def save_model(self, outfile):
        np.save(outfile, self.w)

    # 从文件中加载模型
    def load_model(self, infile):
        self.w = np.load(infile)

if __name__ == '__main__':
    ffm = FFM()

    ffm.train()

    ffm.predict()