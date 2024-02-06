import logging
import numpy as np
from src.binarySvm import binarySvm
from tqdm import tqdm
import random

class SVM:
    svms = None
    labels = None
    total_num = None
    testData = None
    test_X = None
    test_y = None
    n = None
    def __init__(self, X:np.ndarray, y:np.ndarray, C:float=1, useSMO:bool=True, epoches:int=1, **kwargs):
        '''
        支持向量机 可用于多分类
        :param X: x
        :param y: y
        :param C: 惩罚项系数
        :param useSMO: True则实用SMO优化 否则作为QP问题优化
        :param epoches: 迭代次数 若useSMO=False 则默认为1
        :param kwargs: {'test_X':测试集图像, 'test_y':测试集种类}
        '''
        self.epoches = epoches
        self.C = C
        self.useSMO = useSMO
        if not useSMO:
            self.epoches = 1
        #训练集
        train_data = self.parse_Data(X, y)

        #测试集
        self.test = False
        if "test_X" and "test_y" in kwargs.keys():
            self.test = True
            self.test_X = kwargs["test_X"]
            self.test_y = kwargs["test_y"]

        #初始化二分类器
        items = range(self.n * (self.n - 1) // 2)
        pbar = tqdm(items, "初始化二分类器", len(items))
        self.svms = []
        for i in self.labels:
            for j in self.labels:
                if i == j:
                    break
                svm = binarySvm(train_data, i, j, useSMO=useSMO)
                self.svms.append(svm)
                pbar.update(1)
        pbar.close()

    def parse_Data(self, X:np.ndarray, y:np.ndarray):
        '''
        将数据集转化为字典
        :param X:
        :param y:
        :return: 数据集字典
        '''
        #获取标签种类
        self.labels = np.unique(y)
        self.n = len(self.labels)
        #将数据转换为字典
        data = {key:[] for key in self.labels}
        for x_, y_ in zip(X, y):
            data[y_].append(x_)
        #转换为numpy对象
        for key in data.keys():
            data[key] = np.array(data[key])

        return data
    
    def train(self):
        '''
        训练
        '''
        for iteration in range(1, self.epoches+1):
            #训练
            items = range(len(self.svms))
            pbar = tqdm(items, "第{}次迭代".format(iteration), len(items))
            for svm in self.svms:
                svm.train()
                pbar.update(1)
            pbar.close()

            #检测
            if self.test:
                #检测
                acc_num = 0
                for x, y in zip(self.test_X, self.test_y):
                    result = self.detect(x)
                    if result == y:
                        acc_num += 1
                acc_rate = acc_num / len(self.test_y)
                logging.info("--------------------第{}次迭代, 精度为{:.4f}--------------------".format(iteration, acc_rate))

    def detect(self, x:np.ndarray):
        '''
        检测:获取每个二分类向量机的检测结果 对于种类得分+1 得分最高者认为是该图像的种类
        :param x: 待检测图像
        :return: 检测种类
        '''
        #获取每个二分类器的分类结果
        score = [0] * self.n
        for svm in self.svms:
            res = svm.detect(x)
            score[res] += 1
        #得分最大值
        max_score = 0
        for i in range(len(score)):
            if score[i] > max_score:
                max_score = score[i]
        #得分最大值的元素
        posble = []
        for i in range(len(score)):
            if score[i] == max_score:
                posble.append(i)
        #若存在多个得分最大值 则随机返回
        if len(posble) == 1:
            return posble[0]
        else:
            return random.choice(posble)