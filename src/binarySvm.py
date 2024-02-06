import numpy as np
from src.SMO_Solution import train_SMO
from src.QP_Solution import train_QP


class binarySvm:
    '''
    二分类向量机
    '''
    X = None
    y = None
    w = None
    b = None
    N = None
    Ei = None
    k1 = None
    k2 = None
    C = None
    X_metric = None
    toler = None
    kernel = None
    useSMO = None
    alpha = None

    def __init__(self, data, k1:int, k2:int, C:float=1, toler:float=1e-5, kernel:str='linear', useSMO:bool=True) -> None:
        '''
        :param data: 数据集字典
        :param k1: 类别1
        :param k2: 类别2
        :param C: 惩罚系数
        :param toler: 误差项
        :param kernel: 核函数
        :param useSMO: 优化方式 True则使用SMO优化 False作为QP问题优化
        '''

        self.k1 = k1
        self.k2 = k2

        #初始化数据集和标签集
        self.X, self.y = self.select_Data(data, k1, k2)
        #数据集总数
        self.N = self.X.shape[0]
        #初始化b为0
        self.b = 0
        #核函数
        self.kernel = kernel
        #误差项
        self.toler = toler
        #svm惩罚系数
        self.C = C
        #是否使用SMO优化
        self.useSMO = useSMO
        #初始化w
        if self.kernel == "linear":
            self.w = np.zeros(shape=(self.X.shape[1]))
        if useSMO:
            #初始化 alpha为0
            self.alpha = np.zeros(self.N)
            #初始化 Ei
            self.Ei = -1 * self.y.astype(float)
            #初始化点积矩阵
            self.init_Metric()
                        
    def select_Data(self, data, k1, k2):
        '''
        从数据集中字典中选取指定数据
        :param data: 数据集字典
        :param k1: 标签为1的类别
        :param k2: 标签为-1的类别
        :return: X, y
        '''
        x = np.vstack([data[k1], data[k2]])
        y = [1] * len(data[k1]) + [-1] * len(data[k2])

        return x, np.array(y)
    def init_Metric(self):
        '''
        初始化点积矩阵
        '''
        if self.kernel == "linear":
            self.X_metric = np.dot(self.X, self.X.transpose())

    def train(self):
        '''
        训练函数
        '''
        if self.useSMO:
            self.b, self.w = train_SMO(self.X, self.y, self.b, self.X_metric, self.alpha, self.Ei, self.C, self.toler, self.w)
        else:
            self.alpha, self.b, self.w= train_QP(self.X, self.y,self.C)


    def learn_func(self, x):
        '''
        计算f(x)
        '''
        fx = 0
        if self.kernel == "linear":
            fx += np.dot(self.w, x.transpose())
        else:
            pass
        fx += self.b
        return fx   

    def detect(self, x):
        '''
        检测结果大于0返回k1
        否则返回k2
        '''
        y = self.learn_func(x)
        if y > 0:
            return self.k1
        else:
            return self.k2