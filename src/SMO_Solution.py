import random
import numpy as np

def meet_KKT(i):
    '''
    判断 alpha[i] 是否满足KKT条件
    '''
    if 0 <= alpha_[i] <= C_ and -toler_ <= Ei_[i] <= toler_:
        return True
    else:
        return False

def optimize_alpha_ij(i, j)->bool:
    '''
    :param i:   alpha[i]
    :param j:   alpha[j]
    :return:    是否优化成功
    '''
    if i == j:
        return False
    #获取alpha1和y1
    alpha1 = alpha_[i]
    y1 = y_[i]
    #获取E1
    E1 = Ei_[i]
    
    #获取alpha2和y2
    alpha2 = alpha_[j]
    y2 = y_[j]
    #获取E2
    E2 = Ei_[j]

    #计算eta
    k11 = X_metric_[i][i]
    k12 = X_metric_[i][j]
    k22 = X_metric_[j][j]
    eta = k11 + k22 - 2*k12

    if eta <= 0:
        return False
    
    #计算上下界 H L
    if y1 != y2:
        H = min(C_, C_ - alpha1 + alpha2)
        L = max(0, alpha2 - alpha1)
    else:
        H = min(C_, alpha1 + alpha2)
        L = max(0, alpha1 + alpha2 - C_)

    if L == H:
        return False

    alpha2_old = alpha2
    a2 = alpha2 + (E1 - E2)*y2/eta
    if a2 > H:
        alpha2 = H
    elif a2 < L:
        alpha2 = L
    else:
        alpha2 = a2

    #更新alpha1
    alpha1_old = alpha1
    alpha1 = alpha1_old + y1*y2*(alpha2_old - alpha2)
    if -toler_ < alpha1 < toler_:
        alpha1 = 0
    elif C_ - toler_ < alpha1 < C_:
        alpha1 = C_

    #变化太小则认为优化失败
    if abs(alpha1 - alpha1_old) < toler_:
        return False

    #更新b
    global b_
    b_old = b_
    if 0 < alpha1 < C_:
        b_new = -E1 - (alpha1-alpha1_old)*y1*k11 - (alpha2-alpha2_old)*y2*k12 + b_
    elif 0 < alpha2 < C_:
        b_new = -E2 - (alpha1-alpha1_old)*y1*k12 - (alpha2-alpha2_old)*y2*k22 + b_
    else:
        b1 = -E1 - (alpha1-alpha1_old)*y1*k11 - (alpha2-alpha2_old)*y2*k12 + b_
        b2 = -E2 - (alpha1-alpha1_old)*y1*k12 - (alpha2-alpha2_old)*y2*k22 + b_
        b_new = (b1 + b2) / 2

    #更新Ei
    t1 = y1*(alpha1-alpha1_old)
    t2 = y2*(alpha2-alpha2_old)
    deltb = b_new - b_old
    for k in range(N):
        # if 0 < alpha_[k] < C_:
        Ei_[k] = Ei_[k] + t1*X_metric_[i][k] + t2*X_metric_[j][k] + deltb

    #更新alpha1 alpha2 b
    alpha_[i] = alpha1
    alpha_[j] = alpha2
    b_ = b_new

    return True
    
    

def optimize(i)->bool:
    '''
    优化第i个节点
    节点j的选择方式有两种
    第一种: 从不满足KKT条件的节点中随机挑选一个节点
    第二种: 从所有数据集中随机挑选一个节点
    如果成功返回True 否则返回False
    '''
    #随机挑选不满足KKT条件的点
    index = random.randint(0, N-1)
    while index < N:
        if not meet_KKT(index):
            break
        index += 1
    if index < N:
        success = optimize_alpha_ij(i, index)
        if success:
            return True
    #若优化失败则从全部数据集中随机挑选一个
    index = random.randint(0, N-1)
    success = optimize_alpha_ij(i, index)
    if success:
        return True
    #以上三种方式均优化失败则返回False
    return False


def train_SMO(X:np.ndarray, y:np.ndarray, b:float, X_metric:np.ndarray, alpha:np.ndarray, Ei:np.ndarray, C:float, toler:float, w=None):
    '''
    SMO优化迭代一次
    :param X:           x
    :param y:           y
    :param b:           b
    :param X_metric:    x的点积矩阵
    :param alpha:       待优化参数
    :param Ei:          Ei矩阵
    :param C:           惩罚项系数
    :param toler:       误差项系数
    :param w:           如果是采用线性核则可传入w 方便后续用于检测
    :return:            b, w
    '''

    #声明全局变量 便于其他函数使用
    global X_
    global y_
    global b_
    global X_metric_
    global alpha_
    global Ei_
    global C_
    global toler_
    global N

    X_ = X
    y_ = y
    b_ = b
    X_metric_ = X_metric
    alpha_ = alpha
    Ei_ = Ei
    C_ = C
    toler_ = toler
    N = X.shape[0]

    #优化不满足KKT条件的点
    change = 0
    for i in range(N):
        #判断第i个节点是否满足KKT条件
        if not meet_KKT(i):
            if optimize(i):
                change += 1

    #优化全部节点
    change = 0
    for i in range(N):
        if optimize(i):
            change += 1

    #更新w
    if type(w) != type(None):
        w = np.zeros(shape=(X_.shape[1]))
        for i in range(N):
            w += y_[i] * alpha_[i] * X_[i]

    return b_, w