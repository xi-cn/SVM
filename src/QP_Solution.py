import numpy as np
from cvxopt import solvers
from cvxopt import matrix

def train_QP(X:np.ndarray, y:np.ndarray,C:float):
    '''
    使用cvxopt求解二次规划问题
    :param X: x
    :param y: y
    :param C: 惩罚项系数
    :return: alpha, b, w
    '''
    X_ = X.astype(float)
    n = X_.shape[0]
    for i in range(n):
        X_[i] = y[i] * X_[i]
    
    P = matrix(np.dot(X_, X_.transpose()))
    q = matrix(np.ones(n)*-1)
    G = matrix(np.vstack([np.identity(n)*-1, np.identity(n)]))
    h = matrix(np.hstack([np.zeros(n), np.ones(n)*C]))
    A = matrix(y.astype(np.float64), (1, n))
    b = matrix(0.0)

    solvers.options['show_progress'] = False
    solvers.options['abstol'] = 1e-10
    solvers.options['reltol'] = 1e-10
    solvers.options['feastol'] = 1e-10

    solution = solvers.qp(P, q, G, h, A, b)

    alpha = np.ravel(solution['x']) 

    #更新w
    w = np.zeros(shape=(X.shape[1]))
    for i in range(n):
        w += y[i] * alpha[i] * X[i]
    #更新b
    b = 0
    for xi, yi in zip(X, y):
        b += yi
        b -= np.dot(w, xi)
    b /= n

    return alpha, b, w