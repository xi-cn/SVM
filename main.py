from src.SVM import SVM
from src.data_prep import get_Train_Data
from src.data_prep import get_Test_Data
from src.data_prep import get_Test_Img
import joblib
import logging
logging.basicConfig(level=logging.INFO)
from matplotlib import pyplot as plt
from math import sqrt

def draw_Picture(img, y, result, N):
    '''
    绘制检测结果
    :param img: 检测图像
    :param y: 图像标注
    :param result: 检测结果
    :param N: 图像数量
    '''
    if sqrt(N) == int(sqrt(N)):
        n = int(sqrt(N))
    else:
        n = int(sqrt(N)) + 1

    plt.figure()
    for i in range(len(X)):
        ax = plt.subplot(n, n, i+1, xticks=[], yticks=[])
        ax.imshow(img[i], 'gray')
        #将错误数据集的图像边框绘制为红色 并标注错误答案
        if result[i] != y[i]:
            for spine in ax.spines.values():
                spine.set_color('red')
                spine.set_linewidth(2)
            ax.text(0, 10, str(result[i]), color='yellow', fontsize=10)


if __name__ == "__main__":

    #获取训练集和测试集
    train_X, train_y = get_Train_Data(5000)
    test_X, test_y = get_Test_Data(10000)

    #使用SMO优化
    svm = SVM(train_X, train_y, epoches=20, useSMO=True, C=1, test_X=test_X, test_y=test_y)
    svm.train()
    joblib.dump(svm, "svm.dpl")

    # #二次规划求解
    # svm = SVM(train_X, train_y, C=1, useSMO=False, test_X=test_X, test_y=test_y)
    # svm.train()
    # joblib.dump(svm, "svm.dpl")

    #检测
    svm = joblib.load("svm.dpl", 'r')
    X, y, img = get_Test_Img(100)
    result = []
    N = len(X)
    for i in range(N):
        result.append(svm.detect(X[i]))

    #绘制检测结果
    draw_Picture(img, y, result, N)
    plt.show()