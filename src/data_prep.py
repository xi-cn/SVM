import gzip
import os
import numpy as np
import logging
import random

def parse_mnist(minst_file_addr: str = None, flatten: bool = False, one_hot: bool = False) -> np.array:
    """解析MNIST二进制文件, 并返回解析结果
    输入参数:
        minst_file: MNIST数据集的文件地址. 类型: 字符串.
        flatten: bool, 默认Fasle. 是否将图片展开, 即(n张, 28, 28)变成(n张, 784)
        one_hot: bool, 默认Fasle. 标签是否采用one hot形式.

    返回值:
        解析后的numpy数组
    """
    if minst_file_addr is not None:
        minst_file_name = os.path.basename(minst_file_addr)  # 根据地址获取MNIST文件名字
        with gzip.open(filename=minst_file_addr, mode="rb") as minst_file:
            mnist_file_content = minst_file.read()
        if "label" in minst_file_name:  # 传入的为标签二进制编码文件地址
            data = np.frombuffer(buffer=mnist_file_content, dtype=np.uint8, offset=8)  # MNIST标签文件的前8个字节为描述性内容，直接从第九个字节开始读取标签，并解析
            if one_hot:
                data_zeros = np.zeros(shape=(data.size, 10))
                for idx, label in enumerate(data):
                    data_zeros[idx, label] = 1
                data = data_zeros
        else:  # 传入的为图片二进制编码文件地址
            data = np.frombuffer(buffer=mnist_file_content, dtype=np.uint8, offset=16)  # MNIST图片文件的前16个字节为描述性内容，直接从第九个字节开始读取标签，并解析
            data = data.reshape(-1, 784) if flatten else data.reshape(-1, 28, 28)
    else:
        logging.warning(msg="请传入MNIST文件地址!")

    return data


def direct_Data(img:np.ndarray):
    '''
    将矩阵转换为1维 像素值大于等于128赋值为1 否则为0
    @param:img 待转换的图像矩阵
    @return: 转换后的图像矩阵
    '''
    data = img.reshape((784))
    data[data < 128] = 0
    data[data >= 128] = 1
    return data

def get_Train_Data(train_num=5000):
    '''
    获取训练集
    @param:total_num 训练集的总数 -1则选取所有数据集(60000)
    @return: 训练集字典
    '''
    #获取图像数标注吧
    train_imgs = parse_mnist(minst_file_addr="./data/train_set/train-images-idx3-ubyte.gz").copy().astype(np.uint16)
    train_labels = parse_mnist(minst_file_addr="./data/train_set/train-labels-idx1-ubyte.gz").copy()
    #随机排序
    perm = np.random.permutation(train_imgs.shape[0])
    train_imgs = train_imgs[perm]
    train_labels = train_labels[perm]

    #数据处理
    imgs = []
    labels = []
    single_num = train_num / 10
    num = [single_num] * 10
    for i, l in zip(train_imgs, train_labels):
        if num[l] < 0:
            continue
        else:
            num[l] -= 1
            imgs.append(direct_Data(i))
            labels.append(l)

    return np.array(imgs), np.array(labels)

def get_Test_Data(test_num=500):
    '''
    获取测试集
    @param:total_num 测试集的总数 -1则选取所有数据集(10000)
    @return: X, y
    '''
    test_imgs = parse_mnist(minst_file_addr="./data/test_set/t10k-images-idx3-ubyte.gz").copy()
    test_labels = parse_mnist(minst_file_addr="./data/test_set/t10k-labels-idx1-ubyte.gz").copy()

    #随机挑选
    if test_num >= 10 or test_num < test_imgs.shape[0]:
        index = random.sample(range(test_imgs.shape[0]), test_num)
        test_imgs = test_imgs[index]
        test_labels = test_labels[index]

    #数据处理
    imgs = []
    for i in range(test_imgs.shape[0]):
        imgs.append(direct_Data(test_imgs[i]))

    return np.array(imgs), test_labels


def get_Test_Img(num=100):
    '''
    获取用于可视化的数据集
    :param num: 测试集的数量
    :return:
    '''
    test_imgs = parse_mnist(minst_file_addr="./data/test_set/t10k-images-idx3-ubyte.gz").copy()
    test_labels = parse_mnist(minst_file_addr="./data/test_set/t10k-labels-idx1-ubyte.gz").copy()
    #随机挑选
    index = random.sample(range(test_imgs.shape[0]), num)
    test_imgs = test_imgs[index]
    test_labels = test_labels[index]
    #图像二值化处理
    X = []
    for img in test_imgs:
        X.append(direct_Data(img))

    return np.array(X), test_labels, test_imgs
