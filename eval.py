import cv2
import numpy as np


y_true = cv2.imread('0.jpg')
y_pred = cv2.imread('1.jpg')


def evaluation_TPR(y_true, y_pred):
    '''
    真实图是1中预测图是1的概率
    叫真正类率，也就是白色部分的正确率
    :param y_true:
    :param y_pred:
    :return:
    '''

    # 二值化
    y_true, y_pred = y_true / 255, y_pred / 255
    # 所有真实图1
    all1 = np.sum(y_true)
    # 是11的部分
    temp11 = y_pred * y_true
    TP11 = np.sum(temp11)
    # 是10的部分
    # FN10=all1-TP11
    return TP11 / all1
    pass


def evaluation_FPR(y_true, y_pred):
    '''
    原图是0现在1的概率，也叫假正类率
    原预
    =01/（00+01）
    :param y_true: 
    :param y_pred: 
    :return: 
    '''
    # 图像反转0->1
    y_true, y_pred = 255 - y_true, 255 - y_pred
    # 原图0，预测是0的概率=翻转后TP11
    TP11 = evaluation_TPR(y_true, y_pred)
    return 1 - TP11
    pass


def eval_TPR_mean(y_true, y_pred):
    '''
    传入队列
    :param y_true: 
    :param y_pred: 
    :return: 
    '''
    L = len(y_true)
    assert L == len(y_pred)
    R = 0
    for each_true, each_pred in zip(y_true, y_pred):
        R += evaluation_TPR(each_true, each_pred)
        pass
    return R / L


def eval_FPR_mean(y_true, y_pred):
    L = len(y_true)
    assert L == len(y_pred)
    R = 0
    for each_true, each_pred in zip(y_true, y_pred):
        R += evaluation_FPR(each_true, each_pred)
        pass
    return R / L
    pass


def timeRate(predictTime):
    '''
    
    :param predictTime: 单位秒
    :return: 
    '''
    predictTime /= 1000
    return (1 + (predictTime - 300) / 10000)
    pass


def accuracy(TPR, FPR):
    '''

    :param TPR: 真正率的均值
    :param FPR: 
    :return: 
    '''
    R = np.array([TPR, FPR])
    vec = np.array([1, 0])
    temp = np.linalg.norm(R - vec)
    return temp
    pass


def final_score(accuracy, timeRate):
    '''
    最终成绩
    :param accuracy: 
    :param tiemRate: 
    :return: 
    '''
    return timeRate * accuracy


if __name__ == '__main__':
    print(evaluation_TPR(y_true, y_pred))
    TPR_mean = eval_TPR_mean([y_true], [y_pred])
    print(TPR_mean)
    print(evaluation_FPR(y_true, y_pred))
    FPR_mean = eval_FPR_mean([y_true], [y_pred])
    print(FPR_mean)
    accuracy = accuracy(TPR_mean, FPR_mean)
    print('accuracy:', accuracy)
    print('final_score:', final_score(accuracy, 300))
