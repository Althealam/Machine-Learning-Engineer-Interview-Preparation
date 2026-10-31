import numpy as np

y_pred = np.array([
    [0.7, 0.2, 0.1], # 第一个样本的预测概率
    [0.2, 0.3, 0.5] # 第二个样本的预测概率
])

y_true = np.array([
    [1, 0, 0],  # 第一个样本的正确分类为0
    [0, 0, 1] # 第二个样本的正确分类为2
])
def mse_loss(y_pred, y_true):
    return np.mean((y_pred-y_true)**2)

def mae_loss(y_pred, y_true):
    return np.mean(np.abs(y_pred-y_true))

def bce_loss(y_pred, y_true):
    return -y_true*np.log(y_pred)-(1-y_true)*np.log(1-y_pred)

# 假设有N个样本，C个类别，cross entropy loss为
# L = -1/N * sum(1..N) sum(1..C) y_ic log p_ic 里面是先验，外面是后验，并且sum对所有类别和所有样本求和，并且最后对所有样本求平均
def cross_entropy_loss(y_pred, y_true):
    # y_pred是每行之和为1的类别概率
    # y_true是one-hot编码或者目标概率分布
    # axis=-1表示沿着数组的最后一维进行操作，即沿着列求平均；axis=0表示沿着行求平均
    return np.mean(-y_true*np.log(y_pred), axis=-1)

print(cross_entropy_loss(y_pred, y_true))