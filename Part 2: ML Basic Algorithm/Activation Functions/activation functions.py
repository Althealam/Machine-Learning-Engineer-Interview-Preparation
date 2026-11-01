import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))


def tanh(x):
    return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))

def relu(x):
    if x>0:
        return x
    else:
        return 0

def leaky_relu(x, alpha):
    if x>0:
        return x
    else:
        return alpha*x

def softmax(x):
    exp_x = np.exp(x)
    return exp_x/np.sum(exp_x, axis=-1, keepdims=True)
# axis=-1表示沿着最后一个维度求和
# exp_x行是样本，列是特征
# keepdims = True表示计算后保留这一维，只把长度变为1 -> 如果不加上的话，会出现报错operands could not be broadcast together with shapes (2,3) (2,) 
# 不加keepdims=True的话，维度为(2, )；加上之后维度为(2, 1)

x = [
    [1, 2, 3], # 第一个样本
    [4, 5, 6] # 第二个样本
]
print(softmax(x))