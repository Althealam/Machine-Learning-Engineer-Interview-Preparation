import numpy as np
def bce_loss(p, y):
    loss = -y*np.log(p)-(1-y)*np.log(1-p)
    return np.mean(loss)

def focal_bce_loss(p, y, alpha=0.25, gamma=2):
    """
    alpha: 调节正负样本的相对权重，正样本乘上alpha，负样本乘上1-alpha
    gamma: 控制对容易样本的降权程度，gamma越大，表示对容易样本的降权越强，训练越侧重难样本
    假设gamma=2，对于正样本而言
    - 容易样本：p=0.9，权重为0.1^2=0.01
    - 难样本：p=0.1，权重为0.9^2=0.81
    """
    loss = -alpha*(1-p)**gamma*y*np.log(p)-(1-alpha)*p**gamma*(1-y)*np.log(1-p)
    return np.mean(loss)