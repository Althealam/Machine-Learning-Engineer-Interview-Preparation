# layernorm: 对每个样本的特征维度进行归一化，使得归一化后的均值为0，方差接近1，再通过可学习的参数进行缩放和平一
import numpy as np

def layernorm(x, gamma, beta, eps=1e-5):
    """
    x: (batch_size, feature_num, feature_dim)
    layer normalization对每个x在feature_dim维度上做normalization
    沿着feature_dim做归一化 ：对于每个token，取出它的所有特征，计算这些特征的均值和方差，使用该均值和方差对该token的所有特征进行归一化
    """
    # axis=-1表示沿着feature_dim的维度进行归一化 -> 每个token都有自己的独立均值
    # (batch_size, feature_num, 1)
    mu = np.mean(x, axis=-1, keepdims = True)

    # (batch_size, feature_num, 1) -> 每个token都有自己的独立方差
    var = np.var(x, axis=-1, keepdims = True)

    # normalization
    # (batch_size, feature_num, feature_dim)
    # 每个样本减去自己的维度为(batch_size, feature_num, 1)的均值
    # 并且除以自己的维度为(batch_size, feature_num, 1)的方差
    x_hat = (x-mu)/np.sqrt(var+eps)

    out = gamma*x_hat+beta

    return out

batch_size = 3
feature_num = 4
feature_dim = 4
x = np.random.randint(size=(batch_size, feature_num, feature_dim))
# 因为layernorm是沿着feature_dim维度进行归一化的，因此每个特征维度都需要有自己的独立缩放系数
gamma = np.ones(feature_dim)
beta = np.zeros(feature_dim) 
print(layernorm(x, gamma, beta))