import numpy as np

def batch_norm(x, gamma, beta, eps=1e-5):
    """
    x: (batch_size, feature_num, feature_dim)
    BN对每个feature_dim上的特征，沿着batch_size和feature_num计算均值和方差
    """
    # batch mean: mean over batch dimension
    mu = np.mean(x, axis=(0, 1), keepdims = True) # (1, 1, feature_dim)

    var = np.var(x, axis=(0, 1), keepdims = True) # (1, 1, feature_dim)

    x_hat = (x-mu)/np.sqrt(var+eps)
    # (batch_size, feature_num, feature_dim)

    out = gamma*x_hat+beta
    return out

batch_size = 3
feature_num = 4
feature_dim = 4
x = np.random.randint(size = (batch_size, feature_num, feature_dim))

gamma = np.ones(feature_dim)
beta = np.zeros(feature_dim)
print(batch_norm(x, gamma, beta).shape)