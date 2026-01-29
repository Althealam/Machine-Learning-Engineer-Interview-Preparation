import numpy as np

batch_size = 4 # 总共有batch_size个样本
feature_num = 2 # 总共有feature_num个特征
feature_dim = 3 # 每个features的维度

x = np.random.randn(batch_size, feature_num, feature_dim)
gamma = np.ones(feature_dim)
beta = np.ones(feature_dim)

def batch_norm_forward(x, gamma, beta, eps = 1e-5):
    """
    x: (batch_size, feature_num, feature_dim) 
    gamma: (feature_dim, )
    beta: (feature_dim, )
    eps: numerical stablity
    对每个(feature_num, feature_dim)在维度batch_size上做batch norm
    """
    # 1. batch mean：mean over batch dimension
    mu = np.mean(x, axis=0)  # (feature_num, feature_dim)
    print("The shape of mu:", mu.shape)

    # 2. batch variance
    var = np.var(x, axis=0) # (feature_num, feature_dim)
    print("The shape of variance:", var)

    # 3. normalize
    x_hat = (x-mu)/np.sqrt(var+eps)
    print("The shape of x_hat:", x_hat.shape)

    # 4. scale&shift
    out = gamma*x_hat+beta

    return out

output = batch_norm_forward(x, gamma, beta)
print("The shape of output:", output)
print("The value of output:", output)
