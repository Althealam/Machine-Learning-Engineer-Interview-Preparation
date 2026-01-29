import numpy as np

batch_size = 4
feature_num = 2
feature_dim = 3

x = np.random.randn(batch_size, feature_num, feature_dim)
gamma = np.ones(feature_dim)
beta = np.ones(feature_dim)

def layer_norm_forward(x, gamma, beta, eps = 1e-5):
    """
    x: (batch_size, feature_num, feature_dim)
    gamma: (feature_dim, )
    beta: (feature_dim, )
    eps: (feature_dim, )
    对每个(feature_num, feature_dim)在维度feature_dim上做normalization
    """
    # 1. layer mean over feature_dim
    # keepdims = True的作用是，保留被归一化后的维度
    # 如果不加上keepdims = True，则维度为(batch_size, feature_num)
    # 加上keepdims = True后，维度为(batch_size, feature_num, 1) 
    mu = np.mean(x, axis=-1, keepdims = True)
    print("The shape of mu:", mu.shape)

    # 2. layer variance over feature_dim
    var = np.var(x, axis=-1, keepdims = True)
    print("The shape of variance:", var.shape)

    # 3. normalization
    # x_hat: (batch_size, feature_num, feature_dim)
    x_hat = (x-mu)/np.sqrt(var+eps)
    print("The shape of x_hat:", x_hat.shape)

    # 4. scale&shift
    out = gamma*x_hat+beta

    return out

output = layer_norm_forward(x, gamma, beta)
print("The shape of output:", output.shape)
# print("The value of output:", output) 