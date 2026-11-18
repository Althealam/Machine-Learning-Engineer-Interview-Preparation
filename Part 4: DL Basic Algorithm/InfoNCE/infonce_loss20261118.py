import numpy as np

def infonce(Q, K, temperature = 0.7):
    """
    Q: [N, D]
    K: [N, D]
    """
    scores = (Q@K.T)/temperature # [N, N]
    exp_scores = np.exp(scores) # [N, N]

    # 分子：正样本的分数，即矩阵的对角线
    numerator = []
    for i in range(len(exp_scores)):
        numerator.append(exp_scores[i][i])
    numerator = np.array(numerator)

    # 分母：所有候选的分数之和，包含正样本
    denominator = np.sum(exp_scores, axis=1)

    loss_per_sample = -np.log(numerator/denominator)

    return np.mean(loss_per_sample)
