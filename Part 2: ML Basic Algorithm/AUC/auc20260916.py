y_pred = [0.5, 0.3, 0.4, 0.2, 0.2]
y_true = [1, 0, 0, 1, 1, 0]

def auc(y_pred, y_true):
    data = sorted(zip(y_pred, y_true))
    pos = sum(y_true)
    neg = len(y_true)-pos
    n = len(y_pred)
    rank_sum = 0

    # 找到所有和score相同的样本，并计算avg_rank
    # 然后正样本的rank_sum则是加上avg_rank
    i = 0
    while i<n:
        j = i
        while j<n and data[j][0]==data[i][0]: # 分数相同
            j+=1
        avg_rank = ((i+1)+j)/2
        for k in range(i, j):
            if data[k][1]==1: # 累加正样本的avg_rank
                rank_sum+=avg_rank
    total = pos*neg
    auc = (rank_sum-pos*(pos+1)/2)/total
    return auc