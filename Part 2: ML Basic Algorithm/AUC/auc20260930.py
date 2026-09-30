y_pred = [0.5, 0.3, 0.4, 0.2, 0.2]
y_true = [1, 0, 0, 1, 1, 0]
# 假设第k个正样本的排名是rk，那么前面有rk-1个样本，k-1个正样本，rk-k个负样本
# 由此可以知道第k个正样本超过的负样本数为rk-k
# 正样本超过负样本数之和为sum(rk-k)=rank_sum-sum(k)，其中sum(k)=p(p+1)/2

def auc1(y_pred, y_true):
    """
    step1: 将y_pred, y_true按照y_pred从小到大排序
    step2: auc = (pos_rank_sum-pos*(pos+1)/2)/total 其中pos_rank_sum是正样本的累加rank之和，pos是正样本的数量
    
    边界条件：[(0.5, 0), (0.5, 1)]，此时会认为正样本的rank是2，实际上应该是1（相同score应该取平均rank）
    AUC = P(pos>neg)+0.5P(pos=neg)
    """
    data = sorted(zip(y_pred, y_true)) # 按照y_pred从小到大排序
    pos = sum(y_pred)
    neg = len(y_pred)-pos
    rank_sum = 0
    # for i in range(len(data)):
    #     if data[i][1]==1:
    #         rank_sum+=i+1 # 计算正样本的累加和
    # 找到所有和score相同的样本，并计算rank
    while i<len(data):
        j = i
        # 找到所有score相同的样本
        while j<len(data) and data[j][0]==data[i][0]:
            j+=1
        # 连续整数的平均值就是首尾平均值
        avg_rank = ((i+1)+j)/2

        # 所有的正样本都用avg_rank
        for k in range(i, j):
            if data[k][1]==1:
                rank_sum+=avg_rank 

    i = 0
    total = pos*neg
    auc = (rank_sum-pos*(pos+1)/2)/total
    return auc

