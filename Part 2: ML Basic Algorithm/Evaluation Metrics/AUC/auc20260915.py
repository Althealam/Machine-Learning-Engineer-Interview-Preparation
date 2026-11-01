y_pred = [0.5, 0.4, 0.6, 0.4, 0.3]
y_true = [1, 0, 0, 1, 0]
# 时间复杂度：O(nlogn)
# 空间复杂度：O(n)
# 边界条件：假设有一个正样本和一个负样本，其打分一样，那么排序后就是[(0.4, 0), (0.4, 1)]，会认为正样本的rank为2
# 相同score的样本使用平均的rank
def auc(y_pred, y_true):
    data = sorted(zip(y_pred, y_true)) # 按照y_pred来排序，如果y_pred一样就会按照y_true来排序 
    pos = sum(y_true)
    neg = len(y_true)-pos
    n = len(y_pred)
    rank_sum = 0
    
    # 找到所有score相同的样本，并计算rank
    i = 0
    while i<n:
        j = i
        # 找到所有score相同的样本
        while j<n and data[j][0]==data[i][0]:
            j+=1
        # rank从1开始，这一组占据的rank是i+1到j，由此计算首尾平均值
        # 连续整数的平均值就是首尾平均值
        avg_rank = ((i+1)+j)/2

        # 这一组里所有的正样本都使用avg_rank
        for k in range(i, j):
            if data[k][1]==1:
                rank_sum+=avg_rank
        
        i = j

    total = pos*neg
    auc = (rank_sum-pos*(pos+1)/2)/total
    return auc