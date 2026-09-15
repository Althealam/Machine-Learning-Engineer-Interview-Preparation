y_pred = [0.5, 0.4, 0.6, 0.4, 0.3]
y_true = [1, 0, 0, 1, 0]
# 时间复杂度：O(nlogn)
# 空间复杂度：O(n)
def auc(y_pred, y_true):
    data = sorted(zip(y_pred, y_true)) # 按照y_pred来排序
    pos = sum(y_true)
    neg = len(y_true)-pos
    rank_sum = 0
    for i in range(len(data)):
        if data[i][1]==1: # 找到正样本的排序之和
            rank_sum+=i+1
    total = pos*neg
    auc = (rank_sum-pos*(pos+1)/2)/total
    return auc