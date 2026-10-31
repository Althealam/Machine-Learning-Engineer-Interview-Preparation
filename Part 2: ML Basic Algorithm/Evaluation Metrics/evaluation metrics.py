import numpy as np

# 分类问题

def accuracy(y_true, y_pred):
    """
    accuracy = tn+tp / (tn+tp+fn+fp)
    """
    tn, tp, fn, fp = 0, 0, 0, 0
    for (true, pred) in zip(y_true, y_pred):
        if true==pred:
            if true==1:
                tp+=1
            else:
                tn+=1
        else:
            if true==1:
                fn+=1
            else:
                fp+=1
    if tn+tp+fn+fp==0:
        return 0
    return (tn+tp)/(tn+tp+fn+fp)


def precision(y_true, y_pred):
    """
    precision = tp/(tp+fp)
    """
    tp, fp = 0, 0
    for true, pred in zip(y_true, y_pred):
        if pred==1:
            if true==1:
                tp+=1
            else:
                fp+=1
    if tp+fp==0:
        return 0
    return tp/(tp+fp)

def recall(y_true, y_pred):
    """
    recall = tp/(tp+fn)
    """
    tp, fn = 0, 0
    for true, pred in zip(y_true, y_pred):
        if true==1:
            if pred==1:
                tp+=1
            else:
                fn+=1
    if tp+fn==0:
        return 0
    return tp/(tp+fn)

def auc1(y_true, y_pred):
    """
    auc = p(score_pos>score_neg)+0.5p(score_pos==score_neg)
    时间复杂度：O(p*m)，其中p为正样本的数量，m为负样本的数量，p+m=n，最坏的情况下为O(n^2)
    空间复杂度：O(n) 最坏的情况下所有的样本都是正样本/负样本
    """
    pos_scores = []
    neg_scores = []

    for i in range(len(y_true)):
        if y_true[i]==1:
            pos_scores.append(y_true[i])
        else:
            neg_scores.append(y_true[i])

    # 正负样本两两比较
    correct = 0
    for pos in pos_scores:
        for neg in neg_scores:
            total+=1
            if pos>neg:
                correct+=1
            elif pos==neg:
                correct+=0.5
    if total==0:
        return 0
    return correct/total

def auc2(y_true, y_pred):
    """
    「关键在于，找到正样本分数【超过/等于】负样本数量的配对数」
    1. 找正样本【超过/等于】负样本的数量 -> 计1分
    假设第n个正样本按照分数升序排序时的排名是m，其前面有n-1个正样本，m-1个样本，m-n个负样本
    所有正样本分数超过负样本分数的配对数为sum(m-n)
    - 根据等差求和公式，sum(m) = rank_sum
    - sum(n)=pos*(pos+1)，其中pos为正样本的数量
    （等差求和推到：s=1+...+p=p+(p-1)+...+1 -> 2s = (1+p)+...+(1+p) = p*(p+1) -> s=p(p+1)/2
    sum(m-n) = (rank_sum-pos*(pos+1))/total

    2. 找正样本【等于】负样本的配对数
    假设[(0.5, 0), (0.5, 1), (0.5, 1)]
    此时的正样本数量为2，负样本数量为1，因此正负样本分数相同的配对数为1x2，即正样本数量x负样本数量
    因此，只要找到同分数下的正样本/负样本数量即可
    
    时间复杂度：O(nlogn)
    空间复杂度：O(n)
    """
    data = sorted(zip(y_pred, y_true)) # 按照第一个分数升序排序
    pos = sum(y_true) # 正样本的数量
    neg = len(y_true)-pos # 负样本数量

    # 找到正样本的排名总和
    rank_sum = 0 
    for i in range(len(data)):
        if data[i][1]==1: # 正样本
                rank_sum+=i+1 # 正样本的排名

    # 找到正样本和负样本分数相同的配对数
    equal_cnt = 0
    i = 0
    while i<len(data):
        j = i # 必须从i+1开始，避免遗漏当前的样本
        group_pos, group_neg = 0, 0 # 本组的正负样本数量
        while j<len(data) and data[j][0] == data[i][0]:  # 两个样本的分数相同
            if data[j][1]==1: # 该分数下的正样本数量
                group_pos+=1
            else:  # 该分数下的负样本数量
                group_neg+=1

        # 正样本和负样本分数相同的配对数为group_pos*group_neg
        equal_cnt+= group_pos*group_neg
        # 从当前遍历到的样本开始判断
        i = j
    
    wins = rank_sum-pos*(pos+1)/2 # 正样本分数【超过/等于】负样本分数的配对数
    total = pos*neg # 正样本和负样本的配对数
    auc = (wins-0.5*equal_cnt)/total
    return auc




# 回归问题
def mse(y_true, y_pred):
    return np.mean((y_true-y_pred)**2)

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true, y_pred))

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true-y_pred)**2))

