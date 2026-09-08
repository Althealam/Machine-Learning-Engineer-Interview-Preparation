y_true = [0, 1, 1, 0]
y_score = [0.1, 0.8, 0.7, 0.3]

# 方法一：暴力法
# AUC = (correct+0.5*equal)/pos*neg（这个公式可以通过ROC曲线下面积的积分推导得到）
# 其中pos是正样本数量，neg是负样本数量
# correct是正样本分数高于负样本的样本对数，equal是正样本分数和负样本分数相同的样本对数
# 概率解释：随机选择一个正样本和一个负样本，模型给正样本的分数高于负样本分数的概率

def auc1(y_true, y_score):
    # AUC = P(score_pos>score_neg)+0.5P(score_pos=score_neg)
    # 其中P(score_pos>score_neg) = count(score_pos>score_neg)/count
    # P(score_pos=score_neg) = count(score_pos=score_neg)/count

    # 时间复杂度：O(P*N)，假设正样本数量有P个，负样本数量有N个
    # 空间复杂度：O(n) 
    pos_scores = []
    neg_scores = []

    # 分离正负样本的预测分数
    for i in range(len(y_true)):
        if y_true[i]==1:
            pos_scores.append(y_score[i])
        else:
            neg_scores.append(y_score[i])
    
    total = 0
    numerator_large = 0
    numerator_equal = 0
    for pos in pos_scores:
        for neg in neg_scores:
            total+=1
            if pos>neg:
                numerator_large+=1
            elif pos==neg:
                numerator_equal+=1
    return numerator_large/total + 0.5*numerator_equal/total

    # # 正负样本两两比较
    # for pos in pos_scores:
    #     for neg in neg_scores:
    #         total+=1
    #         if pos>neg:
    #             correct+=1
    #         elif pos==neg:
    #             correct+=0.5
    # if total==0:
    #     return 0
    # return correct/total


# 方法二：使用正样本的排名去统计超过多少负样本来替换auc的分子的计算公式为rank_sum-P(P+1)/2（正确排序的正负样本对数）

# 假设第k个正样本的排名是rk，那么其前面有k-1个正样本，rk-1个样本，rk-k个负样本
# 由此可以知道第k个正样本超过的负样本数为rk-k
# 正样本超过的负样本数之和为sum(rk-k)（对所有的正样本去计算一下，假设共有P个正样本）
# 根据等比求和的公式，可以知道sum(k) = P(P+1)/2，其中P是加的次数
# 由此sum(rk)是rank_sum，即所有正样本的排名之和；sum(k)=pos*(pos+1)/2

def auc2(y_true, y_score):
    # 不用逐个比较的情况下，通过排名直接算出有多少个负样本在正样本的后面
    # 时间复杂度：O(nlogn)，排序的时间复杂度为O(nlogn)
    # 空间复杂度：O(n)
    data = sorted(zip(y_score, y_true)) # sorted会按照第一个元素从小到大排序，zip会将两个元素绑定在一起
    # 如果想按照第二个元素来排序：data = sorted(zip(y_score, y_true), key = lambda x: x[1])
    # 按照第一个元素来排序：data = sorted(zip(y_score, y_true))
    pos = sum(y_true)
    neg = len(y_true)-pos
    rank_sum = 0
    for i in range(len(data)):
        if data[i][1]==1:
            rank_sum+=i+1 # 正样本的排名
    total = pos*neg
    # rank_sum-pos*(pos+1)/2相当于所有正样本超过了多少个负样本
    auc = (rank_sum-pos*(pos+1)/2)/total
    return auc