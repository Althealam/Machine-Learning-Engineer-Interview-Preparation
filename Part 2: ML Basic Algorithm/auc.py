y_true = [0, 1, 1, 0]
y_score = [0.1, 0.8, 0.7, 0.3]

def auc1(y_true, y_score):
    # AUC = P(score_pos>score_neg)+0.5P(score_pos=score_neg)
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
    correct = 0

    # 正负样本两两比较
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

def auc2(y_true, y_score):
    # 不用逐个比较的情况下，通过排名直接算出有多少个负样本在正样本的后面
    # 时间复杂度：O(nlogn)，排序的时间复杂度
    # 空间复杂度：O(n)
    data = sorted(zip(y_score, y_true)) # sorted会按照第一个元素从小到大排序，zip会将两个元素绑定在一起
    # 如果想按照第二个元素来排序：data = sorted(zip(y_score, y_true), key = lambda x: x[1])
    # 按照第一个元素来排序：data = sorted(zip(y_score, y_true))
    pos = sum(y_true)
    neg = len(y_true)-pos
    rank_sum = 0
    for i in range(len(data)):
        if data[i][1]==1:
            rank_sum+=i+1
    total = pos*neg
    # rank_sum-pos*(pos+1)/2相当于所有正样本超过了多少个负样本
    auc = (rank_sum-pos*(pos+1)/2)/total
    return auc