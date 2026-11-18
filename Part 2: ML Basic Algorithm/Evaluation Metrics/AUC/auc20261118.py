
# auc = p(score_pos>score_neg)+0.5p(score_pos==score_neg)
def auc(y_true, y_pred):
    """
    1. 找到正样本超过/等于负样本分数的pair数
    假设第n个正样本按照升序排序时的排名为m，因此前面有n-1个正样本，m-1个样本，m-n个负样本
    该正样本分数超过/等于负样本分数的pair数为 m-n
    所有正样本分数超过/等于负样本分数的pair数字为sum(m-n)
    - sum(m) = rank_sum
    - sum(n) = 1+...+p -> s=p(p+1)/2
    sum(m-n) = rank_sum-p(p+1)/2
    2. 找到正样本分数等于负样本的pair数
    假设[(0.5, 0), (0.5, 1), (0.5, 1)]
    正样本数量为2，负样本数量为1，因此pair数为2*1=2
    只要找到同分数下的正样本和负样本数量即可
    """
    pos_num = sum(y_true)
    neg_num = len(y_true)-pos_num
    data = sorted(zip(y_pred, y_true)) # 按照y_pred排序

    # 正样本的排名之和
    rank_sum = 0
    for i in range(len(data)):
        if data[i][1] == 1:
            rank_sum+=i+1

    # 正样本分数超过/等于负样本分数的配对数
    wins = rank_sum-(pos_num+1)*pos_num/2

    # 正样本分数等于负样本分数的配对数
    equal_cnt = 0
    i = 0
    # 遍历所有的样本，寻找该样本下分数相同的正负样本数量
    while i<len(data):
        group_pos = 0
        group_neg = 0
        j = i
        while j<len(data) and data[i][0]==data[j][0]:
            if data[j][1]==1:
                group_pos+=1
            else:
                group_neg+=1
            j+=1
        equal_cnt+=group_pos*group_neg
        i = j

        total = pos_num*neg_num
        return (wins-0.5*equal_cnt)/total