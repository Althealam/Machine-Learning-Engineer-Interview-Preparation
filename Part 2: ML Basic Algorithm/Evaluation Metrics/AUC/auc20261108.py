def auc(y_true, y_pred):
    pos_num = sum(y_true)
    neg_num = len(y_true)-pos_num
    data = sorted(zip(y_pred, y_true))

    # 正样本的排名之和
    rank_sum = 0
    for i in range(len(data)):
        if data[i][1]==1:
            rank_sum+=i+1

    # 正样本分数超过/等于负样本分数的配对数
    wins = rank_sum-(pos_num+1)*pos_num/2

    # 正样本分数等于负样本分数的配对数
    equal_cnt = 0
    i = 0
    while i<len(data):
        j = i
        group_pos = 0
        group_neg = 0
        while j<len(data) and data[i][0]==data[j][0]:
            if data[j][1]==1:
                group_pos+=1
            else:
                group_neg+=1
            j+=1
        equal_cnt+=group_pos*group_neg
        i = j

    total = pos_num*neg_num
    if pos_num*neg_num==0:
        return 0
    return (wins-0.5*equal_cnt)/total
