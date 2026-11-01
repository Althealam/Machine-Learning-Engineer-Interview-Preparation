def auc1(y_true, y_pred):
    """
    auc = p(score_pos>score_neg)+0.5p(score_pos=score_neg)
    时间复杂度：O(pm)=O(p(n-p))，最坏情况下为O(n^2)
    空间复杂度：O(p+m)=O(n)
    """
    pos_scores = []
    neg_scores = []
    for i in range(len(y_true)): # O(n)
        if y_true[i]==1:
            pos_scores.append(y_pred[i])
        else:
            neg_scores.append(y_pred[i])

    correct = 0
    for pos in pos_scores: # O(p)
        for neg in neg_scores: # O(m)
            if pos>neg:
                correct+=1
            elif pos==neg:
                correct+=0.5
    if len(pos_scores)*len(neg_scores)==0:
        return 0
    return correct/(len(pos_scores)*len(neg_scores))

def auc2(y_true, y_pred):
    """
    找到正样本分数「超过/等于」负样本分数的配对数
    1. 找到正样本分数「超过/等于」负样本的配对数
    假设第n个正样本按照分数升序排序时的排名是m，因此前面有n-1个正样本，m-1个样本，m-n个负样本
    该正样本分数超过/等于负样本数量的配对数为m-n
    所有正样本分数超过/等于负样本数量的配对数为sum(m-n)
    - sum(m)=rank_sum
    - sum(n)=1+...+p -> s=1+...+p = p+..+1 -> 2s = p(p+1) -> s=p(p+1)/2
    sum(m-n) = rank_sum-p(p+1)/2，其中p是所有正样本的数量
    2. 找到正样本分数「等于」负样本的配对数
    假设[(0.5, 0), (0.5, 1), (0.5, 1)]
    此时正样本的数量为2，负样本的数量为1，那么正样本分数等于负样本分数的配对数为2*1=2
    因此只要找到同分数下的正负样本数量即可找到配对数

    时间复杂度: O(nlogn)
    空间复杂度：O(n)
    """
    pos_num = sum(y_true)
    neg_num = len(y_true)-pos_num
    data = sorted(zip(y_pred, y_true)) # O(nlogn)

    # 正样本的排名之和
    rank_sum = 0 
    for i in range(len(data)): # O(n)
        if data[i][1]==1:
            rank_sum+=i+1

    # 正样本分数超过/等于负样本分数的配对数
    wins = rank_sum-(pos_num+1)*pos_num/2

    # 正样本分数等于负样本分数的配对数
    equal_cnt = 0
    i = 0
    # 遍历所有的样本，寻找该样本下分数相同的正负样本数量
    while i<len(data): # O(n)
        j = i # 【重要】
        group_pos = 0
        group_neg = 0
        while j<len(data) and data[i][0]==data[j][0]:
            if data[j][1]==1:
                group_pos+=1
            else:
                group_neg+=1
            j+=1 # 遍历下一个样本
        equal_cnt+=group_pos*group_neg
        i = j # 【重要】

    total = pos_num*neg_num # 所有的配对数
    if pos_num*neg_num==0:
        return 0
    return (wins-0.5*equal_cnt)/total


y_true = [1, 0, 0, 1, 0]
y_pred = [0.4, 0.2, 0.3, 0.4, 0.8]
print(auc1(y_true, y_pred))