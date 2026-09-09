y_true = [0, 1, 1, 0]
y_score = [0.2, 0.6, 0.3, 0.4]

def auc1(y_true, y_score):
    # 时间复杂度：O(n)
    # 空间复杂度：O(n)
    pos_score = []
    neg_score = []

    for i in range(len(y_score)):
        if y_true[i]==1:
            pos_score.append(y_score[i])
        else:
            neg_score.append(y_score[i])
    
    total = 0
    correct = 0
    for i in range(len(pos_score)):
        for j in range(len(neg_score)):
            total+=1
            if pos_score[i]>neg_score[j]:
                correct+=1
            elif pos_score[i]==neg_score[j]:
                correct+=0.5
    return correct/total

print(auc1(y_true, y_score))

# 假设第k个正样本的排名是rk，那么前面有k-1个正样本，rk-1个样本，rk-k个负样本
# 由此可以知道第k个正样本超过的负样本数为rk-k
# 正样本超过负样本数之和为sum(rk-k)=rank_sum-sum(k)，其中sum(k)=P(P+1)/2，P是正样本的数量

def auc2(y_true, y_score):
    data = sorted(zip(y_score, y_true)) # 按照y_score从小到大排序
    pos = sum(y_true)
    neg = len(y_true)-pos
    rank_sum = 0
    for i in range(len(data)):
        if data[i][1]==1:
            rank_sum+=i+1
    total = pos*neg
    auc = (rank_sum-pos*(pos+1)/2)/total
    return auc