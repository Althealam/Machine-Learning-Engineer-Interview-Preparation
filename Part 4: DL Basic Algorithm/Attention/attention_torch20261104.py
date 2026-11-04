import torch
import math
def attention(query, key, value):
    """
    input: 
    - query: (b, l, dk)
    - key: (b, l, dk)
    - value: (b, l, dv)
    """
    scores = query@key.transpose(-2, -1) # (b, l, l) (2, 4, 4)
    dk = query.size(-1)
    scores = scores/math.sqrt(dk)
    attention_scores = torch.softmax(scores, dim=-1) # (b, l, l)
    output = attention_scores@value # (b, l, l)@(b, l, dv)
    return output

batch_size = 2
l = 4
dk = dv = 4
query = torch.randn(batch_size, l, dk) # (2, 4, 4)
key = torch.randn(batch_size, l, dk) # (2, 4, 4)
value = torch.randn(batch_size, l, dv)
print(attention(query, key, value).size())
# print(query.size())
# print(key.size())
# print(key.T.size())

# 注意事项：
# 1. key的transpose不可以直接用key.T
# 2. 在用softmax的时候要指明dim，也就是按照哪一个维度去做归一化