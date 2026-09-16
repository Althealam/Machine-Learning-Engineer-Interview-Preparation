import math
import torch

def attention(query, key, value):
    """
    input:
        query: (b, query_len, d_k)
        key: (b, key_len, d_k)
        value: (b, key_len, d_v)
    output:
        outputs: (b, query_len, d_v)
        weights: (b, query_len, key_len)
    """

    d_k = query.size(-1)

    scores = torch.matmul(query, key.transpose(-2, -1))
    # scores: (b, query_len, key_len)
    scores = scores/math.sqrt(d_k)

    # 这里为啥是dim=-1
    # 表示沿着key的维度做softmax
    # 每个query对所有key的attention weight加起来为1
    weights = torch.softmax(scores, dim=-1)

    output = torch.matmul(weights, value)

    return output, weights

batch_size = 2
seq_len = 4
d_k = 8
d_v = 6

query = torch.randn(batch_size, seq_len, d_k)
key = torch.randn(batch_size, seq_len, d_k)
value = torch.randn(batch_size, seq_len, d_v)

print(attention(query, key, value))