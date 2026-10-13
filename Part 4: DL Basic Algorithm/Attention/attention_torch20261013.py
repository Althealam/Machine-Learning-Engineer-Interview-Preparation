import torch
import math

def attention(query, key, value):
    """
    input:
        query: (batch_size, query_len, d_k)
        key: (batch_size, key_len, d_k)
        value: (batch_size, key_len, d_v)
    """
    d_k = query.size(-1)

    # scores = QK^T
    # query: (b, lq, dk)
    # key: (b, dk, lk)
    # scores: (b, lq, lk)
    scores = torch.matmul(query, key.transpose(-2, -1))
    scores = scores/math.sqrt(d_k)

    # softmax
    weights = torch.softmax(scores, dim=-1)
    # 每一行的相加之和为1

    # weight: (b, lq, lk)
    # value: (b, lk, dv)
    # output: (b, lq, dv)
    output = torch.matmul(weights, value)

    return output, weights

batch_size = 2
seq_len_k = 4
seq_len_v = 6
d_k = 8
d_v = 6

query = torch.randn(batch_size, seq_len_k, d_k)
key = torch.randn(batch_size, seq_len_v, d_k)
value = torch.randn(batch_size, seq_len_v, d_v)
print(attention(query, key, value))