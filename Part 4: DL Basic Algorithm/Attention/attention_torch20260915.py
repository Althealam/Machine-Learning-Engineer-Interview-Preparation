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

    # QK^T/sqrt(d_k)
    scores = torch.matmul(query, key.transpose(-2, -1))
    scores = scores/math.sqrt(d_k)

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

print(query)