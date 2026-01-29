import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# == generate data ==
batch_size = 1
seq_len = 2
d_model = 2 # hidden dimension
num_heads = 4

x = torch.randn(batch_size, seq_len, d_model)
print("The shape of x is:", x.shape)
print("The value of x is:", x)

# == generate Q, K, V ==
Q = x # (B, L, d)
K = x
V = x

# == scaled dot-product attention ==
def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    batch_size表示一个batch里有多少个样本（序列），seq_len表示每个样本里有多少个token，d_model表示每个token被表示成多少维的向量
    Q: (batch_size, seq_len_q, d_k)
    K: (batch_size, seq_len_k, d_k)
    V: (batch_size, seq_len_k, d_v)
    seq_len is the number of tokens in the sequence for query/key/value
    In the cross attention, seq_len_q is the number of tokens in the query sequence, and seq_len_k is the number of tokens in the key sequence.
    In the self attention, seq_len_q is the number of tokens in the sequence.
    Attention: seq_len for Key and Value is the same, but different from query. 
    Because we want to compute the attention between the query and the key, and the value is the information we want to attend to.
    mask: (batch_size, seq_len_q, seq_len_k) or None
    """
    d_k = Q.size(-1) # dimension of the key and value
    
    # K.transpose(-2, -1) is (B, d_k, L_k)
    # Q is (B, L_q, d_k)
    # scores is (B, L_q, L_k)
    scores = torch.matmul(Q, K.transpose(-2, -1))/math.sqrt(d_k)
    print("The shape of scores is:", scores.shape)

    if mask is not None:
        scores = scores.masked_fill(mask==0, float('-inf'))
    
    # attn_weights is (B, L_q, L_k)
    attn_weights = F.softmax(scores, dim=-1)
    print("The shape of attn_weights is:", attn_weights.shape)

    # V is (B, L_k, d_v)
    # otuput is (B, L_q, d_v)
    output = torch.matmul(attn_weights, V)
    print("The shape of output is:", output.shape)

    return output, attn_weights


output, attn_weights = scaled_dot_product_attention(Q, K, V)
print("The value of output is:", output)
print("The attention weights are:", attn_weights)