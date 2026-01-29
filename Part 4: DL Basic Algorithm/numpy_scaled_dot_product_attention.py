import numpy as np
import math


# == generate data ==
batch_size = 2
seq_len = 2
d_model = 4

# d_k和d_v分别表示key和value的维度
d_k = 2
d_v = 3
# 表示总共有batch_size个样本，每个样本的序列长度为seq_len，每个token的维度为d_model
X = np.random.randn(batch_size, seq_len, d_model)
print("The shape of X:", X.shape)
print("The value of X:", X)

W_q = np.random.randn(d_model, d_k) # (4, 2)
W_k = np.random.randn(d_model, d_k) # (4, 2)
W_v = np.random.randn(d_model, d_v) # (4, 3)
print("The shape of W_q:", W_q.shape)
print("The shape of W_k:", W_k.shape)
print("The shape of W_v:", W_v.shape)

# == simplest causal mask ==
# shape: (L, L)
mask = np.tril(np.ones((seq_len, seq_len)))
print("Mask - 1:", mask)
# expand to batch: (B, L, L)
mask = np.broadcast_to(mask, (batch_size, seq_len, seq_len))
print("Mask - 2:", mask)

def dot_product_attention(W_q, W_k, W_v, mask):
    """
    W_q: (d_model, d_k)
    W_k: (d_model, d_k)
    W_v: (d_model, d_v)
    """
    # Step1: Q = X@W_q, K = X@W_k, V = X@W_v
    # Q: (B, L_q, d_k)
    # K: (B, L_k, d_k)
    # V: (B, L_k, d_v)
    Q = np.matmul(X, W_q) # (2, 2, 2) 
    K = np.matmul(X, W_k) # (2, 2, 2)
    V = np.matmul(X, W_v) # (2, 2, 3)
    print("The shape of Q:", Q.shape)
    print("The shape of K:", K.shape)
    print("The shape of V:", V.shape)

    # Step2: QK^T/np.sqrt(d_k)
    d_k = Q.shape[-1]
    scores = np.matmul(Q, K.transpose(0, 2, 1))/np.sqrt(d_k)
    # Scores: (B, L_q, L_k)
    print("The shape of scores:", scores.shape)

    # Step3: mask
    if mask is not None:
        scores = np.where(mask==0, -1e9, scores)
    
    # Step4: softmax(QK^T/sqrt(d_k))
    # attn_weights: (B, L_q, L_k)
    exp_scores = np.exp(scores)
    attn_weights = exp_scores/np.sum(exp_scores, axis=-1, keepdims = True)
    print("The shape of attn_weights:", attn_weights.shape)

    # Step5: softmax(QK^T/sqrt(d_k))@V
    # attn_weights: (B, L_q, L_k)
    # V: (B, L_k, d_v)
    # 注意：V不需要做transpose，因为(L_q, L_k)@(L_k, d_v)最后的维度是(L_q, d_v)
    output = np.matmul(attn_weights, V)
    return output, attn_weights


output, attn_weights = dot_product_attention(W_q, W_k, W_v, mask)
print("The shape of output:", output)
