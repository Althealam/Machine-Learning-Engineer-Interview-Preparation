import torch
import torch.nn.functional as F

def self_attention(X):
    """
    X: (n, d) sequence_length n, hidden dim d
    """
    n, d = X.shape

    d_k = d
    d_v = d-1
    # 1. define (W_Q, W_K, W_V)
    W_Q = torch.randn(d, d_k)
    W_K = torch.randn(d, d_k)
    W_V = torch.randn(d, d_v)

    # 2. compute Q, K, V
    Q = X@W_Q # (n, dk)
    K = X@W_K # (n, dk)
    V = X@W_V # (n, dv)

    # 3. compute attention socres
    scores = Q@K.T # (n, n)

    # 4. scaling
    scores = scores / (d_k**0.5)

    # 5. softmax by row
    A = F.softmax(scores, dim=-1) # (n, n)

    # 6. weighted sum
    output = A@V # (n, dv)

    return output

