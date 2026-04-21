import torch
import torch.nn.functional as F

def attention(X):
    n, d = X.shape
    # n: sequence length
    # d: hidden dimension

    dk, dv = d, d
    # 0. randomly init weight matrics
    Wq = torch.randn(d, dk)
    Wk = torch.randn(d, dk)
    Wv = torch.randn(d, dv)

    # 1. Q/K/V
    Q = X@Wq # (n, dk)
    K = X@Wk # (n, dk)
    V = X@Wv # (n, dv)
    
    # 2. QK^T
    score = Q@K.T  # (n, n)

    # 3. scale by dk**0.5
    score = score/dk**0.5 # (n, n)

    # 4. softmax
    attention_scores = F.softmax(score, dim=-1) # (n, n)

    # 5. scale with V
    res = attention_scores@V # (n, dv)

    return res

if __name__ == "__main__":
    n, d = 4, 8 # 4 tokens, each dimension is 8
    X = torch.randn(n, d)

    output = attention(X)
    print(X)
    print(X.shape)

