import torch
import torch.nn.functional as F

def multi_head_attention(X, num_heads):
    n, d = X.shape
    # n = sequence length
    # d = hidden dimension

    if d%num_heads!=0:
        print("d must by divisible by num_heads")
        raise ValueError 

    dk = d//num_heads
    dv = d//num_heads

    # 1. randomly init weight matricsc
    Wq = torch.randn(d, d)
    Wk = torch.randn(d, d)
    Wv = torch.randn(d, d)
    Wo = torch.randn(d, d)

    # 2. Q/K/V
    Q = X@Wq # (n, d)
    K = X@Wk # (n, d)
    V = X@Wv # (n, d)
    
    # 3. split into multiple heads
    # reshape: (n, d) -> (n, num_heads, dk)
    Q = Q.reshape(n, num_heads, dk).transpose(0, 1) # (num_heads, n, dk)
    K = K.reshape(n, num_heads, dk).transpose(0, 1) # (num_heads, n, dk)
    V = V.reshape(n, num_heads, dv).transpose(0, 1) # (num_heads, n, dv)

    # 4. QK^T for each head
    score = Q@K.transpose(-2, -1) # (num_heads, n, n)

    # 5. scale by dk**0.5
    score = score/(dk**0.5) # (num_heads, n, n)

    # 6. softmax
    attention_scores = F.softmax(score, dim=-1) # (num_heads, n, n)

    # 7. scale with V
    head_outputs = attention_scores@V # (num_heads, n, dv)

    # 8. concat all heads
    # (num_heads, n, dv) -> (n, num_heads, dv) -> (n, d)
    concat_output = head_outputs.transpose(0, 1).reshape(n, d)

    # 9. final linear projection
    res = concat_output@Wo # (n, d)

    return res

if __name__ == "__main__":
    n, d = 2, 4
    num_heads = 2
    X = torch.randn(n, d)

    output = multi_head_attention(X, num_heads)
    print(X)
    print(X.shape)
    print(output.shape)
    print(output)