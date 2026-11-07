import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# == generate data ==
B, L, d_model = 2, 4, 8
num_heads = 2

x = torch.randn(B, L, d_model)
print("The shape of x is:", x.shape)
print("The value of x is:", x)



class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()

        assert d_model%num_heads == 0, "d_model must by divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model//num_heads # dimension of the key and value for each head

        # Linear Projection
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # Output projection
        self.W_o = nn.Linear(d_model, d_model)

    
    def forward(self, Q, K, V, mask = None):
        """
        Q: (B, Lq, d_model)
        K: (B, Lk, d_model)
        V: (B, Lv, d_model)
        # NOTE: Q, K, V的每个样本的token的维度是否一定都是一样的？都为d_model?
        mask: (B, Lq, Lk)
        """
        B = Q.size(0)
        
        # 1. Linear Projection
        Q = self.W_q(Q)
        K = self.W_k(K)
        V = self.W_v(V)

        # 2. Split into multiple heads
        Q = Q.view(B, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(B, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(B, -1, self.num_heads, self.d_k).transpose(1, 2)

        # 3. Scaled Dot-Product Attention
        scores = torch.matmul(Q, K.transpose(-2, -1))/math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1)==0, float('-inf'))
        
        attn_weights = torch.softmax(scores, dim=-1)

        attn_output = torch.matmul(attn_weights, V)

        # 4. Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(B, -1, self.d_model)

        # 5. Final linear projection
        output = self.W_o(attn_output)

        return output, attn_weights
    

mha = MultiHeadAttention(d_model, num_heads)
out, attn = mha(x, x, x)
print(out.shape)   # (2, 4, 8)
print(attn.shape)  # (2, 2, 4, 4)