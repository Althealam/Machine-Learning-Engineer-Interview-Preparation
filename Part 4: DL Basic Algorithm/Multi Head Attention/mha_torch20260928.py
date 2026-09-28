import torch
import torch.nn as nn
import torch.nn.functional as F

# 总体流程
# 1. 通过W_q, W_k, W_v得到映射后的Q/K/V
# 2. 分头，将Q/K/V按照num_head分头
# 3. 计算attention scores，将每个头的[seq_len, head_dim]@[head_dim, seq_len]，最终得到[seq_len, seq_len]的矩阵，第一维为Q，第二维为K
# 4. attention scores除以head_dim的根号
# 5. 计算attention scores的softmax值，即获得对每个query，在所有key上计算概率（对每个query，所有key相加之和为1）
# 6. 计算V的attention weights的加权和，即attention weights*V，此时维度为[batch_size, num_heads, seq_len, head_dim]
# 7. 转换一下维度，变成[batch_size, seq_len, num_heads, head_dim]，然后将最后两维拼接在一起得到[batch_size, seq_len, d_model] （前面一定要确保d_model==num_heads*head_dim）

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model%num_heads==0

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model//num_heads

        # Q K V linear projection
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # final output projection
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        """
        - input:
            x: [batch_size, seq_len, d_model]
        mask:
            [batch_size, 1, 1, seq_len]
        """
        batch_size, seq_len, _ = x.shape

        # 1. Linear projection
        # [B, L, d_model]
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # 2. Split heads
        # [B, L, d_model] -> [B, L, num_heads, head_dim] -> [B, num_heads, L, head_dim] 
        Q = Q.view(
            batch_size,
            seq_len, 
            self.num_heads, 
            self.head_dim
        ).transpose(1, 2)
        K = K.view(
            batch_size, 
            seq_len, 
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)
        V = V.view(
            batch_size, 
            seq_len, 
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)

        # 3. Scaled Dot-product attention
        scores = torch.matmul(
            Q,  # [B, num_heads, L, head_dim]
            K.transpose(-2, -1) # [B, num_heads, head_dim, L]
        )
        # scores: [B, num_heads, L, L]
        scores = scores/(self.head_dim**0.5)

        # 4. softmax
        # 对每个query，在所有key上计算概率
        attention_weights = F.softmax(scores, dim=-1)

        # 5. weighted sum of V
        # [B, H, L, L] @ [B, H, L, D] -> [B, H, L, D]
        output = torch.matmul(attention_weights, V)

        # 6. merge heads
        # [B, H, L, D] -> [B, L, H, D]
        output = output.transpose(1, 2)

        output = output.contiguous().view(
            batch_size, 
            seq_len, 
            self.d_model
        )

        # 7. final linear
        output = self.W_o(output)
        return output
