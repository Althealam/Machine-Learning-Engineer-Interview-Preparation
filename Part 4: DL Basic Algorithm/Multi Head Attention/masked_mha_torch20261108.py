import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, d_k, d_v):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.Wq = nn.Linear(d_model, num_heads*d_k)
        self.Wk = nn.Linear(d_model, num_heads*d_k)
        self.Wv = nn.Linear(d_model, num_heads*d_v)

    def forward(self, X):
        """
        X: (batch_size, seq_len, d_model)
        """
        batch_size = X.size(0)
        seq_len = X.size(1)
        Q = self.Wq(X) # (batch_size, seq_len, num_heads*dk)
        K = self.Wk(X) # (batch_size, seq_len, num_heads*dk)
        V = self.Wv(X) # (batch_size, seq_len, num_heads*dv)

        Q = Q.view(
            batch_size, 
            seq_len,
            self.num_heads, 
            self.d_k
        )
        K = K.view(
            batch_size, 
            seq_len, 
            self.num_heads, 
            self.d_k
        )
        V = V.view(
            batch_size, 
            seq_len, 
            self.num_heads, 
            self.d_v
        )
        Q = Q.transpose(1, 2) # (batch_size, num_heads, seq_len, d_k)
        K = K.transpose(1, 2) # (batch_size, num_heads, seq_len, d_k)
        V = V.transpose(1, 2) # (batch_size, num_heads, seq_len, d_v)

        scores = Q@K.transpose(-2, -1) # (batch_size, num_heads, seq_len, seq_len)
        scores = scores/self.d_k**0.5

        # torch.ones用来生成全1矩阵，torch.tril用于生成下三角矩阵
        causal_mask = torch.tril(torch.ones(
            seq_len, seq_len, dtype=torch.bool
        )) # 下三角矩阵，并且全部为1
        # 非下三角矩阵，全部设置为float('-inf') -> float('-inf')时softmax为0
        # mask矩阵：当j<=i时mask=0；当j>i时mask=float('-inf')，未来token的attention权重为0
        scores = scores.masked_fill(~causal_mask, float("-inf"))
        # tensor.masked_fill(mask, value) 其中mask为bool，true表示需要替换；value表示被替换为什么值
        attention_scores = torch.softmax(scores, dim=-1) # (batch_size, num_heads, seq_len, seq_len)

        output = attention_scores@V # (batch_size, num_heads, seq_len, d_v)
        output = output.transpose(2, 1) # (batch_size, seq_len, num_heads, d_v)
        output = output.contiguous().view(
            batch_size, 
            seq_len, 
            self.num_heads*self.d_v
        )
        return output

batch_size = 2
seq_len = 3
d_k = 4
d_v = 2
num_heads = 4
d_model = 2
attn = MultiHeadAttention(d_model, num_heads, d_k, d_v)
X = torch.randn(batch_size, seq_len, d_model)
print(X.shape)
print(attn(X))