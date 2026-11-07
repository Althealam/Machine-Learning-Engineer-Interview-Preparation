import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, d_model, d_k, d_v):
        super().__init__()
        self.Wq = nn.Linear(d_model, d_k)
        self.Wk = nn.Linear(d_model, d_k)
        self.Wv = nn.Linear(d_model, d_v)

    def forward(self, X):
        """
        - X: (batch_size, seq_len, d_model)
        """
        Q = self.Wq(X) # (batch_size, seq_len, d_k)
        K = self.Wk(X) # (batch_size, seq_len, d_k)
        V = self.Wv(X) # (batch_size, seq_len, d_v)

        d_k = Q.size(-1)
        scores = Q@K.transpose(-2, -1) # (b, l, dk)@(d, dk, l) -> (batch_size, seq_len, seq_len)
        scores = scores/torch.sqrt(d_k)

        attention_scores = torch.softmax(scores, dim=-1)

        output = attention_scores@V # (batch_size, seq_len, d_v)
        return output



batch_size = 2
seq_len = 4
d_model = 4
d_k = 2
d_v = 3
attn = Attention(d_model, d_k, d_v) # 初始化的时候记得带上d_model, d_k, d_v

X = torch.randn(batch_size, seq_len, d_model)
print(X)