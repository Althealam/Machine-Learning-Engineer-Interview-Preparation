import torch
import torch.nn as nn
import math

class Attention(nn.Module):
    def __init__(self, d_model, dk, dv):
        super().__init__()
        self.d_model = d_model
        self.dk = dk
        self.dv = dv
        self.Wq = nn.Linear(d_model, dk)
        self.Wk = nn.Linear(d_model, dk)
        self.Wv = nn.Linear(d_model, dv)

    def forward(self, X):
        """
        input:
        - X: (B, l, d_model)
        """
        # 1. Q/k/V
        Q = self.Wq(X) # (B, l, dk)
        K = self.Wk(X) # (B, l, dk)
        V = self.Wv(X) # (B, l, dv)

        # 2. attention scores
        scores = Q@K.transpose(-2, -1) # (B, l, l)
        dk = Q.size(-1)
        scores/=math.sqrt(dk)
        attention_scores = torch.softmax(scores, dim=-1)

        # 3. weighted sum
        weighted_sum = attention_scores@V # (b, l, dv)

        return weighted_sum

seq_len = 4
d_model = 2
dk = 4
dv = 3
model = Attention(d_model, dk, dv)
X = torch.randn(seq_len, d_model)
output = model(X)
print(output)