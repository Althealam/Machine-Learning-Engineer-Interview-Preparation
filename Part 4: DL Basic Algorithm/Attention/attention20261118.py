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

    def forward(self, x):
        """
        x: (batch_size, seq_len, d_model)
        """ 
        Q = self.Wq(x) # (batch_size, seq_len, dk)
        K = self.Wk(x) # (batch_size, seq_len, dk)
        V = self.Wv(x) # (batch_size, seq_len, dv)
        dk = Q.size(-1)

        scores = Q@K.transpose(1, 2) # (batch_size, seq_len, seq_len)
        scores = scores/math.sqrt(dk)

        attention_scores = torch.softmax(scores, dim=-1)

        output = attention_scores@V

        return output

d_model = 2
d_k = 3
d_v = 3
batch_size = 3
seq_len = 4
attn = Attention(d_model, d_k, d_v)
x = torch.randn(batch_size, seq_len, d_model)
res = attn(x)
print(res)