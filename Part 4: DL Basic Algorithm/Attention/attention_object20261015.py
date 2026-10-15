import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, d_model, dk, dv):
        super(Attention, self).__init__()
        self.d_model = d_model
        self.dk = dk
        self.dv = dv

        self.Wq = nn.Linear(d_model, dk)
        self.Wk = nn.Linear(d_model, dk)
        self.Wv = nn.Linear(d_model, dv)

    def forward(self, X):
        """
        X: (sequence_length, hidden_dimension) (n, d)
        """
        # 1. Q/K/V
        Q = self.Wq(X)
        K = self.Wk(X)
        V = self.Wv(X)

        # 2. attention score
        score = torch.matmul(Q, K.transpose(-2, -1))

        # 3. scale by sqrt(dk)
        score = score/(self.dk**0.5)

        # 4. softmax
        attention_weights = F.softmax(score, dim=-1)

        # 5. weighted sum with v
        output = torch.matmul(attention_weights, V)

        return output


d_model = 8
seq_len = 4
model = Attention(d_model, d_model, d_model)
X = torch.randn(seq_len, d_model)

output = model(X)
print(output)