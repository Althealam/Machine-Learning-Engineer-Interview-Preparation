import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, dmodel, dk, dv):
        super(Attention, self).__init__()

        self.dmodel = dmodel
        self.dk = dk
        self.dv = dv

        self.Wq = nn.Linear(dmodel, dk)
        self.Wk = nn.Linear(dmodel, dk)
        self.Wv = nn.Linear(dmodel, dv)

        self.Wo = nn.Linear(dv, dmodel)

    def forward(self, X):
        """
        X: (sequence length n, hidden dimension dmodel)
        """
        # 1. Q/K/V
        Q = self.Wq(X) # (n, dk)
        K = self.Wk(X) # (n, dk)
        V = self.Wv(X) # (n, dv)

        # 2. QK.T
        # (n, n)
        scores = torch.matmul(Q, K.transpose(-2, -1))

        # 3. scale by dk**0.5
        scores = scores/self.dk**0.5

        # 4. softmax by row dimension
        # (n, n)
        attention_scores = F.softmax(scores, dim=-1)

        # 5. multiply by V
        output = torch.matmul(attention_scores, V) # (n, dv)

        # 6. linear projection to dmodel
        output = self.Wo(output)

        return output
    
if __name__ == "__main__":
    n = 8
    dmodel = 4
    X = torch.randn(n, dmodel)
    model = Attention(dmodel=4, dk=3, dv=5)
    output = model(X)
    print(X)
    print(X.shape)