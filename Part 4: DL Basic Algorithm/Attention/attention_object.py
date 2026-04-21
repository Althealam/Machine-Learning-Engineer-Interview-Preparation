import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, d_model, dk, dv):
        """
        d_model: hidden dimension
        dk: key matrix dimension
        dv: value matrix dimension
        """
        super(Attention, self).__init__()
        self.d_model = d_model
        self.dk = dk
        self.dv = dv

        self.Wq = nn.Linear(d_model, dk)
        self.Wk = nn.Linear(d_model, dk)
        self.Wv = nn.Linear(d_model, dv)

    
    def forward(self, X):
        """
        X: (sequence length, hidden dimension) (n, d)
        """
        # 1. Q, K, V
        Q = self.Wq(X) #(n, dk)
        K = self.Wk(X) #(n, dk)
        V = self.Wv(X) #(n, dv)

        # 2. Attention scores
        score = torch.matmul(Q, K.transpose(-2, -1)) # (n, n)

        # 3. Scale by sqrt(dk)
        score = score/(self.dk**0.5)

        # 4. Softmax
        attention_weights = F.softmax(score, dim=-1) # (n, n)

        # 5. Weighted sum with V
        output = torch.matmul(attention_weights, V) # (n, dv)

        return output
    
if __name__=="__main__":
    d_model = 8
    seq_len = 4

    model = Attention(d_model, d_model, d_model)
    X = torch.randn(seq_len, d_model)

    output = model(X)
    print(X)
    print(X.shape)


