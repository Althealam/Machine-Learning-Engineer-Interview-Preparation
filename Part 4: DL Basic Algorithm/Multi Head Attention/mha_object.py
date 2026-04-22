import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, dmodel, numheads):
        """
        dmodel: hidden dimension
        numheads: number of attention heads
        """
        super().__init__()
        self.dmodel = dmodel
        self.numheads = numheads

        if self.dmodel%self.numheads!=0:
            print("dmodel must by divisible by numheads")
            raise ValueError
    
        self.dk = dmodel//numheads
        self.dv = dmodel//numheads

        self.Wq = nn.Linear(dmodel, dmodel)
        self.Wk = nn.Linear(dmodel, dmodel)
        self.Wv = nn.Linear(dmodel, dmodel)
        self.Wo = nn.Linear(dmodel, dmodel)

    def forward(self, X):
        n = X.shape[0] # sequence length

        # 1. Q/K/V
        Q = self.Wq(X) # (n, dmodel)
        K = self.Wk(X) # (n, dmodel)
        V = self.Wv(X) # (n, dmodel)

        # 2. split heads
        # (n, dmodel) -> (n, numheads, dk) -> (numheads, n, dk)
        Q = Q.view(n, self.numheads, self.dk).transpose(0, 1) # (numheads, n, dk)
        K = K.view(n, self.numheads, self.dk).transpose(0, 1) # (numheads, n, dk)
        V = V.view(n, self.numheads, self.dk).transpose(0, 1) # (numheads, n, dv)

        # 3. compute attention scores
        score = torch.matmul(Q, K.transpose(-2, -1)) # (numheads, n, n)

        # 4. scale
        score = score/(self.dk**0.5)

        # 5. softmax
        attention_weights = F.softmax(score, dim=-1) # (numheads, n, n)

        # 6. output
        head_output = torch.matmul(attention_weights, V) # (numheads, n, dv)

        # 7. concat all heads
        concat_output = head_output.transpose(0, 1).contiguous().view(n, self.dmodel) # (n, dmodel)

        # 8. linear projection
        output = self.Wo(concat_output) # (n, dmodel)

        return output


if __name__=="__main__":
    dmodel = 8
    seqlen = 4
    numheads = 2

    model = MultiHeadAttention(dmodel, numheads)
    X = torch.randn(seqlen, dmodel)

    output = model(X)
    print(X.shape)
    print(X)

    print(output)
    print(output.shape)