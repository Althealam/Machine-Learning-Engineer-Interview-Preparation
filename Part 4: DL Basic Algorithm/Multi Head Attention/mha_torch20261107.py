import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, num_heads):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.num_heads = num_heads

        self.W_q = nn.Linear(d_model, num_heads*d_k)
        self.W_k = nn.Linear(d_model, num_heads*d_k)
        self.W_v = nn.Linear(d_model, num_heads*d_v)


    def forward(self, X):    
        """
        X: (batch_size, seq_len, d_model)
        """    
        Q = self.W_q(X) # (batch_size, seq_len, num_heads*d_k)
        K = self.W_k(X) # (batch_size, seq_len, num_heads*d_k)
        V = self.W_v(X) # (batch_size, seq_len, num_heads*d_v)
        batch_size = Q.size(0)
        seq_len = Q.size(1)

        # (batch_size, seq_len, num_heads, d_k)
        Q = Q.view(
            batch_size, 
            seq_len, 
            self.num_heads,
            self.d_k
        )
        # (batch_size, seq_len, num_heads, d_k)
        K = K.view(
            batch_size,
            seq_len, 
            self.num_heads, 
            self.d_k
        )
        # (batch_size, seq_len, num_heads, d_v)
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

        attention_scores = torch.softmax(scores, dim=-1) 

        output = attention_scores@V  # (batch_size, num_heads, seq_len, d_v)

        output = output.transpose(2, 1) # (batch_size, seq_len, num_heads, d_v)
        output = output.contiguous().view(
            batch_size, 
            seq_len, 
            self.d_v * self.num_heads
        )
        return output


d_model = 4
d_k = 2
d_v = 3
batch_size = 2
seq_len = 6
num_heads = 2
mha = MultiHeadAttention(d_model, d_k, d_v, num_heads)
X = torch.randn(batch_size, seq_len, d_model)
res = mha(X)
print(res)
