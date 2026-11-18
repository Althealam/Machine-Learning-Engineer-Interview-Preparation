import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, num_heads):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.num_heads = num_heads
        self.Wq = nn.Linear(d_model, d_k*num_heads)
        self.Wk = nn.Linear(d_model, d_k*num_heads)
        self.Wv = nn.Linear(d_model, d_v*num_heads)

    def forward(self, x):
        """
        x: (batch_size, seq_len, d_model)
        """
        Q = self.Wq(x) # (batch_size, seq_len, d_k*num_heads)
        K = self.Wk(x) # (batch_size, seq_len, d_k*num_heads)
        V = self.Wv(x) # (batch_size, seq_len, d_v*num_heads)
        batch_size = Q.size(0)
        seq_len = Q.size(1)
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
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2) # (batch_size, num_heads, seq_len, dk)
        V = V.transpose(1, 2) 
        scores = Q@K.transpose(2, 3) # (batch_size, num_heads, seq_len, seq_len)

        scores = scores/self.d_k**0.5

        attention_scores = torch.softmax(scores, dim=-1)

        output = attention_scores@V # (batch_size, num_heads, seq_len, dv)
        output = output.transpose(1,2) # (batch_size, seq_len, num_heads, dv)
        output = output.contiguous().view(
            batch_size,
            seq_len, 
            self.num_heads*self.d_v
        )
        return output

