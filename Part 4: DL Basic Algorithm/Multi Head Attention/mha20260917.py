import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()

        assert d_model%num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model//num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        self.W_o = nn.Linear(d_model, d_model)
    
    def forward(self, x, mask=None):
        """
        x: [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.shape

        # 1. 计算Q/K/V
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        # Q/K/V shape: [batch_size, seq_len, d_model]

        # 2. 拆分成多个head
        Q = Q.view(
            batch_size, 
            seq_len, 
            self.num_heads, 
            self.head_dim
        )
        K = K.view(
            batch_size, 
            seq_len, 
            self.num_heads,
            self.head_dim
        )
        V = V.view(
            batch_size, 
            seq_len, 
            self.num_heads,
            self.head_dim
        )

        # 更换维度
        Q = Q.transpose(1, 2) # [batch_size, num_heads, seq_len, head_dim]
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # 3. compute attention score
        scores = torch.matmul(
            Q, 
            K.transpose(-2, -1) # [batch_size, num_heads, head_dim, seq_len]
        )
        scores = scores/(self.head_dim**0.5) # [batch_size, num_heads, seq_len, seq_len]

        # 4. compute softmax
        attention_weights = torch.softmax(
            scores, 
            dim = -1 # 按照最后一列来计算，因此对某个query，其在所有key上的总和为1
        )
        
        # 5. attention_weight * V
        output = torch.matmul(
            attention_weights,
            V
        )
        # [batch_size, num_heads, seq_len, head_dim]
        output = output.transpose(1, 2)
        # [batch_size, seq_len, num_heads, head_dim]

        # 6. 还原形状
        output = output.contiguous().view(
            batch_size, 
            seq_len, 
            self.d_model
        )

        output = self.W_o(output)
        return output