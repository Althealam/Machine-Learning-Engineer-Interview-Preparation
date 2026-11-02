
def ReLU(x):
    if x>0:
        return x
    return 0

class MLP:
    def __init__(self, input_dim, hidden_dim=64, output_dim=32):
        self.linear1 = Linear(input_dim, hidden_dim)
        self.linear2 = Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.linear1.forward(x)
        x = ReLU(x)
        x = self.linear2.forward(x)
        return x