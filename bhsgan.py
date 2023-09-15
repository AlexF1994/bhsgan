import torch.nn as nn


class GeneratorBhsSim(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(1, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(True),
            nn.Linear(32 , 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)
    
    
class DiscriminatorBhsSim(nn.Module):
    def __init__(self):
        super().__init__()

        self.main = nn.Sequential(
            nn.Linear(1, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(True),
            nn.Linear(32, 1),
            nn.Softplus()
        )

    def forward(self, input):
        return self.main(input)
    
