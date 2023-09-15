import torch.nn as nn

class GeneratorWassersteinSim(nn.Module):
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
    
    
class DiscriminatorWassersteinSim(nn.Module):
    def __init__(self):
        super().__init__()

        self.main = nn.Sequential(
            nn.Linear(1, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(True),
            nn.Linear(32, 1),
        )

    def forward(self, input):
        return self.main(input)