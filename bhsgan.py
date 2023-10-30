import torch.nn as nn

from utils import Positive, TanhScale, ReLUn


class GeneratorBhsSim(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(True),
            nn.Linear(16 , 8),
            nn.ReLU(True),
            nn.Linear(8 , 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)
    
    
class DiscriminatorBhsSim(nn.Module):
    def __init__(self):
        super().__init__()

        self.main = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(True),
            nn.Linear(16 , 8),
            nn.ReLU(True),
            nn.Linear(8 , 1),
            nn.Softplus()
        )

    def forward(self, input):
        return self.main(input)
    

class GeneratorBhsMnist(nn.Module):

    def __init__(self, z_dim=100, output_dim=28*28, hidden_dim=512):
        super().__init__()
        
        self.z_dim = z_dim
        
        self.main = nn.Sequential(
            
            self.get_generator_block(z_dim, 
                                     hidden_dim
                                     ),
            
            self.get_generator_block(hidden_dim, 
                                     hidden_dim * 2,
                                     ),
        
            self.get_generator_final_block(hidden_dim * 2,
                                           output_dim,
                                           )
            

        )
        
        
    def get_generator_block(self, input_dim, output_dim):
        return nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.BatchNorm1d(output_dim),
                nn.ReLU(inplace=True)
        )
    
    
    def get_generator_final_block(self, input_dim, output_dim):
        return  nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.Sigmoid()
            )
    
    
    def forward(self, x):
        return self.main(x)
    
    
class DiscriminatorBhsMnist(nn.Module):

    def __init__(self, input_dim=1, hidden_dim=784):
        super().__init__()
        self.main = nn.Sequential(
            self.get_critic_block(input_dim,
                                         hidden_dim,
                                         ),
            
            self.get_critic_block(hidden_dim,
                                         hidden_dim // 2,
                                         ),
            self.get_critic_block(hidden_dim // 2,
                                         hidden_dim // 4,
                                         ),
            self.get_critic_final_block(hidden_dim // 4,
                                               1,
                                            ),

        )

        
    def get_critic_block(self, input_dim, output_dim):
        return nn.Sequential(
                nn.Linear(input_dim, output_dim),
                #nn.BatchNorm1d(output_dim),
                #nn.ELU(inplace=True)
                #TanhScale()
                #nn.LeakyReLU(inplace=True)

        )
    
    
    def get_critic_final_block(self, input_dim, output_dim):
        return  nn.Sequential(
                nn.Linear(input_dim, output_dim),
                Positive()
            )
    
    def forward(self, image):
        return self.main(image)
    