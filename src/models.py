import torch
import torch.nn as nn


# Generator Class
class Generator(nn.Module):

    def __init__(self, z_dim=128, img_channels=3, base=64):
        
        super().__init__()
        
        self.net = nn.Sequential(
            
            # z -> (base*16) x 4 x 4
            nn.ConvTranspose2d(z_dim, base*16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(base*16),
            nn.ReLU(True),

            # 4 -> 8
            nn.ConvTranspose2d(base*16, base*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base*8),
            nn.ReLU(True),

            # 8 -> 16
            nn.ConvTranspose2d(base*8, base*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base*4),
            nn.ReLU(True),

            # 16 -> 32
            nn.ConvTranspose2d(base*4, base*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base*2),
            nn.ReLU(True),

            # 32 -> 64
            nn.ConvTranspose2d(base*2, base, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base),
            nn.ReLU(True),

            # 64 -> 128
            nn.ConvTranspose2d(base, img_channels, 4, 2, 1, bias=False),
            nn.Tanh(),
        )
    
    def forward(self, z: torch.Tensor):
        
        z = z.view(z.size(0), z.size(1), 1, 1)
        return self.net(z)


class Discriminator(nn.Module):
    
    def __init__(self, img_channels = 3, base = 64):
        
        super().__init__()
        self.net = nn.Sequential(
            # 128 -> 64
            nn.Conv2d(img_channels, base, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 64 -> 32
            nn.Conv2d(base, base*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base*2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 32 -> 16
            nn.Conv2d(base*2, base*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base*4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 16 -> 8
            nn.Conv2d(base*4, base*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base*8),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 8 -> 4
            nn.Conv2d(base*8, base*16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base*16),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 4 -> 1 (logit)
            nn.Conv2d(base*16, 1, 4, 1, 0, bias=False),
        )
        
    
    def forward(self, x):
        out = self.net(x)
        return out.view(x.size(0))