import torch
import torch.nn as nn


# conv layers based on alpha zero architecture
# https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
class ppo_network(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(18, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
        )
        
        self.policy_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 2048),
            nn.ReLU(),
            nn.Linear(2048, 4096)
        )
        
        self.value_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1),
            nn.Tanh()
        )
    
    def forward(self, x):
        x = self.conv(x)
        policy = self.policy_head(x)
        value = self.value_head(x)
        return policy, value
