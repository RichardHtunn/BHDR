import torch
import torch.nn as nn
import torch.nn.functional as F

class BurmeseDigitCNN_V2(nn.Module):
    def __init__(self) -> None:
        super(BurmeseDigitCNN_V2, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        
        self.fc1 = nn.Linear(in_features=32 * 7 * 7, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pass 1
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2) 
        
        # Pass 2
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2) 
        
        # Classifier
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        
        return x