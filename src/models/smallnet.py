
import torch
import torch.nn as nn
import torch.nn.functional as F
class SmallNet(nn.Module):
    def __init__(self, in_shape, num_classes=10, p_drop=0.1):
        super().__init__()
        c,h,w = in_shape
        self.flatten = nn.Flatten()

        # Simple MLP for MNIST, small CNN for CIFAR
        if (c,h,w) == (1,28,28):
            self.backbone = nn.Sequential(
                nn.Linear(28*28, 256),
                nn.ReLU(),
                nn.Dropout(p_drop),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(p_drop),
            )
            self.head = nn.Linear(128, num_classes)
        else:
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Dropout2d(p_drop),
                nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Dropout2d(p_drop),
            )
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(64*8*8, 256), nn.ReLU(),
                nn.Dropout(p_drop),
                nn.Linear(256, num_classes)
            )

    def forward(self, x):
        if x.dim() == 4 and x.shape[1] == 3:
            z = self.backbone(x)
            return self.head(z)
        z = self.flatten(x)
        z = self.backbone(z)
        return self.head(z)
