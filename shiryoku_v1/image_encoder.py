import torch.nn as nn
import torch 
from torchvision.models import resnet152

class ConvNetEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU().nn.Conv2d(32, 64, kernel_size=3),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 256),
            nn.Linear(256, 256),
            nn.BatchNorm1d(128, momentum=0.01)
        )

    def forward(self, image):
        encoded_image = self.conv_net(image)

        return encoded_image
