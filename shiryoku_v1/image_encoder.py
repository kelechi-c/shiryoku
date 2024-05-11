import torch.nn as nn
import torch 

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
            nn.Linear(128 * 7 * 7, 128),
            nn.Linear(128, 128),
        )

    def forward(self, image):
        encoded_image = self.conv_net(image)

        return encoded_image
