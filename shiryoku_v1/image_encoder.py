from networkx import rescale_layout
import torch.nn as nn
import torch 
from torchvision import models

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


class PretrainedConvNet(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        resnet = models.resnet152(pretrained=True)
        resnet_modules = list(resnet.children())[:-1]
        
        self.resnet = nn.Sequential(*resnet_modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.batch_norm = nn.BatchNorm1d(embed_size, momentum=0.01)
    
    def forward(self, images):
        with torch.no_grad():
            features = self.resnet(images)
        
        features = features.reshape(features.size(0), -1)
        features = self.batch_norm(self.linear(features))
        
        return features 