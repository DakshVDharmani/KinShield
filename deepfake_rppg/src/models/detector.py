import torch
import torch.nn as nn
from torchvision import models

class RPPGResNet(nn.Module):
    def __init__(self, pretrained=True):
        super(RPPGResNet, self).__init__()
        # Load ResNet18
        self.resnet = models.resnet18(pretrained=pretrained)
        
        # Modify first layer to accept 1 channel (Spectrogram) instead of 3 (RGB)
        # Weights are averaged to Initialize
        original_first_layer = self.resnet.conv1
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        with torch.no_grad():
            self.resnet.conv1.weight[:] = torch.mean(original_first_layer.weight, dim=1, keepdim=True)
        
        # Add dropout after pooling
        self.dropout = nn.Dropout(p=0.3)
        
        # Modify last layer for Binary Classification (Real vs Fake)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (Batch, 1, Freq, Time)
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        
        x = self.dropout(x)
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        
        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.resnet.fc(x)
        return self.sigmoid(x)


class SimpleRPPGNet(nn.Module):
    """Simpler model for small datasets - avoids overfitting."""
    def __init__(self):
        super(SimpleRPPGNet, self).__init__()
        # Simple CNN-based architecture
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(128 * 28 * 28, 256),
            nn.ReLU(inplace=False),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=False),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class RPPGEfficientNet(nn.Module):
    """Improved model using EfficientNet backbone with better regularization."""
    def __init__(self, pretrained=True):
        super(RPPGEfficientNet, self).__init__()
        # Load EfficientNet-B0 for better accuracy with fewer parameters
        if pretrained:
            weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
            self.efficientnet = models.efficientnet_b0(weights=weights)
        else:
            self.efficientnet = models.efficientnet_b0(weights=None)
        
        # Modify first layer to accept 1 channel (Spectrogram) instead of 3 (RGB)
        original_first_layer = self.efficientnet.features[0][0]
        self.efficientnet.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        
        with torch.no_grad():
            self.efficientnet.features[0][0].weight[:] = torch.mean(original_first_layer.weight, dim=1, keepdim=True)
        
        # Modify classifier with better regularization
        num_ftrs = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=False),
            nn.Linear(num_ftrs, 128),
            nn.ReLU(inplace=False),
            nn.Dropout(p=0.2, inplace=False),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.efficientnet(x)
