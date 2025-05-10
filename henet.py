import torch
import torch.nn as nn
from torchvision import models
from common import HEMaxBlock

BACKBONE_MAPPING = {
    'resnet18': {
        'm': models.resnet18,
        'last_layer': 'fc',
        'in_features': 512
    },
    'resnet50': {
        'm': models.resnet50,
        'last_layer': 'fc',
        'in_features': 2048
    }
}

class Backbone(nn.Module):
    def __init__(self, backbone='resnet18'):
        super().__init__()
        self.bb_info = BACKBONE_MAPPING[backbone]
        self.backbone = self.bb_info['m'](pretrained=True)
    
    def forward(self, x):
        return self.backbone(x)

class HENet(nn.Module):
    def __init__(self, backbone='resnet18', n_class=200, beta=1.5, block_expansion=1):
        super().__init__()
        net = Backbone(backbone)
        self.backbone = nn.Sequential(*list(net.backbone.children())[:-2])
        n_features = net.bb_info['in_features']
        self.reduce_channel = nn.Conv2d(n_features, n_class * block_expansion, kernel_size=1)
        self.HE_block = HEMaxBlock(beta)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(n_class * block_expansion, n_class)

    def forward(self, x):
        x = self.backbone(x)
        x = self.reduce_channel(x)
        x = self.HE_block(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
