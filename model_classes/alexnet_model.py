# Third-party imports
import torch
import torch.nn as nn

class AlexNet(nn.Module):
    def __init__(self, type_net, stride_size, padding_size, kernel_size, channels_of_color, inplace):
        super(AlexNet, self).__init__()
        num_classes = 2
        self.features = nn.Sequential(
            nn.Conv2d(channels_of_color, 96, kernel_size= kernel_size[0], stride=stride_size[0], padding=padding_size[0]),
            nn.ReLU(inplace=inplace),
            nn.MaxPool2d(kernel_size=kernel_size[2], stride=stride_size[1]),
            nn.Conv2d(96, 256, kernel_size=kernel_size[1], padding=padding_size[0]),
            nn.ReLU(inplace=inplace),
            nn.MaxPool2d(kernel_size=kernel_size[2], stride=stride_size[1]),
            nn.Conv2d(256, 384, kernel_size=kernel_size[2], padding=padding_size[1]),
            nn.ReLU(inplace=inplace),
            nn.Conv2d(384, 384, kernel_size=kernel_size[2], padding=padding_size[1]),
            nn.ReLU(inplace=inplace),
            nn.Conv2d(384, 256, kernel_size=kernel_size[2], padding=padding_size[1]),
            nn.ReLU(inplace=inplace),
            nn.MaxPool2d(kernel_size=kernel_size[2], stride=stride_size[1]),
            )
        
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=inplace),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=inplace),
            nn.Linear(4096, num_classes),
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x
