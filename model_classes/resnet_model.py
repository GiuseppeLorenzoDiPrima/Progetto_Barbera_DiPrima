# Third-party imports
import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1, downsample = None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU())
        self.conv2 = nn.Sequential(
                        nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1),
                        nn.BatchNorm2d(out_channels))
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out
    
class ResNet(nn.Module):
    def __init__(self, block, layers, type_net, stride_size, padding_size, kernel_size, channels_of_color, planes, in_features):
        super(ResNet, self).__init__()
        num_classes = 2
        self.inplanes = 64
        self.conv1 = nn.Sequential(
                        nn.Conv2d(channels_of_color, 64, kernel_size = kernel_size[0], stride = stride_size[0], padding = padding_size[0]),
                        nn.BatchNorm2d(64),
                        nn.ReLU())
        self.maxpool = nn.MaxPool2d(kernel_size = kernel_size[1], stride = stride_size[0], padding = padding_size[1])
        self.layer0 = self._make_layer(block, planes[0], layers[0], stride_size[1])
        self.layer1 = self._make_layer(block, planes[1], layers[1], stride_size[0])
        self.layer2 = self._make_layer(block, planes[2], layers[2], stride_size[0])
        self.layer3 = self._make_layer(block, planes[3], layers[3], stride_size[0])
        self.avgpool = nn.AvgPool2d(kernel_size=kernel_size[0], stride=stride_size[1])
        self.fc = nn.Linear(in_features, num_classes)
        
    def _make_layer(self, block, planes, blocks, stride):

        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
    