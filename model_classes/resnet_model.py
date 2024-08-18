# Third-party imports
import torch
import torch.nn as nn

# Class to define the ResidualBlock by inheriting nn.Module
class ResidualBlock(nn.Module):
    """
    A PyTorch implementation of a residual block.
    """
    def __init__(self, in_channels, out_channels, stride = 1, downsample = None):
        """
        A PyTorch implementation of a residual block.

        :param in_channels: The number of input channels.
        :type in_channels: int
        :param out_channels: The number of output channels.
        :type out_channels: int
        :param stride: The stride size for the convolutional layers.
        :type stride: int
        :param downsample: The downsample function for adjusting the size of the residual connection.
        :type downsample: nn.Module or None
        """
        super(ResidualBlock, self).__init__()
        # Conv2d -> Convolutional layer
        # BatchNorm2d -> Regularization function
        # ReLU -> Activation function
        # downsample -> Pooling layer
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
        
    # Foward step
    def forward(self, x):
        """
        Defines the forward pass of the residual block.

        :param x: The input to the block.
        :type x: torch.Tensor
        :return: Returns the output of the block.
        :rtype: torch.Tensor
        """
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

# Class to define the ResNet by inheriting nn.Module
class ResNet(nn.Module):
    """
        A PyTorch implementation of the ResNet model.
    """
    def __init__(self, block, layers, type_net, stride_size, padding_size, kernel_size, channels_of_color, planes, in_features, inplanes):
        """
        A PyTorch implementation of the ResNet model.

        :param block: The type of block to use in the model.
        :type block: nn.Module
        :param layers: The number of layers for each block in the model.
        :type layers: list
        :param type_net: The type of the network (e.g., 'binary', 'ternary').
        :type type_net: str
        :param stride_size: The stride size for the convolutional layers.
        :type stride_size: list
        :param padding_size: The padding size for the convolutional layers.
        :type padding_size: list
        :param kernel_size: The kernel size for the convolutional and pooling layers.
        :type kernel_size: list
        :param channels_of_color: The number of color channels in the input images.
        :type channels_of_color: int
        :param planes: The number of output channels for each block in the model.
        :type planes: list
        :param in_features: The number of input features for the final fully connected layer.
        :type in_features: int
        :param inplanes: The number of input channels for the first convolutional layer.
        :type inplanes: int
        """
        super(ResNet, self).__init__()
        # Set the number of classes according to the configuration you choose
        if type_net.lower() == 'binary':
            num_classes = 2
        else:
            num_classes = 3
        # Conv2d -> Convolutional layer
        # BatchNorm2d -> Regularization function
        # ReLU -> Activation function
        # MaxPool2d -> Max pooling layer
        # AvgPool2d -> Average pooling layer
        self.inplanes = inplanes
        self.conv1 = nn.Sequential(
                        nn.Conv2d(channels_of_color, self.inplanes, kernel_size = kernel_size[0], stride = stride_size[0], padding = padding_size[0]),
                        nn.BatchNorm2d(self.inplanes),
                        nn.ReLU())
        self.maxpool = nn.MaxPool2d(kernel_size = kernel_size[1], stride = stride_size[0], padding = padding_size[1])
        self.layer = nn.ModuleList()
        self.layer.append(self._make_layer(block, planes[0], layers[0], stride_size[1]))
        for index in range(1, len(layers)):
            self.layer.append(self._make_layer(block, planes[index], layers[index], stride_size[0]))
        self.avgpool = nn.AvgPool2d(kernel_size=kernel_size[0], stride=stride_size[1])
        self.fc = nn.Linear(in_features, num_classes)
        
    # Create new layer
    def _make_layer(self, block, planes, blocks, stride):
        """
        Makes a layer for the ResNet model.

        :param block: The type of block to use in the layer.
        :type block: nn.Module
        :param planes: The number of output channels for the block.
        :type planes: int
        :param blocks: The number of blocks in the layer.
        :type blocks: int
        :param stride: The stride size for the convolutional layers in the block.
        :type stride: int
        :return: Returns the layer.
        :rtype: nn.Sequential
        """
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
    
    # Foward step
    def forward(self, x):
        """
        Defines the forward pass of the ResNet.

        :param x: The input to the network.
        :type x: torch.Tensor
        :return: Returns the output of the network.
        :rtype: torch.Tensor
        """
        x = self.conv1(x)
        x = self.maxpool(x)
        for index in range(len(self.layer)):
            x = self.layer[index](x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    