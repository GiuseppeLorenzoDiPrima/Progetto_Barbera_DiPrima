# Third-party imports
import torch
import torch.nn as nn

# Class to define the AlexNet model by inheriting nn.Module
class AlexNet(nn.Module):
    """
    A PyTorch implementation of the AlexNet model.

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
    :param inplace: Whether to use inplace ReLU.
    :type inplace: bool
    """
    def __init__(self, type_net, stride_size, padding_size, kernel_size, channels_of_color, inplace):
        super(AlexNet, self).__init__()
        # Set the number of classes according to the configuration you choose
        if type_net.lower() == 'binary':
            num_classes = 2
        else:
            num_classes = 3
        # Conv2d -> Convolutional layer
        # ReLU -> Activation function
        # MaxPool2d -> Pooling layer
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
        # Dropout -> Regularization function
        # Linear -> Linear layer
        # ReLU -> Activation function
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=inplace),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=inplace),
            nn.Linear(4096, num_classes),
        )
    
    # Foward step
    def forward(self, x):
        """
        Defines the forward pass of the AlexNet.

        :param x: The input to the network.
        :type x: torch.Tensor
        :return: Returns the output of the network.
        :rtype: torch.Tensor
        """
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x
