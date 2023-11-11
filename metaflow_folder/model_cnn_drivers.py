import torch.nn.functional as F
from torch import nn


class CNN(nn.Module):
    """
    A convolutional neural network model implemented using the PyTorch library.

    Args:
        num_classes (int): The number of output classes for classification.
        n_channel (int): The number of input channels.

    Attributes:
        conv1 (nn.Conv2d): The first convolutional layer with 32 output channels and a kernel size of 3.
        pool (nn.MaxPool2d): The max pooling layer with a kernel size of 2 and a stride of 2.
        conv2 (nn.Conv2d): The second convolutional layer with 16 output channels and a kernel size of 4.
        fc1 (nn.Linear): The first fully connected layer with 16 output features.
        fc2 (nn.Linear): The second fully connected layer with `num_classes` output features.

    Methods:
        __init__(self, num_classes, n_channel): Initializes the CNN model by defining the layers and their parameters.
        forward(self, x): Performs forward propagation of the input tensor `x` through the model to obtain the output predictions.
    """

    def __init__(self, num_classes, n_channel):
        super().__init__()
        self.conv1 = nn.Conv2d(n_channel, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 16, 4)
        self.fc1 = nn.Linear(16 * 4 * 6, 16)
        self.fc2 = nn.Linear(
            16, num_classes
        )  # Output has 2 classes for binary classification

    def forward(self, x):
        """
        Performs forward propagation of the input tensor `x` through the model to obtain the output predictions.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor containing the predictions.
        """
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 6)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
