
import torch
import torch.nn as nn
import torch.nn.functional as F


class AutoEncoder(nn.Module):
    """
    The auto encoder takes image data as an input, compresses it into a single vector, then
    decompresses it again to attempt to recreate the initial image.

    Attributes:
        decoder (Decoder): A decoder to decompress the vector from the encoder.
        encoder (Encoder): An encoder to compress image data into a single vector.
    """

    def __init__(self, encoder, decoder):
        """
        Constructor.

        Args:
            encoder (Encoder): An encoder object.
            decoder (Decoder): A decoder object.
        """
        super(AutoEncoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (Tensor):   Input image data tensor of size (BATCH_SIZE, 1, 28, 28).

        Returns:
            (Tensor): The output image data.
        """
        z, indices = self.encoder(x)
        y = self.decoder(z, indices)
        return y


class Encoder(nn.Module):
    """
    The encoder uses convolutional neural networks to condense image data into a single vector.

    Attributes:
        conv1 (Conv2d): The first CNN.
        conv2 (Conv2d): The second CNN.
        lin1 (Linear):  The first linear layer.
        lin2 (Linear):  The second linear layer.
    """

    def __init__(self):
        """
        Constructor.
        """
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.lin1 = nn.Linear(320, 160)
        self.lin2 = nn.Linear(160, 50)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (Tensor): Input image data tensor of size (BATCH_SIZE, 1, 28, 28).

        Returns:
            (Tensor): An encoded vector for the inputted image
            (tuple):  The indices for the max pools.
        """
        y = self.conv1(x)
        y, i1 = F.max_pool2d(y, kernel_size=(2, 2), return_indices=True)
        y = torch.sigmoid(y)
        y = self.conv2(y)
        y, i2 = F.max_pool2d(y, kernel_size=(2, 2), return_indices=True)
        y = torch.sigmoid(y)
        y = y.view(-1, 320)
        y = torch.sigmoid(self.lin1(y))
        y = torch.sigmoid(self.lin2(y))
        return y, (i1, i2)


class Decoder(nn.Module):
    """
    The decoder takes the compressed vector from the encoder and decompresses it again in an attempt
    to recreate the initial image.

    Attributes:
        batch_size (int):        The batch size.
        conv1 (ConvTranspose2d): The first transpose CNN.
        conv2 (ConvTranspose2d): The second transpose CNN.
        lin1 (Linear):           The first linear layer.
        lin2 (Linear):           The second linear layer.
    """

    def __init__(self, batch_size=1):
        """
        Constructor.

        Args:
            batch_size (int): The batch size.
        """
        super(Decoder, self).__init__()
        self.batch_size = batch_size

        self.lin1 = nn.Linear(50, 160)
        self.lin2 = nn.Linear(160, 320)
        self.conv1 = nn.ConvTranspose2d(20, 10, kernel_size=5)
        self.conv2 = nn.ConvTranspose2d(10, 1, kernel_size=5)

    def forward(self, x, indices):
        """
        Forward pass.

        Args:
            x (Tensor):      The compressed vector from the encoder.
            indices (tuple): The indices from the max pools from the encoder.

        Returns:
            (Tensor/Image): The output image data (or image).
        """
        y = torch.sigmoid(self.lin1(x))
        y = torch.sigmoid(self.lin2(y))
        y = y.view(self.batch_size, 20, 4, 4)
        y = F.max_unpool2d(y, indices[1], kernel_size=(2, 2))
        y = torch.sigmoid(self.conv1(y))
        y = F.max_unpool2d(y, indices[0], kernel_size=(2, 2))
        y = torch.sigmoid(self.conv2(y))
        return y
