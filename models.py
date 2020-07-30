
import torch
import torch.nn as nn


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
        z = self.encoder(x)
        y = self.decoder(z)
        return y


class Encoder(nn.Module):
    """
    The encoder uses convolutional neural networks to condense image data into a single vector.

    Attributes:
        main (Sequential): A sequential network of convolutions, batch norms, and relu activations.
    """

    def __init__(self, k_size=3, stride=2):
        """
        Constructor.

        Args:
            k_size (int):      The kernel size for the convolutional layers.
            stride (int):      The stride of each convolutional layer.
        """
        super(Encoder, self).__init__()

        # Set up convolutional layers
        conv = (
            nn.Conv2d(1, 5, k_size, stride=stride),
            nn.Conv2d(5, 10, k_size, stride=stride),
            nn.Conv2d(10, 15, k_size, stride=stride),
        )

        # Add batch norms and activations
        modules = []
        for c in conv:
            modules.append(c)
            modules.append(nn.BatchNorm2d(c.out_channels))
            modules.append(nn.ReLU())

        # Package in sequential model
        self.main = nn.Sequential(*modules)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (Tensor): Input image data tensor of size (BATCH_SIZE, 1, 28, 28).

        Returns:
            (Tensor): An encoded vector for the inputted image
        """
        y = self.main(x)
        y = y.flatten(start_dim=1)
        y = torch.sigmoid(y)
        return y


class Decoder(nn.Module):
    """
    The decoder takes the compressed vector from the encoder and decompresses it again in an attempt
    to recreate the initial image.

    Attributes:
        main (Sequential): A sequential network of transpose convolutions, batch norms, and relu
                               activations.
    """

    def __init__(self, k_size=3, stride=2):
        """
        Constructor.

        Args:
            k_size (int):      The kernel size for the deconvolutional layers.
            stride (int):      The stride of each deconvolutional layer.
        """
        super(Decoder, self).__init__()

        # Set up deconvolutional layers
        deconv = (
            nn.ConvTranspose2d(15, 10, k_size, stride=stride, output_padding=1),
            nn.ConvTranspose2d(10, 5, k_size, stride=stride, output_padding=0),
            nn.ConvTranspose2d(5, 1, k_size, stride=stride, output_padding=1),
        )

        # Add batch norms and activations
        modules = []
        for d in deconv:
            modules.append(d)
            modules.append(nn.BatchNorm2d(d.out_channels))
            modules.append(nn.ReLU())

        # Package in sequential model
        self.main = nn.Sequential(*modules)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (Tensor): The compressed vector from the encoder.

        Returns:
            (Tensor): The output image data tensor.
        """
        y = x.view(-1, 15, 2, 2)
        y = self.main(y)
        return y
