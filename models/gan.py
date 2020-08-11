
import torch
import torch.nn as nn


class Descriminator(nn.Module):
    """
    The Descriminator attempts to classify the given image as real (from the
    dataset) or fake (from the generator).  Therefore, it is a binary
    classifier.
    """

    def __init__(self, n_features=4):
        """
        Constructor.

        Args:
            n_features (Optional[int]): The number of features for the input. Default is 4.
        """
        super(Descriminator, self).__init__()
        conv = (
            nn.Conv2d(n_features, 8, 3, stride=2),
            nn.Conv2d(8, 12, 3, stride=2),
            nn.Conv2d(12, 16, 3, stride=2),
            nn.Conv2d(16, 20, 3, stride=2),
        )
        modules = []
        for c in conv:
            modules.append(c)
            modules.append(nn.BatchNorm2d(c.out_channels))
            modules.append(nn.LeakyReLU())
        self.main = nn.Sequential(*modules)
        self.out = nn.Linear(720, 1)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (Tensor): A tensor of size (BATCH_SIZE, 1, 28, 28).
        """
        y = self.main(x)
        y = y.flatten(start_dim=1)
        y = self.out(y)
        y = torch.sigmoid(y)
        return y


class Generator(nn.Module):
    """
    The generator takes a random noise vector and attempts to generate an image from it.
    """

    def __init__(self):
        """
        Constructor.
        """
        super(Generator, self).__init__()
        deconv = (
            nn.ConvTranspose2d(1, 256, 3),
            nn.ConvTranspose2d(256, 256, 3),
            nn.ConvTranspose2d(256, 256, 3),
            nn.ConvTranspose2d(256, 256, 3, stride=2),
            nn.ConvTranspose2d(256, 128, 3, stride=2),
            nn.ConvTranspose2d(128, 4, 3, stride=2, output_padding=1),
        )
        modules = []
        for d in deconv:
            modules.append(d)
            modules.append(nn.BatchNorm2d(d.out_channels))
            modules.append(nn.LeakyReLU())
        self.main = nn.Sequential(*modules)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (Tensor): A latent tensor of size (BATCH_SIZE, 1, 50)
        """
        y = self.main(x)
        y = torch.tanh(y)
        return y
