
import torch.nn as nn
import torch.nn.functional as F


class AutoEncoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(AutoEncoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        z, indices = self.encoder(x)
        y = self.decoder(z, indices)
        return y


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.lin1 = nn.Linear(320, 160)
        self.lin2 = nn.Linear(160, 50)

    def forward(self, x):
        y = self.conv1(x)
        y, i1 = F.max_pool2d(y, kernel_size=(2, 2), return_indices=True)
        y = F.relu(y)
        y = self.conv2(y)
        y, i2 = F.max_pool2d(y, kernel_size=(2, 2), return_indices=True)
        y = F.relu(y)
        y = y.view(-1, 320)
        y = F.relu(self.lin1(y))
        y = F.relu(self.lin2(y))
        return y, (i1, i2)


class Decoder(nn.Module):
    def __init__(self, batch_size=1):
        super(Decoder, self).__init__()
        self.batch_size = batch_size

        self.lin1 = nn.Linear(50, 160)
        self.lin2 = nn.Linear(160, 320)
        self.conv1 = nn.ConvTranspose2d(20, 10, kernel_size=5)
        self.conv2 = nn.ConvTranspose2d(10, 1, kernel_size=5)

    def forward(self, x, indices):
        y = F.relu(self.lin1(x))
        y = F.relu(self.lin2(y))
        y = y.view(self.batch_size, 20, 4, 4)
        y = F.max_unpool2d(y, indices[1], kernel_size=(2, 2))
        y = F.relu(self.conv1(y))
        y = F.max_unpool2d(y, indices[0], kernel_size=(2, 2))
        y = F.relu(self.conv2(y))
        return y
