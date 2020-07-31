
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

import models


###################
#    Load Data    #
###################

mnist_data = MNIST('data', transform=transforms.ToTensor(), download=True, train=False)
data_loader = DataLoader(mnist_data)

###################
#  Set Up Models  #
###################

encoder = models.Encoder()
decoder = models.Decoder()

encoder.load_state_dict(torch.load('states/encoder.pt'))
decoder.load_state_dict(torch.load('states/decoder.pt'))

auto_encoder = models.AutoEncoder(encoder, decoder)

###################
#     Testing     #
###################

auto_encoder.eval()
status_template = 'Item: {step}/{data_length}  Loss: {loss:.4f}'

total_loss = 0
for step, (x, _) in enumerate(data_loader):
    # Train
    y = auto_encoder(x)
    loss = F.mse_loss(y, x)

    # Display status
    total_loss += loss.item()
    status = status_template.format_map({
        'step': step + 1,
        'loss': total_loss / (step + 1),
        'data_length': len(data_loader),
    })
    print(status, end='\r')
print()

###################
#     Visuals     #
###################

# Get inputs and outputs for each digit
digits = []
digit = 0
for x, t in mnist_data:
    if t == digit:
        y = auto_encoder(x.unsqueeze(0))
        digits.append((x.squeeze(), y.detach().squeeze()))
        digit += 1
    if digit > 9:
        break

# Display inputs and outputs

fig = plt.figure(figsize=(8, 8))
rows = 5
cols = 4
for i, (x, y) in enumerate(digits):
    fig.add_subplot(rows, cols, i * 2 + 1)
    plt.imshow(x)
    fig.add_subplot(rows, cols, i * 2 + 2)
    plt.imshow(y)
plt.show()
