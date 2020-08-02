
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

from analysis import Logger, TestingLogTemplate, visualize_input_output
import models


###################
#    Load Data    #
###################

mnist_data = MNIST('data', transform=transforms.ToTensor(), download=True, train=False)
data_loader = DataLoader(mnist_data)

###################
#  Set Up Models  #
###################

auto_encoder = models.AutoEncoder()
auto_encoder.load_states()

###################
#     Testing     #
###################

auto_encoder.eval()
log = Logger('test.log', TestingLogTemplate())

total_loss = 0
status = ''
for step, (x, _) in enumerate(data_loader):
    # Train
    y = auto_encoder(x)
    loss = F.mse_loss(y, x)

    # Display status
    total_loss += loss.item()
    ctx = {
        'step': step + 1,
        'loss': total_loss / (step + 1),
        'data_len': len(data_loader),
    }
    log.write(ctx, overwrite=True)
log.close()

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
visualize_input_output(digits, save=True)
