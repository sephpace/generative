
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

from . import models

EPOCHS = 20
LEARNING_RATE = 0.001
BATCH_SIZE = 6
SHUFFLE = True


###################
#    Load Data    #
###################

mnist_data = MNIST('data', transform=transforms.ToTensor(), download=True)
data_loader = DataLoader(mnist_data, batch_size=BATCH_SIZE, shuffle=SHUFFLE)

###################
#  Set Up Models  #
###################

encoder = models.Encoder()
decoder = models.Decoder(batch_size=BATCH_SIZE)
auto_encoder = models.AutoEncoder(encoder, decoder)

optimizer = optim.SGD(auto_encoder.parameters(), lr=LEARNING_RATE)
criterion = torch.nn.MSELoss()

###################
#    Training     #
###################

auto_encoder.train()
status_template = 'Epoch: {epoch:{EPOCHS_STR_LEN}d}/{EPOCHS}  Item: {step}/{data_length}  Loss: {loss:.4f}'

for epoch in range(EPOCHS):
    epoch_loss = 0
    for step, (x, _) in enumerate(data_loader):
        # Train
        y = auto_encoder(x)
        optimizer.zero_grad()
        loss = criterion(y, x)
        loss.backward()
        optimizer.step()

        # Display status
        epoch_loss += loss.item()
        status = status_template.format_map({
            'epoch': epoch + 1,
            'EPOCHS': EPOCHS,
            'EPOCHS_STR_LEN': len(str(EPOCHS)),
            'step': step + 1,
            'loss': epoch_loss / (step + 1),
            'data_length': len(data_loader),
        })
        print(status, end='\r')
    print()

###################
#   Save Models   #
###################

torch.save(encoder.state_dict(), 'states/encoder.pt')
torch.save(decoder.state_dict(), 'states/decoder.pt')
