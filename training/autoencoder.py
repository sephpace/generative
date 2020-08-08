
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from analysis import Logger, TrainingLogTemplate
from data import PokemonDataset
import models

EPOCHS = 2000
LEARNING_RATE = 0.001
BATCH_SIZE = 8
SHUFFLE = True


def train():
    ###################
    #    Load Data    #
    ###################

    dataset = PokemonDataset(add_mirrored=True)
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=SHUFFLE)

    ###################
    #  Set Up Models  #
    ###################

    auto_encoder = models.AutoEncoder()

    optimizer = optim.Adam(auto_encoder.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.MSELoss()

    ###################
    #    Training     #
    ###################

    auto_encoder.train()
    log = Logger('train.log', TrainingLogTemplate())

    for epoch in range(EPOCHS):
        epoch_loss = 0
        ctx = {}
        for step, (x, _) in enumerate(data_loader):
            # Train
            y = auto_encoder(x)
            optimizer.zero_grad()
            loss = criterion(y, x)
            loss.backward()
            optimizer.step()

            # Display status
            epoch_loss += loss.item()
            ctx = {
                'epoch': epoch + 1,
                'epochs': EPOCHS,
                'step': step + 1,
                'loss': epoch_loss / (step + 1),
                'data_len': len(data_loader),
            }
            log.write(ctx, overwrite=True)
        log.write(ctx)
    log.close()

    ###################
    #   Save Models   #
    ###################

    auto_encoder.save_states()
