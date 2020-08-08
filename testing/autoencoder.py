
import torch.nn.functional as F
from torch.utils.data import DataLoader

from analysis import Logger, TestingLogTemplate, visualize_input_output
from data import PokemonDataset
import models


def test():
    ###################
    #    Load Data    #
    ###################

    dataset = PokemonDataset()
    data_loader = DataLoader(dataset)

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

    # Get inputs and outputs for nine pokemon
    samples = []
    for x, t in dataset[:10]:
        y = auto_encoder(x.unsqueeze(0))
        samples.append((x.squeeze(), y.detach().squeeze()))

    # Display inputs and outputs
    visualize_input_output(samples, save=True)
