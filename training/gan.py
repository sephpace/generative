
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from analysis import Logger, LogTemplate
from data import PokemonDataset
import models

EPOCHS = 200
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

    generator = models.Generator()
    descriminator = models.Descriminator()

    g_opt = optim.Adam(generator.parameters(), lr=LEARNING_RATE)
    d_opt = optim.SGD(descriminator.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.BCELoss()

    ###################
    #    Training     #
    ###################

    generator.train()
    descriminator.train()
    template_str = '[{train_type}]  Epoch: {epoch:{epochs_str_len}d}/{epochs}  Item: {step}/{data_len}  Loss: {loss:.4f}'
    log = Logger('train.log', LogTemplate(template_str))

    for epoch in range(EPOCHS):
        d_pass = True
        epoch_loss = 0
        ctx = {}
        for _ in range(2):
            for step, (real_img, _) in enumerate(data_loader):
                current_batch_size = real_img.shape[0]

                # Generate image(s)
                z = torch.randn(current_batch_size, 1, 8, 8)
                fake_img = generator(z)

                # Descriminator target values
                real_t = torch.ones((current_batch_size, 1))
                fake_t = torch.zeros((current_batch_size, 1))

                # Train
                if d_pass:
                    # Descriminator training pass
                    train_type = 'D'
                    real_y = descriminator(real_img)
                    fake_y = descriminator(fake_img)
                    d_opt.zero_grad()
                    real_loss = criterion(real_y, real_t)
                    fake_loss = criterion(fake_y, fake_t)
                    loss = (real_loss + fake_loss) / 2
                    loss.backward()
                    d_opt.step()
                    epoch_loss += loss.item()
                else:
                    # Generator training pass
                    train_type = 'G'
                    y = descriminator(fake_img)
                    g_opt.zero_grad()
                    loss = criterion(y, real_t)
                    loss.backward()
                    g_opt.step()
                    epoch_loss += loss.item()

                # Display status
                ctx = {
                    'train_type': train_type,
                    'epoch': epoch + 1,
                    'epochs': EPOCHS,
                    'epochs_str_len': len(str(EPOCHS)),
                    'step': step + 1,
                    'loss': epoch_loss / (step + 1),
                    'data_len': len(data_loader),
                }
                log.write(ctx, overwrite=True)
            d_pass = False
            log.write(ctx)
    log.close()

    ###################
    #   Save Model    #
    ###################

    torch.save(generator.state_dict(), 'states/generator.pt')
