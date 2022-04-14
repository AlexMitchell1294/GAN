import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.tensorboard import SummaryWriter
from Discriminator import Discriminator
from Generator import Generator
import utils


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

discriminator = Discriminator(utils.channels_img, utils.discriminator_features).to(device)
generator = Generator(utils.channels_noise, utils.channels_img, utils.generator_features).to(device)

optimizerD = optim.Adam(discriminator.parameters(), lr=utils.lr, betas=(0.1, 0.99))
optimizerG = optim.Adam(generator.parameters(), lr=utils.lr, betas=(0.1, 0.99))

noise_array = torch.randn(64, utils.channels_noise, 1, 1).to(device)
writer_real = SummaryWriter(f"runs/test_real")
writer_fake = SummaryWriter(f"runs/test_fake")

generator.train()
discriminator.train()
loss_function = nn.BCELoss()

print("Starting GAN Training...")
real_label = 1
fake_label = 0
step = 0
for epoch in range(utils.num_epochs):
    for batch_index, (data, targets) in enumerate(utils.dataloader):
        data = data.to(device)
        image = data[:32]
        batch_size = data.shape[0] #* data.shape[1] * data.shape[1]

        discriminator.zero_grad()
        label = (torch.ones(batch_size) * 0.9).to(device)
        output = discriminator(data).reshape(-1)
        lossD_real = loss_function(output, label)
        D_x = output.mean().item()

        # save memory
        del label, output, data
        torch.cuda.empty_cache()

        noise = torch.randn(batch_size, utils.channels_noise, 1, 1).to(device)
        fake = generator(noise)
        label = (torch.ones(batch_size) * 0.1).to(device)

        output = discriminator(fake.detach()).reshape(-1)
        lossD_fake = loss_function(output, label)
        del label, output, fake, noise
        torch.cuda.empty_cache()


        lossD = lossD_real + lossD_fake

        # save memory
        del lossD_fake, lossD_real
        torch.cuda.empty_cache()
        lossD.backward()
        optimizerD.step()

        ### Train Generator: max log(D(G(z)))
        generator.zero_grad()
        label = torch.ones(batch_size).to(device)
        noise = torch.randn(batch_size, utils.channels_noise, 1, 1).to(device)
        fake = generator(noise)
        output = discriminator(fake).reshape(-1)
        lossG = loss_function(output, label)
        lossG.backward()
        optimizerG.step()

        # save memory
        del fake, output, label, noise
        torch.cuda.empty_cache()

        if batch_index % 100 == 0:
            step += 1
            print(
                f"Epoch [{epoch}/{utils.num_epochs}] Batch \
                {batch_index}/{len(utils.dataloader)} \
                  Loss D: {lossD:.4f}, loss G: {lossG:.4f}"
            )

            with torch.no_grad():
                fake = generator(noise_array)
                image_grid_real = torchvision.utils.make_grid(image, normalize=True)
                image_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)
                writer_real.add_image(
                    "Real", image_grid_real, global_step=step
                )
                writer_fake.add_image(
                    "Fake", image_grid_fake, global_step=step
                )

        # make sure exists then save memory
        if "fake" in locals():
            del image, fake, image_grid_real, image_grid_fake
        else:
            del image
        torch.cuda.empty_cache()