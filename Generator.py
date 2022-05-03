import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, channels_noise, channels_img, features_g):
        super(Generator, self).__init__()

        self.net = nn.Sequential(
            # layer
            nn.ConvTranspose2d(
                channels_noise, features_g * 16, kernel_size=4, stride=1, padding=0
            ),
            nn.BatchNorm2d(features_g * 16),
            nn.ReLU(),
            # layer
            nn.ConvTranspose2d(
                features_g * 16, features_g * 8, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(features_g * 8),
            nn.ReLU(),
            # layer
            nn.ConvTranspose2d(
                features_g * 8, features_g * 4, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(features_g * 4),
            nn.ReLU(),
            # layer
            nn.ConvTranspose2d(
                features_g * 4, features_g * 2, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(features_g * 2),
            nn.ReLU(),

            # final layer
            nn.ConvTranspose2d(
                features_g * 2, channels_img, kernel_size=4, stride=2, padding=1
            ),
            # not sure why this works better
            nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x)