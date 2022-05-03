import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, channels_img, features_d):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            #layer
            nn.Conv2d(channels_img, features_d, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            # layer
            nn.Conv2d(features_d, features_d * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(features_d * 4),
            nn.LeakyReLU(0.2),
            # layer
            nn.Conv2d(
                features_d * 4, features_d * 8, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(features_d * 8),
            nn.LeakyReLU(0.2),
            # layer
            nn.Conv2d(
                features_d * 8, features_d * 16, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(features_d * 16),
            nn.LeakyReLU(0.2),

            # layer
            nn.Conv2d(features_d * 16, 1, kernel_size=4, stride=2, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)