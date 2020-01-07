import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(11, 21)),  # b, 16, 10, 10
            nn.Conv2d(16, 16, kernel_size=(11, 21)),  # b, 16, 10, 10
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=(7, 13)),  # b, 16, 10, 10
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=(5, 9)),  # b, 8, 3, 3
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, kernel_size=(3, 5)),  # b, 8, 3, 3
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=(3, 5)),  # b, 16, 5, 5
            nn.LeakyReLU(0.2, True),
            nn.ConvTranspose2d(64, 32, kernel_size=(5, 9)),  # b, 8, 15, 15
            nn.LeakyReLU(0.2, True),
            nn.ConvTranspose2d(32, 32, kernel_size=(2, 3), stride=2),  # b, 1, 28, 28
            nn.LeakyReLU(0.2, True),
            nn.ConvTranspose2d(32, 16, kernel_size=(7, 13)),  # b, 1, 28, 28
            nn.LeakyReLU(0.2, True),
            nn.ConvTranspose2d(16, 16, kernel_size=(2, 3), stride=2),  # b, 1, 28, 28
            nn.LeakyReLU(0.2, True),
            nn.ConvTranspose2d(16, 1, kernel_size=(11, 21)),  # b, 1, 28, 28
            nn.LeakyReLU(0.2, True),
            nn.ConvTranspose2d(1, 1, kernel_size=(11, 21)),  # b, 1, 28, 28
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        # print(x.shape)
        x = self.decoder(x)
        return x


# model = AutoEncoder().cuda()
# inputs = torch.ones(32, 1, 128, 219).cuda()
# print(inputs.shape)
# outputs = model(inputs)
# print(outputs.shape)
