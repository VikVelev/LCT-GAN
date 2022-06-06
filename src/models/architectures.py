import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

import numpy as  np

class FCEncoder(nn.Module):
    def __init__(self, output_size, input_size):
        super(FCEncoder, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        h0 = F.relu(self.fc1(x))
        h1 = F.relu(self.fc2(h0))
        return F.relu(self.fc3(h1))

class FCDecoder(nn.Module):
    def __init__(self, embedding_size, input_size):
        super(FCDecoder, self).__init__()
        self.fc4 = nn.Linear(embedding_size, 64)
        self.fc5 = nn.Linear(64, 128)
        self.fc6 = nn.Linear(128, input_size)

    def forward(self, z):
        h3 = F.relu(self.fc4(z))
        h4 = F.relu(self.fc5(h3))
        return torch.sigmoid(self.fc6(h4))

class FCGenerator(nn.Module):
    def __init__(self, output_size, latent_size):
        super(FCGenerator, self).__init__()

        self.output_size = output_size
        self.latent_dim = latent_size

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(self.latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 256),
            *block(256, 128),
            nn.Linear(128, self.output_size),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], self.output_size)
        return img


class FCDiscriminator(nn.Module):
    def __init__(self, input_size, batch_size):
        super(FCDiscriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(16, 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(8, 1),
        )
        
        self.input_size = input_size
        self.batch_size = batch_size

    def forward(self, data):
        data_flat = data.view(data.shape[0], self.input_size)
        return self.model(data_flat)