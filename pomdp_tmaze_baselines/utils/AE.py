
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class Autoencoder(nn.Module):
    def __init__(
            self, input_dims, latent_dims, hidden_size, in_channels,
            kernel_size=3, padding=3, dilation=1,
            conv_hidden_size=16, conv1_stride=4, maxpool_stride=1):
        super(Autoencoder, self).__init__()

        self.kernel_size = kernel_size
        self.padding = padding
        self.dilation = dilation
        self.conv_hidden_size = conv_hidden_size
        self.conv1_stride = conv1_stride
        self.maxpool_stride = maxpool_stride

        self.encoder = Encoder(
            input_dims, latent_dims, hidden_size, in_channels)
        self.decoder = Decoder(
            input_dims, latent_dims, hidden_size, in_channels)
        # self.z = torch.empty((batch_size, latent_dims))

    def forward(self, x):
        self.z = self.encoder(x)
        return self.decoder(self.z)


# Source: https://avandekleut.github.io/vae/
class Encoder(nn.Module):
    def __init__(
            self, input_dims, latent_dims, hidden_size, in_channels,
            kernel_size, padding, dilation,
            conv_hidden_size, conv1_stride, maxpool_stride):
        super(Encoder, self).__init__()
        self.input_dims = input_dims

        # First Conv Layer
        h_out1 = int(np.floor(((
            input_dims + 2*padding - dilation * (kernel_size-1) - 1)
            / conv1_stride) + 1))

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=conv_hidden_size,
            kernel_size=kernel_size,
            stride=conv1_stride,
            padding=padding,
            dilation=dilation)
        self.relu1 = nn.ReLU()

        self.maxpool1 = nn.MaxPool2d(
            kernel_size=kernel_size, stride=maxpool_stride)
        h_out2 = int(np.floor(((
            h_out1 - dilation * (kernel_size-1) - 1)
            / maxpool_stride) + 1))

        # Linear Layers
        self.linear1 = nn.Linear(h_out2*h_out2*conv_hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, latent_dims)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))
        return self.linear2(x)


class Decoder(nn.Module):
    def __init__(
            self, input_dims, latent_dims, hidden_size, in_channels,
            kernel_size, padding, dilation,
            conv_hidden_size, conv1_stride, maxpool_stride):
        super(Decoder, self).__init__()
        self.input_dims = input_dims
        self.hidden_size = hidden_size

        self.conv_hidden_size = conv_hidden_size

        self.decoder_h_in = (
            conv1_stride*(input_dims-1) + 1 - 2*padding
            + dilation * (kernel_size - 1))

        self.linear1 = nn.Linear(latent_dims, hidden_size)
        self.linear2 = nn.Linear(
            hidden_size, self.decoder_h_in*self.decoder_h_in*conv_hidden_size)

        self.conv1 = nn.Conv2d(
            in_channels=conv_hidden_size,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=conv1_stride,
            padding=padding,
            dilation=dilation)

    def forward(self, z):
        z = F.relu(self.linear1(z))
        z = torch.sigmoid(self.linear2(z))
        z = z.reshape((
            -1, self.conv_hidden_size,
            self.decoder_h_in, self.decoder_h_in))
        return self.conv1(z)
