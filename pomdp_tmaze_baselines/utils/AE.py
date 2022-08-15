
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torchsummary import summary


# k: kernel_size, p: padding, s: stride, d: dilation.
def conv_shape(x, k=1, p=0, s=1, d=1):
    return int((x + 2*p - d*(k - 1) - 1)/s + 1)


# TODO: There exists a bug possibly in the layer sizes, which training/test
# losses always returns the same.
class ConvAutoencoder(nn.Module):
    def __init__(
            self, input_dims, latent_dims, hidden_size, in_channels,
            kernel_size=3, padding=3, dilation=1,
            conv_hidden_size=16, conv1_stride=4, maxpool_stride=1):
        super(ConvAutoencoder, self).__init__()

        self.kernel_size = kernel_size
        self.padding = padding
        self.dilation = dilation
        self.conv_hidden_size = conv_hidden_size
        self.conv1_stride = conv1_stride
        self.maxpool_stride = maxpool_stride

        self.encoder = EncoderConv(
            input_dims, latent_dims, hidden_size, in_channels,
            kernel_size, padding, dilation,
            conv_hidden_size, conv1_stride, maxpool_stride)
        self.decoder = DecoderConv(
            input_dims, latent_dims, hidden_size, in_channels,
            kernel_size, padding, dilation,
            conv_hidden_size, conv1_stride, maxpool_stride)
        summary(self.to("cuda"), (in_channels, input_dims, input_dims))

    def forward(self, x):
        z = self.encoder(x)
        result = self.decoder(z)
        return result, z


# Source: https://avandekleut.github.io/vae/
class EncoderConv(nn.Module):
    def __init__(
            self, input_dims, latent_dims, hidden_size, in_channels,
            kernel_size, padding, dilation,
            conv_hidden_size, conv1_stride, maxpool_stride):
        super(EncoderConv, self).__init__()
        self.input_dims = input_dims

        self.shape2d_1 = conv_shape(self.input_dims, k=kernel_size)
        self.shape2d_2 = conv_shape(self.shape2d_1, k=kernel_size)
        self.shape2d_3 = conv_shape(self.shape2d_2, k=kernel_size)

        self.shape1d_1 = conv_shape(self.shape2d_3, k=kernel_size)
        self.shape1d_2 = conv_shape(self.shape1d_1, k=kernel_size)
        self.shape1d_3 = conv_shape(self.shape1d_2, k=kernel_size)

        # encoder
        self.enc2d_1 = nn.Conv2d(3, 32, kernel_size)
        self.enc2d_2 = nn.Conv2d(32, 32, kernel_size)
        self.enc2d_3 = nn.Conv2d(32, 1, kernel_size)

        self.enc1d_1 = nn.Conv1d(self.shape2d_3, 32, kernel_size)
        self.enc1d_2 = nn.Conv1d(32, 32, kernel_size)
        self.enc1d_3 = nn.Conv1d(32, 8, kernel_size)

        self.linear1 = nn.Linear(8*self.shape1d_3, hidden_size)
        self.linear2 = nn.Linear(hidden_size, latent_dims)

    def forward(self, x):
        x = F.relu(self.enc2d_1(x))
        x = F.relu(self.enc2d_2(x))
        x = F.relu(self.enc2d_3(x))
        x = x.flatten(start_dim=1, end_dim=2)
        x = F.relu(self.enc1d_1(x))
        x = F.relu(self.enc1d_2(x))
        x = F.relu(self.enc1d_3(x))
        x = x.flatten(start_dim=1, end_dim=2)
        x = F.relu(self.linear1(x))
        x = F.sigmoid(self.linear2(x))
        return x


class DecoderConv(nn.Module):
    def __init__(
            self, input_dims, latent_dims, hidden_size, in_channels,
            kernel_size, padding, dilation,
            conv_hidden_size, conv1_stride, maxpool_stride):
        super(DecoderConv, self).__init__()
        self.input_dims = input_dims
        self.hidden_size = hidden_size

        self.conv_hidden_size = conv_hidden_size

        ####
        self.shape2d_1 = conv_shape(self.input_dims, k=kernel_size)
        self.shape2d_2 = conv_shape(self.shape2d_1, k=kernel_size)
        self.shape2d_3 = conv_shape(self.shape2d_2, k=kernel_size)

        self.shape1d_1 = conv_shape(self.shape2d_3, k=kernel_size)
        self.shape1d_2 = conv_shape(self.shape1d_1, k=kernel_size)
        self.shape1d_3 = conv_shape(self.shape1d_2, k=kernel_size)

        # decode
        self.linear1 = nn.Linear(latent_dims, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 8*self.shape1d_3)

        self.dec1d_1 = nn.ConvTranspose1d(8, 32, kernel_size)
        self.dec1d_2 = nn.ConvTranspose1d(32, 32, kernel_size)
        self.dec1d_3 = nn.ConvTranspose1d(32, self.shape2d_3, kernel_size)

        self.dec2d_1 = nn.ConvTranspose2d(1, 32, kernel_size)
        self.dec2d_2 = nn.ConvTranspose2d(32, 32, kernel_size)
        self.dec2d_3 = nn.ConvTranspose2d(32, 3, kernel_size)

    def forward(self, z):

        z = F.relu(self.linear1(z))
        z = F.relu(self.linear2(z))
        z = z.unflatten(1, (8, self.shape1d_3))
        z = F.relu(self.dec1d_1(z))
        z = F.relu(self.dec1d_2(z))
        z = F.relu(self.dec1d_3(z))
        z = z.unflatten(1, (1, self.shape2d_3))
        z = F.relu(self.dec2d_1(z))
        z = F.relu(self.dec2d_2(z))
        z = F.sigmoid(self.dec2d_3(z))
        return z


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
            input_dims, latent_dims, hidden_size, in_channels,
            kernel_size, padding, dilation,
            conv_hidden_size, conv1_stride, maxpool_stride)
        self.decoder = Decoder(
            input_dims, latent_dims, hidden_size, in_channels,
            kernel_size, padding, dilation,
            conv_hidden_size, conv1_stride, maxpool_stride)
        # z = torch.empty((batch_size, latent_dims))

    def forward(self, x):
        z = self.encoder(x)
        result = self.decoder(z)
        return result, z


# Source: https://avandekleut.github.io/vae/
class Encoder(nn.Module):
    def __init__(
            self, input_dims, latent_dims, hidden_size, in_channels,
            kernel_size, padding, dilation,
            conv_hidden_size, conv1_stride, maxpool_stride):
        super(Encoder, self).__init__()
        self.input_dims = input_dims

        # First Conv Layer
        h_out1 = conv_shape(
            input_dims, k=kernel_size,
            p=padding, s=conv1_stride, d=dilation)

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
        h_out2 = conv_shape(
            h_out1, k=kernel_size, s=maxpool_stride, d=dilation)

        # Linear Layers
        self.linear1 = nn.Linear(h_out2*h_out2*conv_hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, latent_dims)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        x = F.sigmoid(x)
        return x


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


class VariationalAutoencoder(nn.Module):
    def __init__(
            self, input_dims, latent_dims, hidden_size, in_channels,
            kernel_size=3, padding=3, dilation=1,
            conv_hidden_size=16, conv1_stride=4, maxpool_stride=1):
        super(VariationalAutoencoder, self).__init__()

        self.kernel_size = kernel_size
        self.padding = padding
        self.dilation = dilation
        self.conv_hidden_size = conv_hidden_size
        self.conv1_stride = conv1_stride
        self.maxpool_stride = maxpool_stride

        self.encoder = VariationalEncoder(
            input_dims, latent_dims, hidden_size, in_channels,
            kernel_size, padding, dilation,
            conv_hidden_size, conv1_stride, maxpool_stride)
        self.decoder = Decoder(
            input_dims, latent_dims, hidden_size, in_channels,
            kernel_size, padding, dilation,
            conv_hidden_size, conv1_stride, maxpool_stride)
        # z = torch.empty((batch_size, latent_dims))

    def forward(self, x):
        z = self.encoder(x)
        result = self.decoder(z)
        return result, z


class VariationalEncoder(nn.Module):
    def __init__(
            self, input_dims, latent_dims, hidden_size, in_channels,
            kernel_size, padding, dilation,
            conv_hidden_size, conv1_stride, maxpool_stride):
        super(VariationalEncoder, self).__init__()
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

        self.linear1 = nn.Linear(h_out2*h_out2*conv_hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, latent_dims)
        self.linear3 = nn.Linear(hidden_size, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.cuda()  # hack to get sampling on the GPU
        self.N.scale = self.N.scale.cuda()
        self.kl = 0

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))
        mu = self.linear2(x)
        sigma = torch.exp(self.linear3(x))
        z = mu + sigma*self.N.sample(mu.shape)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z
