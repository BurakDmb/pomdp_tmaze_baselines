import torch
import torch.nn.functional as F
from torch import nn
from torchsummary import summary


# k: kernel_size, p: padding, s: stride, d: dilation.
def conv_shape(x, k=1, p=0, s=1, d=1):
    return int((x + 2*p - d*(k - 1) - 1)/s + 1)


class ConvBinaryAutoencoder(nn.Module):
    def __init__(
            self, input_dims, latent_dims, hidden_size, in_channels, **kwargs):
        super(ConvBinaryAutoencoder, self).__init__()

        self.latent_dims = latent_dims
        self.kernel_size = kwargs.get('kernel_size', 7)
        self.padding = kwargs.get('padding', 0)
        self.dilation = kwargs.get('dilation', 1)
        self.conv_hidden_size = kwargs.get('conv_hidden_size', 128)
        self.conv1_stride = kwargs.get('conv1_stride', 1)
        self.maxpool_stride = kwargs.get('maxpool_stride', 1)

        self.encoder = EncoderConv(
            input_dims, latent_dims, hidden_size, in_channels,
            self.kernel_size, self.padding, self.dilation,
            self.conv_hidden_size, self.conv1_stride, self.maxpool_stride)
        self.decoder = DecoderConv(
            input_dims, latent_dims, hidden_size, in_channels,
            self.kernel_size, self.padding, self.dilation,
            self.conv_hidden_size, self.conv1_stride, self.maxpool_stride)
        summary(self.to("cuda"), (in_channels, input_dims, input_dims))

    def forward(self, x):
        # Creating a uniform random variable with U(-0.3, 0.3)
        rand_tensor = torch.rand(self.latent_dims, device=x.device)*0.6 - 0.3
        z = self.encoder(x) + rand_tensor
        result = self.decoder(z)
        return result, z


class ConvAutoencoder(nn.Module):
    def __init__(
            self, input_dims, latent_dims, hidden_size, in_channels, **kwargs):
        super(ConvAutoencoder, self).__init__()

        self.kernel_size = kwargs.get('kernel_size', 7)
        self.padding = kwargs.get('padding', 0)
        self.dilation = kwargs.get('dilation', 1)
        self.conv_hidden_size = kwargs.get('conv_hidden_size', 128)
        self.conv1_stride = kwargs.get('conv1_stride', 1)
        self.maxpool_stride = kwargs.get('maxpool_stride', 1)

        self.encoder = EncoderConv(
            input_dims, latent_dims, hidden_size, in_channels,
            self.kernel_size, self.padding, self.dilation,
            self.conv_hidden_size, self.conv1_stride, self.maxpool_stride)
        self.decoder = DecoderConv(
            input_dims, latent_dims, hidden_size, in_channels,
            self.kernel_size, self.padding, self.dilation,
            self.conv_hidden_size, self.conv1_stride, self.maxpool_stride)
        summary(self.to("cuda"), (in_channels, input_dims, input_dims))

    def forward(self, x):
        z = self.encoder(x)
        result = self.decoder(z)
        return result, z


class ConvVariationalAutoencoder(nn.Module):
    def __init__(
            self, input_dims, latent_dims, hidden_size, in_channels,
            **kwargs):
        super(ConvVariationalAutoencoder, self).__init__()

        self.kernel_size = kwargs.get('kernel_size', 7)
        self.padding = kwargs.get('padding', 0)
        self.dilation = kwargs.get('dilation', 1)
        self.conv_hidden_size = kwargs.get('conv_hidden_size', 128)
        self.conv1_stride = kwargs.get('conv1_stride', 1)
        self.maxpool_stride = kwargs.get('maxpool_stride', 1)

        self.encoder = VariationalEncoderConv(
            input_dims, latent_dims, hidden_size, in_channels,
            self.kernel_size, self.padding, self.dilation,
            self.conv_hidden_size, self.conv1_stride, self.maxpool_stride)
        self.decoder = DecoderConv(
            input_dims, latent_dims, hidden_size, in_channels,
            self.kernel_size, self.padding, self.dilation,
            self.conv_hidden_size, self.conv1_stride, self.maxpool_stride)
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
        # Conv hidden size is the conv channel hidden channel size.
        self.conv_hidden_size = conv_hidden_size

        self.shape2d_1 = conv_shape(
            self.input_dims, k=kernel_size,
            p=padding, s=conv1_stride, d=dilation)
        self.shape2d_2 = conv_shape(
            self.shape2d_1, k=kernel_size,
            p=padding, s=conv1_stride, d=dilation)

        self.flatten_size = in_channels*self.shape2d_2*self.shape2d_2

        # encoder
        self.enc2d_1 = nn.Conv2d(
            in_channels, conv_hidden_size, kernel_size=kernel_size,
            padding=padding, stride=conv1_stride, dilation=dilation)
        self.enc2d_2 = nn.Conv2d(
            conv_hidden_size, in_channels, kernel_size=kernel_size,
            padding=padding, stride=conv1_stride, dilation=dilation)

        self.linear1 = nn.Linear(self.flatten_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, latent_dims)

    def forward(self, x):
        x = F.relu(self.enc2d_1(x))
        x = F.relu(self.enc2d_2(x))

        x = x.flatten(start_dim=1, end_dim=3)
        x = F.relu(self.linear1(x))
        x = torch.sigmoid(self.linear2(x))
        return x


class DecoderConv(nn.Module):
    def __init__(
            self, input_dims, latent_dims, hidden_size, in_channels,
            kernel_size, padding, dilation,
            conv_hidden_size, conv1_stride, maxpool_stride):
        super(DecoderConv, self).__init__()
        self.input_dims = input_dims
        self.hidden_size = hidden_size
        self.in_channels = in_channels

        # Conv hidden size is the conv channel hidden channel size.
        self.conv_hidden_size = conv_hidden_size

        #
        self.shape2d_1 = conv_shape(
            self.input_dims, k=kernel_size,
            p=padding, s=conv1_stride, d=dilation)
        self.shape2d_2 = conv_shape(
            self.shape2d_1, k=kernel_size,
            p=padding, s=conv1_stride, d=dilation)

        self.flatten_size = in_channels*self.shape2d_2*self.shape2d_2

        # decode
        self.linear1 = nn.Linear(latent_dims, hidden_size)
        self.linear2 = nn.Linear(hidden_size, self.flatten_size)

        self.dec2d_1 = nn.ConvTranspose2d(
            in_channels, conv_hidden_size, kernel_size=kernel_size,
            padding=padding, stride=conv1_stride, dilation=dilation)
        self.dec2d_2 = nn.ConvTranspose2d(
            conv_hidden_size, in_channels, kernel_size=kernel_size,
            padding=padding, stride=conv1_stride, dilation=dilation)

    def forward(self, z):

        z = F.relu(self.linear1(z))
        z = F.relu(self.linear2(z))
        z = z.unflatten(1, (self.in_channels, self.shape2d_2, self.shape2d_2))
        z = F.relu(self.dec2d_1(z))
        z = torch.sigmoid(self.dec2d_2(z))
        return z


class Autoencoder(nn.Module):
    def __init__(
            self, input_dims, latent_dims, hidden_size, in_channels,
            **kwargs):
        super(Autoencoder, self).__init__()

        self.kernel_size = kwargs.get('kernel_size', 7)
        self.padding = kwargs.get('padding', 0)
        self.dilation = kwargs.get('dilation', 1)
        self.conv_hidden_size = kwargs.get('conv_hidden_size', 128)
        self.conv1_stride = kwargs.get('conv1_stride', 1)
        self.maxpool_stride = kwargs.get('maxpool_stride', 1)

        self.encoder = Encoder(
            input_dims, latent_dims, hidden_size, in_channels,
            self.kernel_size, self.padding, self.dilation,
            self.conv_hidden_size, self.conv1_stride, self.maxpool_stride)
        self.decoder = Decoder(
            input_dims, latent_dims, hidden_size, in_channels,
            self.kernel_size, self.padding, self.dilation,
            self.conv_hidden_size, self.conv1_stride, self.maxpool_stride)
        summary(self.to("cuda"), (in_channels, input_dims, input_dims))

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

        # Linear Layers
        self.linear1 = nn.Linear(
            input_dims*input_dims*in_channels, hidden_size)
        self.linear2 = nn.Linear(hidden_size, latent_dims)

    def forward(self, x_in):
        x = torch.flatten(x_in, start_dim=1)
        x = F.relu(self.linear1(x))
        x = torch.sigmoid(self.linear2(x))
        return x


class Decoder(nn.Module):
    def __init__(
            self, input_dims, latent_dims, hidden_size, in_channels,
            kernel_size, padding, dilation,
            conv_hidden_size, conv1_stride, maxpool_stride):
        super(Decoder, self).__init__()
        self.input_dims = input_dims
        self.hidden_size = hidden_size
        self.in_channels = in_channels

        self.linear1 = nn.Linear(latent_dims, hidden_size)
        self.linear2 = nn.Linear(
            hidden_size, in_channels*input_dims*input_dims)

    def forward(self, z_in):
        z = F.relu(self.linear1(z_in))
        z = torch.sigmoid(self.linear2(z))
        z = z.unflatten(
            1, (self.in_channels, self.input_dims, self.input_dims))
        return z


class VariationalAutoencoder(nn.Module):
    def __init__(
            self, input_dims, latent_dims, hidden_size, in_channels,
            **kwargs):
        super(VariationalAutoencoder, self).__init__()

        self.kernel_size = kwargs.get('kernel_size', 7)
        self.padding = kwargs.get('padding', 0)
        self.dilation = kwargs.get('dilation', 1)
        self.conv_hidden_size = kwargs.get('conv_hidden_size', 128)
        self.conv1_stride = kwargs.get('conv1_stride', 1)
        self.maxpool_stride = kwargs.get('maxpool_stride', 1)

        self.encoder = VariationalEncoder(
            input_dims, latent_dims, hidden_size, in_channels,
            self.kernel_size, self.padding, self.dilation,
            self.conv_hidden_size, self.conv1_stride, self.maxpool_stride)
        self.decoder = Decoder(
            input_dims, latent_dims, hidden_size, in_channels,
            self.kernel_size, self.padding, self.dilation,
            self.conv_hidden_size, self.conv1_stride, self.maxpool_stride)
        summary(self.to("cuda"), (in_channels, input_dims, input_dims))
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
        self.in_channels = in_channels

        # Linear Layers
        self.linear1 = nn.Linear(
            input_dims*input_dims*in_channels, hidden_size)
        self.linear2_mean = nn.Linear(hidden_size, latent_dims)
        self.linear2_variance = nn.Linear(hidden_size, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.cuda()  # hack to get sampling on the GPU
        self.N.scale = self.N.scale.cuda()
        self.kl = 0

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))
        mu = self.linear2_mean(x)
        sigma = torch.exp(self.linear2_variance(x))

        x = mu + sigma*self.N.sample(mu.shape)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum() / (
                self.in_channels *
                self.input_dims *
                self.input_dims *
                x.shape[0])
        return x


class VariationalEncoderConv(nn.Module):
    def __init__(
            self, input_dims, latent_dims, hidden_size, in_channels,
            kernel_size, padding, dilation,
            conv_hidden_size, conv1_stride, maxpool_stride):
        super(VariationalEncoderConv, self).__init__()
        self.input_dims = input_dims
        self.in_channels = in_channels
        self.conv_hidden_size = conv_hidden_size

        self.shape2d_1 = conv_shape(
            self.input_dims, k=kernel_size,
            p=padding, s=conv1_stride, d=dilation)
        self.shape2d_2 = conv_shape(
            self.shape2d_1, k=kernel_size,
            p=padding, s=conv1_stride, d=dilation)

        self.flatten_size = in_channels*self.shape2d_2*self.shape2d_2

        # encoder
        self.enc2d_1 = nn.Conv2d(
            in_channels, conv_hidden_size, kernel_size=kernel_size,
            padding=padding, stride=conv1_stride, dilation=dilation)
        self.enc2d_2 = nn.Conv2d(
            conv_hidden_size, in_channels, kernel_size=kernel_size,
            padding=padding, stride=conv1_stride, dilation=dilation)

        self.linear1 = nn.Linear(self.flatten_size, hidden_size)
        self.linear2_mean = nn.Linear(hidden_size, latent_dims)
        self.linear2_variance = nn.Linear(hidden_size, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.cuda()  # hack to get sampling on the GPU
        self.N.scale = self.N.scale.cuda()
        self.kl = 0

    def forward(self, x):
        x = F.relu(self.enc2d_1(x))
        x = F.relu(self.enc2d_2(x))

        x = x.flatten(start_dim=1, end_dim=3)
        x = F.relu(self.linear1(x))

        mu = self.linear2_mean(x)
        sigma = torch.exp(self.linear2_variance(x))
        x = mu + sigma*self.N.sample(mu.shape)

        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum() / (
                self.in_channels *
                self.input_dims *
                self.input_dims *
                x.shape[0])
        return x


if __name__ == "__main__":
    # Print summaries of autoencoders
    ae1 = Autoencoder(
        input_dims=48, latent_dims=10,
        hidden_size=128, in_channels=3,
        conv_hidden_size=128)
    ae2 = VariationalAutoencoder(
        input_dims=48, latent_dims=10,
        hidden_size=128, in_channels=3,
        conv_hidden_size=128)
    ae3 = ConvAutoencoder(
        input_dims=48, latent_dims=10,
        hidden_size=128, in_channels=3,
        conv_hidden_size=128)
    ae4 = ConvVariationalAutoencoder(
        input_dims=48, latent_dims=10,
        hidden_size=128, in_channels=3,
        conv_hidden_size=128)
    ae5 = ConvBinaryAutoencoder(
        input_dims=48, latent_dims=10,
        hidden_size=128, in_channels=3,
        conv_hidden_size=128)
