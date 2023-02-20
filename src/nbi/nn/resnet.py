import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm


class classifier_custom(nn.Module):
    def __init__(
        self,
        num_inputs,
        num_class,
        nlayer,
        kernel_size,
        hidden_conv=64,
        max_hidden=256,
        mode="mean",
        norm="weight_norm",
        attention=False,
        n_head=-1,
        bottleneck=False,
    ):
        super(type(self), self).__init__()
        if type(kernel_size) == int:
            kernel_size = [kernel_size] * len(nlayer)
        network = list()
        network.append(ResBlock(num_inputs, hidden_conv, kernel_size[0], norm=norm))
        for j in range(nlayer[0] - 1):
            network.append(
                ResBlock(hidden_conv, hidden_conv, kernel_size[0], norm=norm)
            )
        for i in range(len(nlayer) - 1):  # depth will by default be 9
            h0 = min(max_hidden, hidden_conv * 2 ** max(i - 1, 0))
            h = min(max_hidden, hidden_conv * 2**i)
            network.append(nn.MaxPool1d(2, stride=2))
            network.append(ResBlock(h0, h, kernel_size[i + 1], norm=norm))
            for j in range(nlayer[i + 1] - 1):
                network.append(ResBlock(h, h, kernel_size[i + 1], norm=norm))
        self.conv = nn.Sequential(*network)

        if attention:
            assert n_head >= 1
            self.attn = nn.Conv1d(h, n_head, 1)
            self.linear = nn.Linear(h * n_head, num_class)
        else:
            self.attn = None
            self.linear = nn.Linear(h, num_class)
        self.mode = mode

    def forward(self, x, aux=None):
        # N D L
        y = self.conv(x)
        if self.attn is not None:
            w = F.softmax(self.attn(y), dim=2).unsqueeze(1)  # N 1 H L
            y = (
                (y.unsqueeze(2) * w).sum(dim=-1).reshape(y.shape[0], -1)
            )  # N D H L --> N D*H
        else:
            y = y.mean(dim=2)
        y = self.linear(y)
        return y


class ResNetLinear(nn.Module):
    def __init__(
        self,
        num_inputs,
        num_class,
        depth=9,
        nlayer=2,
        kernel_size=3,
        hidden_conv=32,
        max_hidden=256,
        mode="mean",
        norm="weight_norm",
        attention=False,
        n_head=1,
        maxpool_size=2,
    ):
        super(type(self), self).__init__()
        h = min(max_hidden, hidden_conv * 2 ** (depth - 1))

        self.conv = ResNet(
            num_inputs,
            depth,
            nlayer,
            kernel_size,
            hidden_conv,
            max_hidden,
            norm,
            maxpool_size,
        )

        self.attn = None
        if attention:
            assert n_head >= 1
            self.attn = nn.Conv1d(h, n_head, 1)
            self.linear = nn.Linear(h * n_head, num_class)
        else:
            self.linear = nn.Linear(h, num_class)
        self.mode = mode

    def forward(self, x, aux=None):
        # N D L
        y = self.conv(x)
        if self.attn is not None:
            w = F.softmax(self.attn(y), dim=2).unsqueeze(1)  # N 1 H L
            y = (
                (y.unsqueeze(2) * w).sum(dim=-1).reshape(y.shape[0], -1)
            )  # N D H L --> N D*H
        else:
            y = y.mean(dim=2)
        y = self.linear(y)
        return y


class ResNetRNN(nn.Module):
    def __init__(
        self,
        num_inputs,
        num_class,
        depth=9,
        nlayer=2,
        kernel_size=3,
        hidden_conv=32,
        max_hidden=256,
        mode="mean",
        norm="weight_norm",
        maxpool_size=2,
    ):
        super(type(self), self).__init__()
        h = min(max_hidden, hidden_conv * 2 ** (depth - 1))

        self.conv = ResNet(
            num_inputs,
            depth,
            nlayer,
            kernel_size,
            hidden_conv,
            max_hidden,
            norm,
            maxpool_size,
        )

        self.rnn = nn.GRU(input_size=h, hidden_size=h, num_layers=2)
        self.linear = nn.Linear(h, num_class)

    def forward(self, x, aux=None):
        # N D L
        y = self.conv(x)
        feature = self.rnn(y.permute(2, 0, 1))[1][-1]
        z = self.linear(feature)
        return z


class ResNet(nn.Module):
    def __init__(
        self,
        num_inputs,
        depth=9,
        nlayer=2,
        kernel_size=3,
        hidden_conv=32,
        max_hidden=256,
        norm="weight_norm",
        maxpool_size=2,
    ):
        """

            Residual convolutional network

        Parameters
        ----------
        num_inputs
        depth
        nlayer
        kernel_size
        hidden_conv
        max_hidden
        norm
        maxpool_size
        """

        super(type(self), self).__init__()
        network = list()
        network.append(ResBlock(num_inputs, hidden_conv, kernel_size, norm=norm))
        for j in range(nlayer - 1):
            network.append(ResBlock(hidden_conv, hidden_conv, kernel_size, norm=norm))
        for i in range(depth - 1):
            h0 = min(max_hidden, hidden_conv * 2**i)
            h = min(max_hidden, hidden_conv * 2 ** (i + 1))
            network.append(nn.MaxPool1d(maxpool_size, stride=maxpool_size))
            network.append(ResBlock(h0, h, kernel_size, norm=norm))
            for j in range(nlayer - 1):
                network.append(ResBlock(h, h, kernel_size, norm=norm))
        self.conv = nn.Sequential(*network)

    def forward(self, x, aux=None):
        # N D L
        return self.conv(x)


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k, dropout=0, bottleneck=False):
        super(type(self), self).__init__()
        net = [
            weight_norm(nn.Conv1d(in_ch, out_ch, k, padding=int((k - 1) / 2))),
            nn.ReLU(),
            nn.Dropout(dropout),
            weight_norm(nn.Conv1d(out_ch, out_ch, k, padding=int((k - 1) / 2))),
            nn.ReLU(),
            nn.Dropout(dropout),
        ]
        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k, dropout=0, norm="weight_norm"):
        super(type(self), self).__init__()
        self.conv = ConvBlock(in_ch, out_ch, k, dropout)
        if in_ch != out_ch:
            self.conv0 = nn.Conv1d(in_ch, out_ch, 1)
        else:
            self.conv0 = None

    def forward(self, x):
        y = self.conv(x)
        if self.conv0 is not None:
            return y + self.conv0(x)
        return y + x


"""
class classifier(nn.Module):

    def __init__(
        self,
        num_inputs,
        num_class,
        depth=9,
        nlayer=2,
        kernel_size=3,
        hidden_conv=32,
        max_hidden=256,
        mode='mean',
        norm='weight_norm',
        attention=False,
        n_head=1,
        maxpool_size=2
    ):
        super(type(self), self).__init__()

        if attention:
            assert mode == 'mean'

        total_hidden = min(max_hidden, hidden_conv * 2 ** (depth - 1))

        self.resnet = ResNet(
            num_inputs,
            depth,
            nlayer,
            kernel_size,
            hidden_conv,
            max_hidden,
            norm,
            maxpool_size
        )

        self.attn = None
        if attention:
            self.attn = nn.Conv1d(total_hidden, n_head, 1)

        if mode == 'rnn':
            pooling_layer = nn.GRU(
                input_size=total_hidden * n_head,
                hidden_size=total_hidden * n_head,
                num_layers=2
            )
        else:
            pooling_layer = GlobalPooling1D(-1)

        self.gather = nn.Sequential(
            pooling_layer,
            nn.Linear(total_hidden * n_head, num_class)
        )

    def forward(self, x, aux=None):
        length = x.size(-1)
        feature = self.resnet(x, aux)

        if self.attn is not None:
            logits = self.attn(feature)
            weights = F.softmax(logits, dim=2)
            feature = feature.unsqueeze(2) * weights.unsqueeze(1)  # (N D 1 L) * (N 1 H L)
            feature = feature.reshape(feature.shape[0], -1, length)  # N D H L --> N D*H L

        feature = self.gather(feature)
        return feature
"""
