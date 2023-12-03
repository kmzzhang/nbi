from torch import nn
from torch.nn.utils import weight_norm


class ResNet(nn.Module):
    def __init__(
        self,
        num_inputs,
        num_outputs=-1,
        depth=9,
        nlayer=2,
        kernel_size=3,
        hidden_conv=32,
        max_hidden=256,
        maxpool_size=2,
        rnn_layer=0,
        norm="weight_norm",
    ):
        super(type(self), self).__init__()
        h = min(max_hidden, hidden_conv * 2 ** (depth - 1))
        if num_outputs == -1:
            num_outputs = h
        self.num_outputs = num_outputs

        self.conv = ResNetBase(
            num_inputs,
            depth,
            nlayer,
            kernel_size,
            hidden_conv,
            max_hidden,
            maxpool_size,
            norm,
        )
        if rnn_layer > 0:
            self.rnn = nn.GRU(input_size=h, hidden_size=h, num_layers=rnn_layer)
        else:
            self.rnn = None
        self.linear = nn.Linear(h, num_outputs)

    def forward(self, x, aux=None):
        # N D L
        y = self.conv(x)

        if self.rnn is not None:
            y = self.rnn(y.permute(2, 0, 1))[1][-1]
        else:
            y = y.mean(dim=2)

        y = self.linear(y)
        return y


class ResNetBase(nn.Module):
    def __init__(
        self,
        num_inputs,
        depth=9,
        nlayer=2,
        kernel_size=3,
        hidden_conv=32,
        max_hidden=256,
        maxpool_size=2,
        norm="weight_norm",
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
        network = []
        network.append(ResBlock(num_inputs, hidden_conv, kernel_size, norm))
        for j in range(nlayer - 1):
            network.append(ResBlock(hidden_conv, hidden_conv, kernel_size, norm))
        for i in range(depth - 1):
            h0 = min(max_hidden, hidden_conv * 2**i)
            h = min(max_hidden, hidden_conv * 2 ** (i + 1))
            network.append(nn.MaxPool1d(maxpool_size, stride=maxpool_size))
            network.append(ResBlock(h0, h, kernel_size, norm))
            for j in range(nlayer - 1):
                network.append(ResBlock(h, h, kernel_size, norm))
        self.conv = nn.Sequential(*network)

    def forward(self, x, aux=None):
        # N D L
        return self.conv(x)


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k, norm="weight_norm"):
        super(type(self), self).__init__()
        if norm == "weight_norm":
            net = [
                weight_norm(nn.Conv1d(in_ch, out_ch, k, padding=int((k - 1) / 2))),
                nn.ReLU(),
                # nn.Dropout(dropout),
                weight_norm(nn.Conv1d(out_ch, out_ch, k, padding=int((k - 1) / 2))),
                nn.ReLU(),
                # nn.Dropout(dropout),
            ]
        else:
            net = [
                nn.Conv1d(in_ch, out_ch, k, padding=int((k - 1) / 2)),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
                # nn.Dropout(dropout),
                nn.Conv1d(out_ch, out_ch, k, padding=int((k - 1) / 2)),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
                # nn.Dropout(dropout),
            ]
        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k, norm="weight_norm"):
        super(type(self), self).__init__()
        self.conv = ConvBlock(in_ch, out_ch, k, norm)
        if in_ch != out_ch:
            self.conv0 = nn.Conv1d(in_ch, out_ch, 1)
        else:
            self.conv0 = None

    def forward(self, x):
        y = self.conv(x)
        if self.conv0 is not None:
            return y + self.conv0(x)
        return y + x
