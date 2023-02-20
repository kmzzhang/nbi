import torch
import torch.nn as nn

rnns = {"LSTM": nn.LSTM, "GRU": nn.GRU}


class RNN(nn.Module):
    def __init__(
        self,
        num_inputs,
        hidden_rnn,
        num_layers,
        num_class,
        hidden,
        dropout_rnn=0.15,
        dropout=0,
        bidirectional=False,
        rnn="GRU",
        aux=0,
    ):
        super(type(self), self).__init__()
        self.bidirectional = bidirectional
        self.aux = aux

        network = rnns[rnn]
        self.rnn = rnn
        self.encoder = network(
            input_size=num_inputs,
            hidden_size=hidden_rnn,
            num_layers=num_layers,
            dropout=dropout_rnn,
            bidirectional=bidirectional,
        )
        if bidirectional:
            hidden_rnn *= 2
        if aux != 0:
            self.dropout = nn.Dropout(dropout)
            self.linear1 = nn.Linear(hidden_rnn + aux, hidden)
            self.linear2 = nn.Linear(hidden, num_class)
            self.relu = nn.ReLU()
            self.linear = nn.Sequential(
                self.linear1, self.dropout, self.relu, self.linear2
            )
        else:
            self.linear = nn.Linear(hidden, num_class)
            self.linear = nn.Sequential(self.linear)

    def forward(self, x, aux=None):
        # N, C, L --> L, N, H0
        x = x.permute(2, 0, 1)
        code = self.encoder(x)[1] if self.rnn == "GRU" else self.encoder(x)[1][0]
        # N, H0*(2)
        if self.bidirectional:
            feature = torch.cat((code[-1], code[-2]), dim=1)
        else:
            feature = code[-1]
        if self.aux > 0:
            feature = torch.cat((feature, aux), dim=1)
        logprob = self.linear(feature)
        return logprob
