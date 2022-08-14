import torch.nn as nn
from torch import from_numpy
import torch
import numpy as np
from nbi.nn import flows, ResNetRNN, ResNetLinear, RNN

CELoss = nn.CrossEntropyLoss()

class DataParallelFlow(nn.DataParallel):
    def __init__(self, *args, **kwargs):
        super(type(self), self).__init__(*args, **kwargs)

    def sample(self, x, n=1000, is_feature=False):
        if not is_feature:
            cond_vector = self.module.featurizer(x)
        else:
            cond_vector = x
        return self.module.flow.sample(num_samples=n, cond_inputs=cond_vector)


class Flow(nn.Module):
    def __init__(self, featurizer, model):
        super(type(self), self).__init__()
        self.featurizer = featurizer
        self.flow = model

    def forward(self, x, y=None, reduce='sum', n=1000, aux=None, sample=False):
        B = x.shape[0]
        cond_vector = self.featurizer(x, aux)

        if sample:
            return self.flow.sample(num_samples=n, cond_inputs=cond_vector)

        if reduce == 'APT':
            y = y.repeat(B, 1, 1)  # B B D
            cond_vector = cond_vector.repeat(B, 1, 1)  # B B Dc
            print(y.shape, cond_vector.shape)
            neg_log_probs = -1 * self.flow.log_probs(y, cond_vector)  # B B D
            print(neg_log_probs.shape)
            neg_log_probs = neg_log_probs.sum(-1) # B B

            labels = torch.arange(B).to(neg_log_probs.device)
            print(neg_log_probs.shape)
            neg_log_probs = CELoss(neg_log_probs, labels)

        elif reduce == 'NLL':
            neg_log_probs = -1 * self.flow.log_probs(y, cond_vector)
            neg_log_probs = neg_log_probs.sum(-1, keepdim=True) # B -1
        else:
            neg_log_probs = -1 * self.flow.log_probs(y, cond_vector)

        return neg_log_probs


def get_featurizer(featurizer_type, raw_dim, num_cond_inputs, depth=9, resnet_layer=2,
                   kernel=3, resnet_hidden=32, resnet_max_hidden=256, maxpool_size=2, norm='weight_norm', dropout_rnn=0.15):
    if featurizer_type == 'resnet':
        featurizer = ResNetLinear(raw_dim, num_cond_inputs, depth=depth, nlayer=resnet_layer,
                            kernel_size=kernel, hidden_conv=resnet_hidden, max_hidden=resnet_max_hidden,
                            maxpool_size=maxpool_size, norm=norm)
    elif featurizer_type == 'rnn':
        featurizer = RNN(raw_dim, hidden_rnn=num_cond_inputs, num_layers=depth, num_class=num_cond_inputs,
                         hidden=num_cond_inputs, dropout_rnn=0.15, dropout=0,
                         bidirectional=False, rnn='GRU', aux=0)
    else:
        featurizer = ResNetRNN(raw_dim, num_cond_inputs, depth=depth, nlayer=resnet_layer, kernel_size=kernel,
                               hidden_conv=resnet_hidden, max_hidden=resnet_max_hidden, maxpool_size=maxpool_size,
                               norm=norm)
    return featurizer


def get_flow(featurizer, dim_theta, flow_hidden, num_cond_inputs, num_blocks=5,
             perm_seed=0, clamp_0=-1, clamp_1=1, n_mog=8):

    modules = list()
    MADE = flows.MADE2
    num_blocks -= 1

    for i, _ in enumerate(range(num_blocks)):
        modules += [
            flows.Shuffle(dim_theta, perm_seed + i),
            MADE(dim_theta, flow_hidden, num_cond_inputs, shift_only=False, linear_scale=False,
                 clamp_0=clamp_0, clamp_1=clamp_1),
        ]

    modules += [
        flows.Shuffle(dim_theta, perm_seed + num_blocks + 1),
        flows.MADEMOG(dim_theta, flow_hidden, num_cond_inputs, n_components=n_mog, shift_only=False,
                    linear_scale=False, clamp_0=clamp_0, clamp_1=clamp_1),
    ]
    flow = flows.FlowSequentialMOG(*modules)
    flow.init(n_mog)
    flow.set_num_inputs(dim_theta)

    for module in flow.modules():
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight)
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.data.fill_(0)

    full_model = Flow(featurizer, flow)
    return full_model

if __name__ == '__main__':
    dim_param = 3

    flow_config = {
        'flow_hidden': 512,
        'num_cond_inputs': 512,
        'num_blocks': 20,
        'perm_seed': 3,
        'n_mog': 4
    }

    resnet = get_featurizer('resnetrnn', 2, 512, depth=2)
    model = get_flow(resnet, dim_param, **flow_config)

    flow = model.flow

    # B, C, L
    # f = np.random.normal(0, 1, (7, 512))
    # f = from_numpy(f).type(torch.FloatTensor)

    # B, B + 1, D
    y = np.random.normal(0, 1, (7, 3))
    y = from_numpy(y).type(torch.FloatTensor)
    # loss = flow(y, cond_inputs=f)
    # print(loss[0].shape)

    # B, C, L
    x = np.random.normal(0, 1, (7, 2, 15))
    x = from_numpy(x).type(torch.FloatTensor)
    loss = model(x, y, reduce='apt')
    print(loss.mean().shape)
