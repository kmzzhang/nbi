import torch.nn as nn
from .nn import flows, ResNetRNN, ResNetLinear, RNN


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

    def forward(
        self,
        x,
        y=None,
        is_feature=False,
        reduce=True,
        return_entropy=False,
        n_entropy=10,
        n=1000,
        aux=None,
        sample=False,
    ):
        if sample:
            if not is_feature:
                cond_vector = self.featurizer(x, aux)
            else:
                cond_vector = x
            return self.flow.sample(num_samples=n, cond_inputs=cond_vector)

        if not is_feature:
            cond_vector = self.featurizer(x, aux)
        else:
            cond_vector = x
        self.cond_vector = cond_vector
        neg_log_probs = -1 * self.flow.log_probs(y, cond_vector)
        if reduce:
            neg_log_probs = neg_log_probs.sum(-1, keepdim=True)
        if not return_entropy:
            return neg_log_probs
        else:
            entropy = self.flow.entropy(num_samples=n_entropy, cond_inputs=cond_vector)
            return neg_log_probs, entropy


def get_featurizer(
    network_type,
    config
    # raw_dim,
    # num_cond_inputs,
    # depth=9,
    # resnet_layer=2,
    # kernel=3,
    # resnet_hidden=32,
    # resnet_max_hidden=256,
    # maxpool_size=2,
    # norm="weight_norm",
    # dropout_rnn=0.15,
):
    # if network_type == "resnet":
    #     featurizer = ResNetLinear(
    #         raw_dim,
    #         num_cond_inputs,
    #         depth=depth,
    #         nlayer=resnet_layer,
    #         kernel_size=kernel,
    #         hidden_conv=resnet_hidden,
    #         max_hidden=resnet_max_hidden,
    #         maxpool_size=maxpool_size,
    #         norm=norm,
    #     )
    # elif network_type == "rnn":
    #     featurizer = RNN(
    #         raw_dim,
    #         hidden_rnn=num_cond_inputs,
    #         num_layers=depth,
    #         num_class=num_cond_inputs,
    #         hidden=num_cond_inputs,
    #         dropout_rnn=0.15,
    #         dropout=0,
    #         bidirectional=False,
    #         rnn="GRU",
    #         aux=0,
    #     )
    # else:
    if network_type == 'sequence':
        return ResNetRNN(
            config['dim_in'],
            config['dim_out'],
            depth=config['depth'],
            kernel_size=config.pop('kernel', 3),
            hidden_conv=config.pop('dim_conv_min', 32),
            max_hidden=config.pop('dim_conv_max', 256),
        )


def get_flow(
    featurizer,
    n_dims,
    flow_hidden,
    num_cond_inputs,
    num_blocks=5,
    perm_seed=0,
    clamp_0=-1,
    clamp_1=1,
    n_mog=8,
):
    modules = list()
    MADE = flows.MADE2
    num_blocks -= 1

    for i, _ in enumerate(range(num_blocks)):
        modules += [
            flows.Shuffle(n_dims, perm_seed + i),
            MADE(
                n_dims,
                flow_hidden,
                num_cond_inputs,
                shift_only=False,
                linear_scale=False,
                clamp_0=clamp_0,
                clamp_1=clamp_1,
            ),
        ]

    modules += [
        flows.Shuffle(n_dims, perm_seed + num_blocks + 1),
        flows.MADEMOG(
            n_dims,
            flow_hidden,
            num_cond_inputs,
            n_components=n_mog,
            shift_only=False,
            linear_scale=False,
            clamp_0=clamp_0,
            clamp_1=clamp_1,
        ),
    ]
    flow = flows.FlowSequentialMOG(*modules)
    flow.init(n_mog)
    flow.set_num_inputs(n_dims)

    for module in flow.modules():
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight)
            if hasattr(module, "bias") and module.bias is not None:
                module.bias.data.fill_(0)

    full_model = Flow(featurizer, flow)
    return full_model


if __name__ == "__main__":
    dim_param = 3

    flow_config = {
        "flow_hidden": 512,
        "num_cond_inputs": 512,
        "num_blocks": 20,
        "perm_seed": 3,
        "n_mog": 4,
    }

    # nbi has different featurizers pre-defined
    # light-curve problems: use resnetrnn
    resnet = get_featurizer("resnetrnn", 1, 512, depth=6)
    model = get_flow(resnet, dim_param, **flow_config)
