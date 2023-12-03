from torch import nn

from .nn import RNN, ResNet, flows


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


def get_featurizer(network_type, config):
    if network_type == "resnet-gru":
        return ResNet(
            config["dim_in"],
            config.pop("dim_out", -1),
            depth=config["depth"],
            kernel_size=config.pop("kernel", 3),
            hidden_conv=config.pop("dim_conv_min", 32),
            max_hidden=config.pop("dim_conv_max", 256),
            norm=config.pop("norm", "weight_norm"),
            rnn_layer=config.pop("n_rnn", 2),
        )
    elif network_type == "resnet":
        return ResNet(
            config["dim_in"],
            config.pop("dim_out", -1),
            depth=config["depth"],
            kernel_size=config.pop("kernel", 3),
            hidden_conv=config.pop("dim_conv_min", 32),
            max_hidden=config.pop("dim_conv_max", 256),
            norm=config.pop("norm", "weight_norm"),
            rnn_layer=0,
        )
    elif network_type == "gru":
        return RNN(
            config["dim_in"],
            hidden_rnn=config["dim_hidden"],
            num_layers=config["depth"],
            num_class=config["dim_out"],
            hidden=config["dim_out"],
            dropout_rnn=0.15,
            bidirectional=False,
            rnn="GRU",
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
    modules = []
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
