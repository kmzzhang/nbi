from torch import nn
import torch

from .nn import RNN, ResNet, flows

class MultiFeaturizer(nn.Module):
    """
    Wraps a list of featurizers into a single module.

    Expects x to be a list/tuple of tensors, one per channel.
    Returns concatenated feature vector: [B, sum_j out_j]
    """
    def __init__(self, featurizers):
        super().__init__()
        self.featurizers = nn.ModuleList(featurizers)
        # expose num_outputs so flow can size conditioning dim
        self.num_outputs = int(sum(getattr(f, "num_outputs", 0) for f in featurizers))

    def forward(self, x):
        if not isinstance(x, (list, tuple)):
            raise TypeError(f"MultiFeaturizer expects x as list/tuple, got {type(x)}")

        if len(x) != len(self.featurizers):
            raise ValueError(f"x has {len(x)} channels but featurizers has {len(self.featurizers)}")

        feats = []
        for fj, xj in zip(self.featurizers, x):
            out = fj(xj)
            # flatten to [B, -1] if needed
            if out.ndim > 2:
                out = out.reshape(out.shape[0], -1)
            feats.append(out)
        return torch.cat(feats, dim=1)

class DataParallelFlow(nn.DataParallel):
    def __init__(self, module, device_ids=None, output_device=None, dim=0):
        super().__init__(module, device_ids=device_ids, output_device=output_device, dim=dim)

    def sample(self, x, n=1000, is_feature=False):
        if not is_feature:
            cond_vector = self.module.featurizer(x)
        else:
            cond_vector = x
        return self.module.flow.sample(num_samples=n, cond_inputs=cond_vector)


class Flow(nn.Module):
    def __init__(self, featurizer, model):
        super().__init__()
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
        sample=False,
    ):
        cond_vector = self.featurizer(x) if self.featurizer else x
        # if the channel dimension is (B, C1, C2, ..., D), reshape cond_vector into (B, -1)
        if len(cond_vector.shape) > 2:
            cond_vector = cond_vector.reshape(cond_vector.shape[0], -1)

        if sample:
            return self.flow.sample(num_samples=n, cond_inputs=cond_vector)

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
    else:
        raise ValueError(f"Unknown featurizer type: {network_type}")


def get_flow(
    featurizer,
    n_dims,
    flow_hidden,
    num_cond_inputs=None,
    num_blocks=5,
    perm_seed=0,
    clamp_0=-1,
    clamp_1=1,
    n_mog=8,
):

    # If a list/tuple of per-channel featurizers is provided, wrap them
    if isinstance(featurizer, (list, tuple)):
        featurizer = MultiFeaturizer(featurizer)

    if num_cond_inputs is None:
        if featurizer is None:
            raise ValueError("num_cond_inputs must be provided when featurizer is None")
        if not hasattr(featurizer, "num_outputs"):
            raise ValueError(
                "Could not infer num_cond_inputs: featurizer has no attribute 'num_outputs'. "
                "Pass num_cond_inputs explicitly."
            )
        num_cond_inputs = int(featurizer.num_outputs)

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
