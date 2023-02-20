# Adapted from :https://github.com/kamenbliznashki/normalizing_flows/blob/master/maf.py

import math

import numpy as np
import torch
import torch.nn as nn
import torch.distributions as D
import torch.nn.functional as F


def get_mask(in_features, out_features, in_flow_features, mask_type=None):
    """
    mask_type: input | None | output

    See Figure 1 for a better illustration:
    https://arxiv.org/pdf/1502.03509.pdf
    """
    if mask_type == "input":
        in_degrees = torch.arange(in_features) % in_flow_features
    else:
        in_degrees = torch.arange(in_features) % (in_flow_features - 1)

    if mask_type == "output":
        out_degrees = torch.arange(out_features) % in_flow_features - 1
    else:
        out_degrees = torch.arange(out_features) % (in_flow_features - 1)

    return (out_degrees.unsqueeze(-1) >= in_degrees.unsqueeze(0)).float()


class MaskedLinear(nn.Module):
    def __init__(
        self, in_features, out_features, mask, cond_in_features=None, bias=True
    ):
        super(MaskedLinear, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        if cond_in_features is not None:
            self.cond_linear = nn.Linear(cond_in_features, out_features, bias=False)

        self.register_buffer("mask", mask)

    def forward(self, inputs, cond_inputs=None):
        output = F.linear(inputs, self.linear.weight * self.mask, self.linear.bias)
        if cond_inputs is not None:
            cond = self.cond_linear(cond_inputs)
            output = output + cond
        return output


nn.MaskedLinear = MaskedLinear


def create_masks(
    input_size, hidden_size, n_hidden, input_order="sequential", input_degrees=None
):
    # MADE paper sec 4:
    # degrees of connections between layers -- ensure at most in_degree - 1 connections
    degrees = []

    # set input degrees to what is provided in args (the flipped order of the previous layer in a stack of mades);
    # else init input degrees based on strategy in input_order (sequential or random)
    if input_order == "sequential":
        degrees += (
            [torch.arange(input_size)] if input_degrees is None else [input_degrees]
        )
        for _ in range(n_hidden + 1):
            degrees += [torch.arange(hidden_size) % (input_size - 1)]
        degrees += (
            [torch.arange(input_size) % input_size - 1]
            if input_degrees is None
            else [input_degrees % input_size - 1]
        )

    elif input_order == "random":
        degrees += (
            [torch.randperm(input_size)] if input_degrees is None else [input_degrees]
        )
        for _ in range(n_hidden + 1):
            min_prev_degree = min(degrees[-1].min().item(), input_size - 1)
            degrees += [torch.randint(min_prev_degree, input_size, (hidden_size,))]
        min_prev_degree = min(degrees[-1].min().item(), input_size - 1)
        degrees += (
            [torch.randint(min_prev_degree, input_size, (input_size,)) - 1]
            if input_degrees is None
            else [input_degrees - 1]
        )

    # construct masks
    masks = []
    for d0, d1 in zip(degrees[:-1], degrees[1:]):
        masks += [(d1.unsqueeze(-1) >= d0.unsqueeze(0)).float()]

    return masks, degrees[0]


class MADEMOG(nn.Module):
    """Mixture of Gaussians MADE"""

    def __init__(
        self,
        input_size,
        hidden_size,
        cond_label_size=None,
        n_components=5,
        n_hidden=1,
        activation="relu",
        input_order="sequential",
        input_degrees=None,
        clamp_0=-2,
        clamp_1=3,
        shift_only=False,
        linear_scale=False,
    ):
        """
        Args:
            n_components -- scalar; number of gauassian components in the mixture
            input_size -- scalar; dim of inputs
            hidden_size -- scalar; dim of hidden layers
            n_hidden -- scalar; number of hidden layers
            activation -- str; activation function to use
            input_order -- str or tensor; variable order for creating the autoregressive masks (sequential|random)
                            or the order flipped from the previous layer in a stack of mades
            conditional -- bool; whether model is conditional
        """
        super().__init__()
        self.n_components = n_components
        self.clamp_0 = clamp_0 if not linear_scale else np.exp(clamp_0)
        self.clamp_1 = clamp_1 if not linear_scale else np.exp(clamp_1)

        # base distribution for calculation of log prob under the model
        self.register_buffer("base_dist_mean", torch.zeros(input_size))
        self.register_buffer("base_dist_var", torch.ones(input_size))

        # create masks
        masks, self.input_degrees = create_masks(
            input_size, hidden_size, n_hidden, input_order, input_degrees
        )

        # setup activation
        if activation == "relu":
            activation_fn = nn.ReLU()
        elif activation == "tanh":
            activation_fn = nn.Tanh()
        else:
            raise ValueError("Check activation function.")

        # construct model
        self.net_input = MaskedLinear(
            input_size, hidden_size, masks[0], cond_label_size
        )
        self.net = []
        for m in masks[1:-1]:
            self.net += [activation_fn, MaskedLinear(hidden_size, hidden_size, m)]
        self.net += [
            activation_fn,
            MaskedLinear(
                hidden_size,
                n_components * 3 * input_size,
                masks[-1].repeat(n_components * 3, 1),
            ),
        ]
        self.net = nn.Sequential(*self.net)

    def forward(self, inputs, cond_inputs=None, mode="direct"):
        """

        Parameters
        ----------
        inputs: of shape [N, L] if direct
                of shape [N, C, L]
        cond_inputs: [N, C]
        mode

        Returns
        -------
        [N, C, L] if direct
        [N, L] otherwise
        """
        if mode == "direct":
            x = inputs
            y = cond_inputs
            # shapes
            N, L = x.shape
            C = self.n_components
            m, loga, logr = (
                self.net(self.net_input(x, y)).view(N, C, 3 * L).chunk(chunks=3, dim=-1)
            )  # 3 x (N, C, L)
            loga = torch.clamp(loga, self.clamp_0, self.clamp_1)

            # copy x into C components
            x = x.repeat(1, C).view(N, C, L)  # out (N, C, L)
            u = (x - m) * torch.exp(-loga)  # out (N, C, L)
            log_abs_det_jacobian = -loga  # out (N, C, L)
            # normalize cluster responsibilities
            self.logr = logr - logr.logsumexp(1, keepdim=True)  # out (N, C, L)
            log_abs_det_jacobian += self.logr

            return u, log_abs_det_jacobian
        else:
            u = inputs
            y = cond_inputs  # B, D
            # shapes
            if len(u.shape) == 3:  # u.shape == B, C, L
                u = u.unsqueeze(0)
            N, B, C, L = u.shape
            # init output
            x = torch.zeros(N, B, L).to(u.device)
            for i in self.input_degrees:
                outputs = self.net(self.net_input(x, y))
                outputs = outputs.view(N, B, C, 3 * L)
                m, loga, logr = outputs.chunk(chunks=3, dim=-1)  # (N, B, C, L)

                loga = torch.clamp(loga, self.clamp_0, self.clamp_1)

                # normalize cluster responsibilities and sample cluster assignments from a categorical dist
                logr = logr - logr.logsumexp(-2, keepdim=True)  # out (N, B, C, L)
                z = (
                    D.Categorical(logits=logr[..., i]).sample().unsqueeze(-1)
                )  # out (N, B, 1)

                u_z = torch.gather(u[..., i], -1, z).reshape(N, B)  # out (N, B)
                m_z = torch.gather(m[..., i], -1, z).reshape(N, B)  # out (N, B)
                loga_z = torch.gather(loga[..., i], -1, z).reshape(N, B)

                x[..., i] = u_z * torch.exp(loga_z) + m_z
            log_abs_det_jacobian = loga
            return x, log_abs_det_jacobian.sum(1).sum(-1, keepdim=True)


class MADE(nn.Module):
    """An implementation of MADE
    (https://arxiv.org/abs/1502.03509).
    """

    def __init__(
        self,
        num_inputs,
        num_hidden,
        num_cond_inputs=None,
        act="relu",
        pre_exp_tanh=False,
        shift_only=False,
        linear_scale=False,
        clamp_0=-2,
        clamp_1=3,
    ):
        super(MADE, self).__init__()

        self.linear_scale = linear_scale
        activations = {"relu": nn.ReLU, "sigmoid": nn.Sigmoid, "tanh": nn.Tanh}
        act_func = activations[act]
        self.shift_only = shift_only
        output_dims = num_inputs if shift_only else num_inputs * 2

        input_mask = get_mask(num_inputs, num_hidden, num_inputs, mask_type="input")
        hidden_mask = get_mask(num_hidden, num_hidden, num_inputs)
        output_mask = get_mask(num_hidden, output_dims, num_inputs, mask_type="output")
        self.clamp_0 = clamp_0 if not linear_scale else np.exp(clamp_0)
        self.clamp_1 = clamp_1 if not linear_scale else np.exp(clamp_1)
        self.joiner = nn.MaskedLinear(
            num_inputs, num_hidden, input_mask, num_cond_inputs
        )

        self.trunk = nn.Sequential(
            act_func(),
            nn.MaskedLinear(num_hidden, num_hidden, hidden_mask),
            act_func(),
            nn.MaskedLinear(num_hidden, output_dims, output_mask),
        )

    def forward(self, inputs, cond_inputs=None, mode="direct"):
        if mode == "direct":
            if self.shift_only:
                h = self.joiner(inputs, cond_inputs)
                m = self.trunk(h)
                u = inputs - m
                return u, torch.zeros_like(m).sum(-1, keepdim=True)
            else:
                h = self.joiner(inputs, cond_inputs)
                m, a = self.trunk(h).chunk(2, -1)
                a = torch.clamp(a, self.clamp_0, self.clamp_1)
                if len(inputs.shape) == 3:
                    a = a.unsqueeze(1)
                    m = m.unsqueeze(1)
                if self.linear_scale:
                    u = (inputs - m) / a
                    a[a < 0] = -a
                    a = torch.log(a)
                    return u, a.sum(-1, keepdim=True)
                else:
                    u = (inputs - m) * torch.exp(-a)
                    return u, -a.sum(-1, keepdim=True)

        else:
            x = torch.zeros_like(inputs)
            for i_col in range(inputs.shape[-1]):
                if self.shift_only:
                    h = self.joiner(x, cond_inputs)
                    m = self.trunk(h)
                    x[:, i_col] = inputs[:, i_col] + m[:, i_col]
                elif self.linear_scale:
                    h = self.joiner(x, cond_inputs)
                    m, a = self.trunk(h).chunk(2, -1)
                    a = torch.clamp(a, self.clamp_0, self.clamp_1)
                    x[:, i_col] = inputs[:, i_col] * a[:, i_col] + m[:, i_col]
                else:
                    h = self.joiner(x, cond_inputs)
                    m, a = self.trunk(h).chunk(2, -1)
                    a = torch.clamp(a, self.clamp_0, self.clamp_1)
                    x.transpose(0, -1)[i_col] = (
                        inputs.transpose(0, -1)[i_col]
                        * torch.exp(a.transpose(0, -1)[i_col])
                        + m.transpose(0, -1)[i_col]
                    )

            if self.shift_only:
                return x, torch.zeros_like(m).sum(-1, keepdim=True)
            elif self.linear_scale:
                a = torch.log(a)
                return x, -a.sum(-1, keepdim=True)
            else:
                return x, -a.sum(-1, keepdim=True)


class MADE2(nn.Module):
    """An implementation of MADE
    (https://arxiv.org/abs/1502.03509).
    """

    def __init__(
        self,
        num_inputs,
        num_hidden,
        num_cond_inputs=None,
        act="relu",
        pre_exp_tanh=False,
        shift_only=False,
        linear_scale=False,
        clamp_0=-2,
        clamp_1=3,
    ):
        super(MADE2, self).__init__()

        self.linear_scale = linear_scale
        activations = {"relu": nn.ReLU, "sigmoid": nn.Sigmoid, "tanh": nn.Tanh}
        act_func = activations[act]
        self.shift_only = shift_only
        output_dims = num_inputs if shift_only else num_inputs * 2

        input_mask = get_mask(num_inputs, num_hidden, num_inputs, mask_type="input")
        hidden_mask = get_mask(num_hidden, num_hidden, num_inputs)
        output_mask = get_mask(num_hidden, output_dims, num_inputs, mask_type="output")
        self.clamp_0 = clamp_0 if not linear_scale else np.exp(clamp_0)
        self.clamp_1 = clamp_1 if not linear_scale else np.exp(clamp_1)
        self.joiner = nn.MaskedLinear(
            num_inputs, num_hidden, input_mask, num_cond_inputs
        )

        self.trunk = nn.Sequential(
            act_func(),
            nn.MaskedLinear(num_hidden, num_hidden, hidden_mask),
            act_func(),
            nn.MaskedLinear(num_hidden, output_dims, output_mask),
        )

    def forward(self, inputs, cond_inputs=None, mode="direct"):
        if mode == "direct":
            # print(inputs[0])
            if self.shift_only:
                h = self.joiner(inputs, cond_inputs)
                m = self.trunk(h)
                u = inputs - m
                return u, torch.zeros_like(m).sum(-1, keepdim=True)
            else:
                h = self.joiner(inputs, cond_inputs)
                m, a = self.trunk(h).chunk(2, -1)
                a = torch.clamp(a, self.clamp_0, self.clamp_1)
                if len(inputs.shape) == 3:
                    a = a.unsqueeze(1)
                    m = m.unsqueeze(1)
                if self.linear_scale:
                    u = (inputs - m) / a
                    a[a < 0] = -a
                    a = torch.log(a)
                    return u, -a.sum(-1, keepdim=True)
                else:
                    u = (inputs - m) * torch.exp(-a)
                    return u, -a
                y = torch.exp(log_gamma) * inputs + beta
                return y, log_gamma

        else:
            x = torch.zeros_like(inputs)
            for i_col in range(inputs.shape[-1]):
                h = self.joiner(x, cond_inputs)
                m, a = self.trunk(h).chunk(2, -1)
                a = torch.clamp(a, self.clamp_0, self.clamp_1)
                x[..., i_col] = (
                    inputs[..., i_col] * torch.exp(a[..., i_col]) + m[..., i_col]
                )

            return x, a


class Sigmoid(nn.Module):
    def __init__(self):
        super(Sigmoid, self).__init__()

    def forward(self, inputs, cond_inputs=None, mode="direct"):
        if mode == "direct":
            s = torch.sigmoid
            return s(inputs), torch.log(s(inputs) * (1 - s(inputs))).sum(
                -1, keepdim=True
            )
        else:
            return torch.log(inputs / (1 - inputs)), -torch.log(
                inputs - inputs**2
            ).sum(-1, keepdim=True)


class Logit(Sigmoid):
    def __init__(self):
        super(Logit, self).__init__()

    def forward(self, inputs, cond_inputs=None, mode="direct"):
        if mode == "direct":
            return super(Logit, self).forward(inputs, "inverse")
        else:
            return super(Logit, self).forward(inputs, "direct")


class BatchNormFlow(nn.Module):
    def __init__(self, num_inputs, momentum=0.0, eps=1e-5):
        super(BatchNormFlow, self).__init__()

        self.log_gamma = nn.Parameter(torch.zeros(num_inputs))
        self.beta = nn.Parameter(torch.zeros(num_inputs))
        self.momentum = momentum
        self.eps = eps
        self.count = 0
        self.num_inputs = num_inputs

        self.register_buffer("running_mean", torch.zeros(num_inputs))
        self.register_buffer("running_var", torch.zeros(num_inputs))

    def reset(self):
        self.count = 0

    def forward(self, inputs, cond_inputs=None, mode="direct"):
        if mode == "direct":
            if self.training:
                n_batch = inputs.shape[0]
                mean = inputs.mean(0)
                var = (inputs - mean).pow(2).mean(0) + self.eps
                self._add_stats(mean, var, n_batch)
            else:
                mean = self.running_mean
                var = self.running_var

            x_hat = (inputs - mean) / var.sqrt()
            y = torch.exp(self.log_gamma) * x_hat + self.beta
            return y, self.log_gamma - 0.5 * torch.log(var)

        else:
            x_hat = (inputs - self.beta) / torch.exp(self.log_gamma)
            y = x_hat * self.running_var.sqrt() + self.running_mean

            return y, -self.log_gamma + 0.5 * torch.log(self.running_var)

    def _add_stats(self, mean, var, n_batch):
        self.running_mean.mul_(self.count / (self.count + n_batch))
        self.running_var.mul_(self.count / (self.count + n_batch))

        self.running_mean.add_(mean.data * n_batch / (self.count + n_batch))
        self.running_var.add_(var.data * n_batch / (self.count + n_batch))
        self.count += n_batch


class AffineShift(nn.Module):
    def __init__(self, num_inputs, dim_cond, dim_hidden):
        super(AffineShift, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_cond, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, num_inputs * 2),
        )

    def forward(self, inputs, cond_inputs=None, mode="direct"):
        # log_gamma, beta = self.net(
        #     torch.cat((inputs, cond_inputs), dim=-1)
        # ).chunk(2, -1)

        log_gamma, beta = self.net(cond_inputs).chunk(2, -1)

        if mode == "direct":
            y = torch.exp(log_gamma) * inputs + beta
            return y, log_gamma

        else:
            inputs = (inputs - beta) / torch.exp(log_gamma)
            return inputs, -log_gamma


class Reverse(nn.Module):
    """An implementation of a reversing layer from
    Density estimation using Real NVP
    (https://arxiv.org/abs/1605.08803).
    """

    def __init__(self, num_inputs):
        super(Reverse, self).__init__()
        self.perm = np.array(np.arange(0, num_inputs)[::-1])
        self.inv_perm = np.argsort(self.perm)

    def forward(self, inputs, cond_inputs=None, mode="direct"):
        if mode == "direct":
            return inputs[:, self.perm], torch.zeros(
                inputs.size(0), 1, device=inputs.device
            )
        else:
            return inputs[:, self.inv_perm], torch.zeros(
                inputs.size(0), 1, device=inputs.device
            )


class Shuffle(nn.Module):
    def __init__(self, num_inputs, seed=0):
        super(Shuffle, self).__init__()
        np.random.seed(seed)
        self.perm = np.array(np.arange(0, num_inputs))
        np.random.shuffle(self.perm)
        self.inv_perm = np.argsort(self.perm)

    def forward(self, inputs, cond_inputs=None, mode="direct"):
        if mode == "direct":
            inputs = torch.transpose(inputs, 0, -1)[self.perm]
            inputs = torch.transpose(inputs, 0, -1)
            return inputs, torch.zeros(*inputs.shape[:-1], 1, device=inputs.device)
        else:
            inputs = torch.transpose(inputs, 0, -1)[self.inv_perm]
            inputs = torch.transpose(inputs, 0, -1)
            return inputs, torch.zeros(*inputs.shape[:-1], 1, device=inputs.device)


class ShuffleMOG(nn.Module):
    def __init__(self, num_inputs, seed=0):
        super(ShuffleMOG, self).__init__()
        np.random.seed(seed)
        self.perm = np.array(np.arange(0, num_inputs))
        np.random.shuffle(self.perm)
        self.inv_perm = np.argsort(self.perm)

    def forward(self, inputs, cond_inputs=None, mode="direct"):
        if mode == "direct":
            return inputs[:, :, self.perm], torch.zeros(
                inputs.size(0), 1, device=inputs.device
            )
        else:
            return inputs[:, :, self.inv_perm], torch.zeros(
                inputs.size(0), 1, device=inputs.device
            )


class CouplingLayer(nn.Module):
    """An implementation of a coupling layer
    from RealNVP (https://arxiv.org/abs/1605.08803).
    """

    def __init__(
        self,
        num_inputs,
        num_hidden,
        mask,
        num_cond_inputs=None,
        s_act="tanh",
        t_act="relu",
    ):
        super(CouplingLayer, self).__init__()

        self.num_inputs = num_inputs
        self.mask = mask

        activations = {"relu": nn.ReLU, "sigmoid": nn.Sigmoid, "tanh": nn.Tanh}
        s_act_func = activations[s_act]
        t_act_func = activations[t_act]

        if num_cond_inputs is not None:
            total_inputs = num_inputs + num_cond_inputs
        else:
            total_inputs = num_inputs

        self.scale_net = nn.Sequential(
            nn.Linear(total_inputs, num_hidden),
            s_act_func(),
            nn.Linear(num_hidden, num_hidden),
            s_act_func(),
            nn.Linear(num_hidden, num_inputs),
        )
        self.translate_net = nn.Sequential(
            nn.Linear(total_inputs, num_hidden),
            t_act_func(),
            nn.Linear(num_hidden, num_hidden),
            t_act_func(),
            nn.Linear(num_hidden, num_inputs),
        )

        def init(m):
            if isinstance(m, nn.Linear):
                m.bias.data.fill_(0)
                nn.init.orthogonal_(m.weight.data)

    def forward(self, inputs, cond_inputs=None, mode="direct"):
        mask = self.mask

        masked_inputs = inputs * mask
        if cond_inputs is not None:
            masked_inputs = torch.cat([masked_inputs, cond_inputs], -1)

        if mode == "direct":
            log_s = self.scale_net(masked_inputs) * (1 - mask)
            t = self.translate_net(masked_inputs) * (1 - mask)
            s = torch.exp(log_s)
            return inputs * s + t, log_s.sum(-1, keepdim=True)
        else:
            log_s = self.scale_net(masked_inputs) * (1 - mask)
            t = self.translate_net(masked_inputs) * (1 - mask)
            s = torch.exp(-log_s)
            return (inputs - t) * s, -log_s.sum(-1, keepdim=True)


class FlowSequential(nn.Sequential):
    """A sequential container for flows.
    In addition to a forward pass it implements a backward pass and
    computes log jacobians.
    """

    def forward(self, inputs, cond_inputs=None, mode="direct", logdets=None, C=5):
        """Performs a forward or backward pass for flow modules.
        Args:
            inputs: a tuple of inputs and logdets
            mode: to run direct computation or inverse
        """
        self.num_inputs = inputs.size(-1)
        if logdets is None:
            if len(inputs.shape) == 2:
                logdets = torch.zeros(inputs.size(0), 1, device=inputs.device)
            elif len(inputs.shape) == 3:
                logdets = torch.zeros(
                    cond_inputs.shape[0], inputs.shape[1], 1, device=inputs.device
                )

        assert mode in ["direct", "inverse"]
        if mode == "direct":
            for module in self._modules.values():
                inputs, logdet = module(inputs, cond_inputs, mode)
                # print(logdet.shape)
                logdets += logdet
        else:
            for module in reversed(self._modules.values()):
                inputs, logdet = module(inputs, cond_inputs, mode)
                # print(logdet.shape)
                logdets += logdet
        return inputs, logdets

    def log_probs(self, inputs, cond_inputs=None):
        u, log_jacob = self(inputs, cond_inputs)
        self.u = u
        log_probs = -0.5 * u.pow(2) - 0.5 * math.log(2 * math.pi)
        probs = log_probs + log_jacob
        return probs

    def sample(self, num_samples=None, noise=None, cond_inputs=None):
        N = cond_inputs.shape[0]
        if noise is None:
            torch.manual_seed(0)
            noise = (
                torch.Tensor(1, num_samples, self.num_inputs).normal_().repeat(N, 1, 1)
            )
            # noise = torch.Tensor(num_samples, self.num_inputs).normal_()
        device = next(self.parameters()).device
        noise = noise.to(device)
        if cond_inputs is not None:
            cond_inputs = cond_inputs.to(device)
            cond_inputs = cond_inputs.unsqueeze(1)
        samples = self.forward(noise, cond_inputs, mode="inverse")[
            0
        ]  # N, num_samples, dim
        return samples

    def entropy(self, num_samples=None, noise=None, cond_inputs=None):
        N = cond_inputs.shape[0]
        if noise is None:
            torch.manual_seed(0)
            noise = (
                torch.Tensor(1, num_samples, self.num_inputs).normal_().repeat(N, 1, 1)
            )
            # noise = torch.Tensor(num_samples, self.num_inputs).normal_()
        device = next(self.parameters()).device
        noise = noise.to(device)
        if cond_inputs is not None:
            cond_inputs = cond_inputs.to(device)
            cond_inputs = cond_inputs.unsqueeze(1)
        samples, log_jacob = self.forward(
            noise, cond_inputs, mode="inverse"
        )  # N, num_samples, dim
        log_probs = -0.5 * noise.pow(2) - 0.5 * math.log(2 * math.pi)
        entropy = (log_probs + log_jacob).mean(-1)
        return entropy

    def set_num_inputs(self, num_inputs):
        self.num_inputs = num_inputs


class FlowSequentialMOG(nn.Sequential):
    """A sequential container for flows.
    In addition to a forward pass it implements a backward pass and
    computes log jacobians.
    """

    def init(self, C):
        self.C = C

    def forward(self, inputs, cond_inputs=None, mode="direct", logdets=None):
        """Performs a forward or backward pass for flow modules.
        Args:
            inputs: a tuple of inputs and logdets
            mode: to run direct computation or inverse
        """
        self.num_inputs = inputs.size(-1)  # N C L / N L
        if logdets is None:
            logdets = torch.zeros(
                inputs.size(0), self.C, self.num_inputs, device=inputs.device
            )

        assert mode in ["direct", "inverse"]
        if mode == "direct":
            for module in self._modules.values():
                inputs, logdet = module(inputs, cond_inputs, mode)
                if len(logdet.shape) == 2:
                    logdet = logdet.unsqueeze(1)
                logdets += logdet
        else:
            for module in reversed(self._modules.values()):
                inputs, logdet = module(inputs, cond_inputs, mode)
        return inputs, logdets

    def log_probs(self, inputs, cond_inputs=None):
        u, log_jacob = self(inputs, cond_inputs)  # [N, C, L] [N, C, L]
        self.u = u
        log_probs = -0.5 * u.pow(2) - 0.5 * math.log(2 * math.pi)  # [N, C, L]
        probs = torch.logsumexp(log_probs + log_jacob, dim=1)
        return probs

    def sample(self, num_samples=None, noise=None, cond_inputs=None):
        B = cond_inputs.shape[0]
        device = cond_inputs.device

        if noise is None:
            noise = (
                torch.Tensor(num_samples, 1, self.C, self.num_inputs)
                .normal_()
                .repeat(1, B, 1, 1)
            )
        noise = noise.to(device)
        samples = self.forward(noise, cond_inputs, mode="inverse")[0]
        return samples.permute(1, 0, 2)  # B, N, L

    def entropy(self, num_samples=None, noise=None, cond_inputs=None):
        if noise is None:
            noise = torch.Tensor(num_samples, self.C, self.num_inputs).normal_()
        device = next(self.parameters()).device
        noise = noise.to(device)
        if cond_inputs is not None:
            cond_inputs = cond_inputs.to(device)
        samples = self.forward(noise, cond_inputs, mode="inverse")[0]
        return samples

    def set_num_inputs(self, num_inputs):
        self.num_inputs = num_inputs
