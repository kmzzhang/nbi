from nbi.model import get_flow, get_featurizer
import torch

flow_config = {
    'flow_hidden': 512,
    'num_cond_inputs': 512,
    'num_blocks': 20,
    'perm_seed': 3,
    'n_mog': 4
}

def get_flow_test():
    dim_param = 3
    resnet = get_featurizer('resnetrnn', 1, 512, depth=6)
    model = get_flow(resnet, dim_param, **flow_config)
    X = torch.zeros()




# nbi has different featurizers pre-defined
# light-curve problems: use resnetrnn

model = get_flow(resnet, dim_param, **flow_config)
