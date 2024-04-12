import torch as T
from torch import nn
from gymnasium import spaces
from torch.nn import functional as F
from typing import List


from rl_projects.alg.util.network_util import identity, create_mlp, get_action_dim


class ContCritic(nn.Module):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        features_dim: int,
        net_arch: List[int],
        init_w: float = 3e-3,
        activation_fn=F.relu,
        output_activation=identity,
        n_critics: int = 2
    ):
        super().__init__()

        action_dim = get_action_dim(action_space=action_space)

        self.n_critics = n_critics
        self.q_networks: List[nn.Module] = []

        for idx in range(n_critics):
            q_net_list = create_mlp(features_dim + action_dim, 1, net_arch, activation_fn)
            q_net = nn.Sequential(*q_net_list)
            self.add_module(f"qf{idx}", q_net)
            self.q_networks.append(q_net)
