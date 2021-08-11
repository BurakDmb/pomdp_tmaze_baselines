from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import gym
import torch as th
from torch import nn

from stable_baselines3.common.policies import ActorCriticPolicy


class CustomNetwork(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the feature extractor.

    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """

    def __init__(
        self,
        feature_dim: int,
        net_arch: List[Union[int, Dict[str, List[int]]]],
    ):
        super(CustomNetwork, self).__init__()

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = net_arch[-1]['pi'][-1]
        self.latent_dim_vf = net_arch[-1]['vf'][-1]

        policy_net = []
        value_net = []
        shared_net = []
        last_policy_layer = feature_dim
        last_value_layer = feature_dim
        last_shared_layer = feature_dim

        for layer in net_arch:
            if isinstance(layer, int):
                shared_net.append(nn.Linear(last_shared_layer, layer))
                shared_net.append(nn.Tanh())
                last_shared_layer = layer
            elif isinstance(layer, dict):
                for p_layer in layer['pi']:
                    policy_net.append(nn.Linear(last_policy_layer, p_layer))
                    policy_net.append(nn.Tanh())
                    last_policy_layer = p_layer

                for v_layer in layer['vf']:
                    value_net.append(nn.Linear(last_value_layer, v_layer))
                    value_net.append(nn.Tanh())
                    last_value_layer = v_layer
                break

        self.shared_net = nn.Sequential(*shared_net)
        self.policy_net = nn.Sequential(*policy_net)
        self.value_net = nn.Sequential(*value_net)

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        shared = self.shared_net(features)
        return self.policy_net(shared), self.value_net(shared)


class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Callable[[float], float],
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        *args,
        **kwargs,
    ):

        super(CustomActorCriticPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )
        # Disable orthogonal initialization
        self.ortho_init = False

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomNetwork(self.features_dim, self.net_arch)
