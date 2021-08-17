from typing import Callable, Dict, List, Optional, Tuple, Type, Union, Any

import gym
import torch as th
from torch import nn

from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.dqn.policies import QNetwork, DQNPolicy
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    FlattenExtractor
)


class CustomACNetwork(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the feature extractor.

    :param feature_dim: dimension of the features extracted with the
        features_extractor (e.g. features from a CNN)
    :param net_arch: architecture structure, union of shared layer and
        seperated policy and value network layers.
        pi->policy network layers, vf->value network layers.

        Example usage:
        net_arch=[dict(pi=[64, 64],vf=[64, 64])]
            (No shared layer)
        net_arch=[32, 64, dict(pi=[64, 64],vf=[64, 64])]
            (Two shared layers with sizes 32 and 64)
    """

    def __init__(
        self,
        feature_dim: int,
        net_arch: List[Union[int, Dict[str, List[int]]]],
    ):
        super(CustomACNetwork, self).__init__()

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
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the
            specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        shared = self.shared_net(features)
        return self.policy_net(shared), self.value_net(shared)


class MultiLayerActorCriticPolicy(ActorCriticPolicy):
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

        super(MultiLayerActorCriticPolicy, self).__init__(
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
        self.mlp_extractor = CustomACNetwork(self.features_dim, self.net_arch)


class CustomDQNNetwork(QNetwork):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        features_extractor: nn.Module,
        features_dim: int,
        net_arch: Optional[List[int]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
    ):
        super(CustomDQNNetwork, self).__init__(
            observation_space,
            action_space,
            features_extractor,
            features_dim,
            net_arch,
            activation_fn,
            normalize_images,
        )
        q_net = self.create_mlp(self.features_dim, self.action_space.n,
                                self.net_arch, self.activation_fn)
        self.q_net = nn.Sequential(*q_net)

    def forward(self, obs: th.Tensor) -> th.Tensor:
        """
        Predict the q-values.
        :param obs: Observation
        :return: The estimated Q-Value for each action.
        """
        return self.q_net(self.extract_features(obs))

    def create_mlp(self, input_dim: int, output_dim: int, net_arch: List[int],
                   activation_fn: Type[nn.Module] = nn.ReLU,
                   squash_output: bool = False
                   ) -> List[nn.Module]:
        """
        Create a multi layer perceptron (MLP), which is
        a collection of fully-connected layers each followed by an
        activation function.

        :param input_dim: Dimension of the input vector
        :param output_dim:
        :param net_arch: Architecture of the neural net
            It represents the number of units per layer.
            The length of this list is the number of layers.
        :param activation_fn: The activation function
            to use after each layer.
        :param squash_output: Whether to squash the output using a Tanh
            activation function
        :return:
        """

        if len(net_arch) > 0:
            modules = [nn.Linear(input_dim, net_arch[0]), activation_fn()]
        else:
            modules = []

        for idx in range(len(net_arch) - 1):
            modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1]))
            modules.append(activation_fn())

        if output_dim > 0:
            last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
            modules.append(nn.Linear(last_layer_dim, output_dim))
        if squash_output:
            modules.append(nn.Tanh())
        return modules


class CustomDQNPolicy(DQNPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[List[int]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        features_extractor_class:
            Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super(CustomDQNPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
        )

    def make_q_net(self) -> QNetwork:
        # Make sure we always have separate networks for
        # features extractors etc.
        net_args = self._update_features_extractor(self.net_args,
                                                   features_extractor=None)
        return QNetwork(**net_args).to(self.device)
