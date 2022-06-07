import gym
# from gym import spaces
from gym_minigrid.wrappers import RGBImgObsWrapper

from pomdp_tmaze_baselines.utils.AE import Autoencoder
# import sys
# from gym_minigrid.envs.empty import EmptyEnv
import numpy as np
# import random
# import torch
import torchvision
from PIL import Image

input_dims = 48
latent_dims = 3
in_channels = 3
hidden_size = 128
# batch_size = 8192
# epochs = 50
# train_set_ratio = 0.8
# learning_rate = 1e-2
# maximum_gradient = 1000


class MinigridEnv(gym.Env):
    def __init__(self, **kwargs):
        super(MinigridEnv, self).__init__()
        self.memory_type = kwargs.get('memory_type', 0)
        self.memory_length = kwargs.get('memory_length', 1)
        self.intrinsic_enabled = kwargs.get('intrinsic_enabled', 0)
        self.intrinsic_beta = kwargs.get('intrinsic_beta', 0.5)

        self.env = gym.make('MiniGrid-Empty-Random-6x6-v0')
        self.env = RGBImgObsWrapper(self.env)

        # TODO: Change the observation space after autoencoders.
        original_space = self.env.observation_space['image']
        low = np.moveaxis(original_space.low, -1, 0)
        high = np.moveaxis(original_space.high, -1, 0)
        self.observation_space = gym.spaces.Box(
            low=low,
            high=high,
            shape=(
                original_space.shape[-1],
                original_space.shape[0],
                original_space.shape[1]), dtype=original_space.dtype)

        self.action_space = self.env.action_space
        # TODO
        # np.moveaxis(x, -1, 0)
        # TODO: Initialize autoencoder and load from model file.
        self.ae = Autoencoder(
            input_dims, latent_dims, hidden_size, in_channels)

        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(input_dims),
            torchvision.transforms.ToTensor(), ])

    def _get_observation(self):
        observation = self.env.observation(self.env.gen_obs())['image']
        # observation = np.expand_dims(observation, 0)
        observation = Image.fromarray(observation)
        if self.transforms is not None:
            observation = self.transforms(observation)
        observation = observation[None, :]

        # TODO: Return the latent space observation after autoencoder.
        # obs_tensor = torch.from_numpy(observation)
        # TODO: Add No Grad.
        self.ae(observation)
        # detach from device and to cpu.
        latent_observation = self.ae.z
        return latent_observation

    def step(self, action):
        done = False
        success = 0

        new_state, reward, done, _ = self.env.step(action)

        intrinsic_reward = 0

        # Calculate reconstruction loss with autoencoders.
        # Ex: (new_state-autoencoder(current_state))^2
        # And set the intrinsic reward.

        reward = reward + intrinsic_reward
        self.current_state = new_state

        self.episode_reward += reward
        return self._get_observation(), reward, done, {'success': success}

    def reset(self):
        self.env.reset()

        self.episode_reward = 0

        return self._get_observation()
