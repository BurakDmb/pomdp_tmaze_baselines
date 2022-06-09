import gym
# from gym import spaces
from gym_minigrid.wrappers import RGBImgObsWrapper

from pomdp_tmaze_baselines.utils.AE import Autoencoder
# import sys
# from gym_minigrid.envs.empty import EmptyEnv
import numpy as np
# import random
import torch
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


# Reward: 0 if not reached an end,
# "1 - 0.9 * (self.step_count / self.max_steps)" if goal reached.
# Maxstep limit is 4*size*size, episode ends if maxstep reached.
class MinigridEnv(gym.Env):
    def __init__(self, **kwargs):
        super(MinigridEnv, self).__init__()
        self.memory_type = kwargs.get('memory_type', 0)
        self.memory_length = kwargs.get('memory_length', 1)
        self.intrinsic_enabled = kwargs.get('intrinsic_enabled', 0)
        self.intrinsic_beta = kwargs.get('intrinsic_beta', 0.5)

        self.autoencoder_enabled = kwargs.get('autoencoder_enabled', False)
        self.autoencoder_path = kwargs.get('autoencoder_path', None)

        self.env = gym.make('MiniGrid-Empty-Random-6x6-v0')
        self.env = RGBImgObsWrapper(self.env)

        original_space = self.env.observation_space['image']
        self.action_space = self.env.action_space

        if self.autoencoder_enabled:
            self.ae = Autoencoder(
                    input_dims, latent_dims, hidden_size, in_channels)
            if self.autoencoder_path is not None:
                self.ae.load_state_dict(
                    torch.load(self.autoencoder_path))
                # self.ae = self.ae.module
                self.ae.eval()
                pass

            self.transforms = torchvision.transforms.Compose([
                torchvision.transforms.Resize(input_dims),
                torchvision.transforms.ToTensor(), ])
            self.observation_space = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(latent_dims, 1), dtype=original_space.dtype)
        else:
            low = np.moveaxis(original_space.low, -1, 0)
            high = np.moveaxis(original_space.high, -1, 0)
            self.observation_space = gym.spaces.Box(
                low=low,
                high=high,
                shape=(
                    original_space.shape[-1],
                    original_space.shape[0],
                    original_space.shape[1]), dtype=original_space.dtype)

    def _get_observation(self):
        observation = self.env.observation(self.env.gen_obs())['image']

        if self.autoencoder_enabled:
            observation_ae = Image.fromarray(observation)
            if self.transforms is not None:
                observation_ae = self.transforms(observation_ae)
            observation_ae = observation_ae[None, :]
            with torch.no_grad():
                self.ae(observation_ae)
            # detach from device and to cpu.
            observation = self.ae.z.cpu().numpy()
        return observation

    def step(self, action):
        done = False
        success = 0

        new_state, reward, done, _ = self.env.step(action)

        intrinsic_reward = 0

        # Calculate reconstruction loss with autoencoders.
        # Ex: (new_state-autoencoder(new_state))^2
        # And set the intrinsic reward.
        # TODO: Normalize intrinsic reward.
        with torch.no_grad():
            loss = ((new_state - self.ae(new_state))**2).sum()
            intrinsic_reward = loss

        reward = reward + intrinsic_reward
        self.current_state = new_state

        self.episode_reward += reward
        return self._get_observation(), reward, done, {'success': success}

    def reset(self):
        self.env.reset()

        self.episode_reward = 0

        return self._get_observation()
