import gym
# from gym import spaces
from gym_minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper

# from pomdp_tmaze_baselines.utils.AE import Autoencoder
# import sys
# from gym_minigrid.envs.empty import EmptyEnv
import numpy as np
# import random
import torch
import torchvision
from PIL import Image

input_dims = 48
latent_dims = 128
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
        self.intrinsic_beta = kwargs.get('intrinsic_beta', 0.1)

        self.ae_enabled = kwargs.get('ae_enabled', False)
        self.ae_path = kwargs.get('ae_path', None)
        self.ae_rcons_err_type = kwargs.get('ae_rcons_err_type', "MSE")

        env = gym.make('MiniGrid-MemoryS13-v0')
        env = RGBImgPartialObsWrapper(env)
        self.env = ImgObsWrapper(env)

        original_space = self.env.observation_space
        self.action_space = self.env.action_space

        if self.ae_enabled:
            if self.ae_path is not None:
                self.ae = torch.load(self.ae_path).to("cuda")
                self.ae = self.ae.module
                self.ae.eval()
            else:
                print(
                    "***Autoencoder path is not defined, " +
                    "stopping the execution.***")
                exit(1)

            self.transforms = torchvision.transforms.Compose([
                torchvision.transforms.Resize(input_dims),
                torchvision.transforms.ToTensor(), ])

            self.inverse_transforms = torchvision.transforms.Compose([
                torchvision.transforms.ToPILImage(),
                torchvision.transforms.Resize(
                    self.env.observation_space.shape[0]), ])
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

    def _get_observation(self, obs):
        if self.ae_enabled:
            observation_ae = Image.fromarray(obs)
            if self.transforms is not None:
                observation_ae = self.transforms(observation_ae).to("cuda")
            observation_ae = observation_ae[None, :]
            with torch.no_grad():
                _, observation = self.ae(observation_ae)
            observation = observation.cpu().numpy()
        else:
            observation = obs
        return observation

    def step(self, action):
        done = False
        success = 0

        new_state, reward, done, _ = self.env.step(action)

        if self.intrinsic_enabled:
            if self.ae_enabled:
                # observations are transformed into tensors with torchvision,
                # which automatically converts
                # PIL images in shape of (H x W x C), range [0, 255] to a
                # torch.FloatTensor of shape (C x H x W) in the range
                # [0.0, 1.0]
                with torch.no_grad():
                    observation_ae_img = Image.fromarray(new_state)
                    observation_ae = self.transforms(
                        observation_ae_img).to("cuda")
                    observation_ae = observation_ae[None, :]
                    new_state_gen_tmp, _ = self.ae(observation_ae)
                    new_state_gen_tmp = torch.squeeze(
                        new_state_gen_tmp, 0).cpu().numpy()

                    new_state_img = Image.fromarray(new_state)
                    new_state_orig_tmp = self.transforms(
                        new_state_img).cpu().numpy()

                    # Calculate reconstruction loss (MAE or MSE)
                    # with autoencoders. Default is MSE.
                    if self.ae_rcons_err_type == "MAE":
                        loss = (np.abs(
                            new_state_orig_tmp - new_state_gen_tmp)).sum()
                    else:
                        loss = ((
                            new_state_orig_tmp - new_state_gen_tmp)**2).sum()

                    # Normalizing the loss with the maximum loss(each rgb pixel
                    # density is totally different, error is 1,
                    # and total of 1*input_dims*input_dims*in_channels)
                    loss = loss / (input_dims*input_dims*in_channels)

                    # Higher loss leads to higher positive reward.
                    # Intrinsic motivation is multiplied with intrinsic beta
                    # to tune the density of the intrinsic reward
                    intrinsic_reward = loss
                    reward = (
                        (1-self.intrinsic_beta)*reward
                        ) + (self.intrinsic_beta * intrinsic_reward)
            else:
                print("***Intrinsic reward calculation without autoencoders" +
                      " is not supported in visual environments, " +
                      "stopping the execution.***")
                exit(1)

        self.current_state = new_state

        self.episode_reward += reward
        return self._get_observation(
            new_state), reward, done, {'success': success}

    def reset(self):
        obs = self.env.reset()

        self.episode_reward = 0

        return self._get_observation(obs)
