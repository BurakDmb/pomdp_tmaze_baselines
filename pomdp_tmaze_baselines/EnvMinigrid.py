import gym
from gym import spaces
from gym_minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper

# from pomdp_tmaze_baselines.utils.AE import Autoencoder
# import sys
# from gym_minigrid.envs.empty import EmptyEnv
import numpy as np
# import random
import torch
import torchvision
from PIL import Image
# import time

input_dims = 48
latent_dims = 128
in_channels = 3
hidden_size = 128
action_dim = 3
# batch_size = 8192
# epochs = 50
# train_set_ratio = 0.8
# learning_rate = 1e-2
# maximum_gradient = 1000


# Reward: 0 if not reached an end,
# "1 - 0.9 * (self.step_count / self.max_steps)" if goal reached.
# Maxstep limit is 4*size*size, episode ends if maxstep reached.
# Movement Actions L-R-F -> 0, 1, 2
class MinigridEnv(gym.Env):
    def __init__(self, **kwargs):
        super(MinigridEnv, self).__init__()
        self.env_type = "MinigridEnv"
        self.memory_type = kwargs.get('memory_type', 0)
        self.memory_length = kwargs.get('memory_length', 1)
        self.intrinsic_enabled = kwargs.get('intrinsic_enabled', False)
        self.intrinsic_beta = kwargs.get('intrinsic_beta', 0.1)

        self.ae_enabled = kwargs.get('ae_enabled', False)
        self.ae_integer = kwargs.get('ae_integer', False)
        self.ae_shared = kwargs.get('ae_shared', False)
        self.ae_path = kwargs.get('ae_path', None)
        self.ae_rcons_err_type = kwargs.get('ae_rcons_err_type', "MSE")
        self.device = kwargs.get('device', "cpu")
        self.env_id = kwargs.get('env_id', 0)

        env = gym.make('MiniGrid-MemoryS13-v0')
        env = RGBImgPartialObsWrapper(env)
        self.env = ImgObsWrapper(env)

        self.success_count = 0
        self.episode_count = 0
        self.intrinsic_avg_loss = 0
        self.intrinsic_count = 0
        self.memory_change_counter = 0

        self.transforms = torchvision.transforms.Compose([
                torchvision.transforms.Resize(input_dims),
                torchvision.transforms.PILToTensor(), ])
        self.transforms_ae = torchvision.transforms.Compose([
                torchvision.transforms.Resize(input_dims),
                torchvision.transforms.ToTensor(), ])

        # self.action_space = self.env.action_space
        self.action_space = spaces.Discrete(action_dim)
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(
                in_channels,
                input_dims,
                input_dims), dtype=np.uint8)

        if self.ae_enabled:
            if self.ae_shared:
                self.ae_comm_list = kwargs.get('ae_comm_list', None)
            else:
                ae_model_ = torch.load(self.ae_path).to(self.device)
                ae_model_ = ae_model_.module
                ae_model_.eval()
                self.ae_model = ae_model_

        # Memory type 0 = None
        if self.memory_type == 0 and self.ae_enabled:

            if self.ae_integer:
                self.obs_number_of_dimension = latent_dims
                self.obs_single_size = latent_dims
                self.action_space = spaces.Discrete(3)
                self.observation_space = gym.spaces.Box(
                    low=0,
                    high=1,
                    shape=(self.obs_number_of_dimension, ),
                    dtype=np.int32)
            else:
                self.obs_number_of_dimension = latent_dims
                self.obs_single_size = latent_dims
                self.action_space = spaces.Discrete(3)
                self.observation_space = gym.spaces.Box(
                    low=0,
                    high=1,
                    shape=(self.obs_number_of_dimension, ),
                    dtype=np.float32)

        # Memory type 1 = Kk
        # LastK memory, it automatically keeps track of lastK
        # observation and adds to its memory.
        elif self.memory_type == 1 and self.ae_enabled:
            self.obs_single_size = latent_dims
            self.mem_single_size = self.obs_single_size
            self.obs_number_of_dimension = (
                self.obs_single_size +
                self.mem_single_size * self.memory_length)

            if self.ae_integer:
                self.external_memory = np.zeros(
                    self.obs_number_of_dimension -
                    self.obs_single_size, dtype=np.int32)

                self.high = np.full(
                    self.obs_number_of_dimension, 1, dtype=np.int32)
                self.observation_space = gym.spaces.Box(
                    low=0,
                    high=1,
                    shape=(self.obs_number_of_dimension, ), dtype=np.int32)
            else:
                self.external_memory = np.zeros(
                    self.obs_number_of_dimension -
                    self.obs_single_size, dtype=np.float32)
                self.high = np.full(
                    self.obs_number_of_dimension, 1, dtype=np.float32)
                self.observation_space = gym.spaces.Box(
                    low=0,
                    high=1,
                    shape=(self.obs_number_of_dimension, ), dtype=np.float32)

            self.action_space = self.action_space

        # Memory type 2 = Bk
        # Binary memory, an external memory will hold k bits,
        # Total actions are multiplied with 2^k to store k-bits.
        # External memory can hold values from 0 to 2^k -1 with k bits.
        # Actions: L-R-F cross 2^k. Ex: (0  0), (0  2^k-1), (2  0), (1  2^k-1)
        elif self.memory_type == 2 and self.ae_enabled:
            self.obs_single_size = latent_dims
            self.mem_single_size = 1
            # Actions are defined n e s w, add observation to the memory.
            self.obs_number_of_dimension = (self.obs_single_size +
                                            self.mem_single_size *
                                            self.memory_length)

            if self.ae_integer:
                self.high = np.full(
                    self.obs_number_of_dimension, 1, dtype=np.int32)
                self.observation_space = gym.spaces.Box(
                    low=0,
                    high=1,
                    shape=(self.obs_number_of_dimension, ), dtype=np.int32)

                self.external_memory = np.zeros(
                    self.memory_length, dtype=np.int32)
            else:
                self.high = np.full(
                    self.obs_number_of_dimension, 1, dtype=np.float32)
                self.observation_space = gym.spaces.Box(
                    low=0,
                    high=1,
                    shape=(self.obs_number_of_dimension, ), dtype=np.float32)
                self.external_memory = np.zeros(
                    self.memory_length, dtype=np.float32)

            self.action_space = spaces.MultiDiscrete(
                [3, 2**self.memory_length])
        # Memory type 3 = Ok
        # Observation-k memory, an external memory will hold k observation,
        # Actions: L-R-F cross memory actions ( add obs, clear obs, NOOP)
        elif self.memory_type == 3 and self.ae_enabled:
            self.obs_single_size = latent_dims
            self.mem_single_size = self.obs_single_size
            self.obs_number_of_dimension = (
                self.obs_single_size +
                self.mem_single_size * self.memory_length)

            self.external_memory_recons_losses = np.zeros(
                    self.memory_length,
                    dtype=np.float32)

            if self.ae_integer:
                self.external_memory = np.zeros(
                    self.obs_number_of_dimension - self.obs_single_size,
                    dtype=np.int32)
                self.high = np.full(
                    self.obs_number_of_dimension, 1.0,
                    dtype=np.int32)
                self.observation_space = gym.spaces.Box(
                    low=0,
                    high=1,
                    shape=(self.obs_number_of_dimension, ), dtype=np.int32)
            else:
                self.external_memory = np.zeros(
                    self.obs_number_of_dimension - self.obs_single_size,
                    dtype=np.float32)
                self.high = np.full(
                    self.obs_number_of_dimension, 1.0,
                    dtype=np.float32)
                self.observation_space = gym.spaces.Box(
                    low=0,
                    high=1,
                    shape=(self.obs_number_of_dimension, ), dtype=np.float32)

            self.action_space = spaces.MultiDiscrete(
                [3, 2])

        # Memory type 4 = OAk
        # ObservationAction-k memory, an external memory will hold
        # k observation-action pair,
        # Actions: L-R-F cross memory actions ( add obs, clear obs, NOOP)
        elif self.memory_type == 4 and self.ae_enabled:
            self.obs_single_size = latent_dims
            self.act_single_size = 1
            self.mem_single_size = (
                self.obs_single_size + self.act_single_size)
            self.obs_number_of_dimension = (
                self.obs_single_size +
                self.mem_single_size * self.memory_length)
            self.external_memory_recons_losses = np.zeros(
                    self.memory_length,
                    dtype=np.float32)

            if self.ae_integer:
                self.external_memory = np.zeros(
                    self.obs_number_of_dimension -
                    self.obs_single_size, dtype=np.int32)
                self.high = np.full(
                    self.obs_number_of_dimension, 1.0,
                    dtype=np.int32)
                self.observation_space = gym.spaces.Box(
                    low=0,
                    high=1,
                    shape=(self.obs_number_of_dimension, ), dtype=np.int32)
            else:
                self.external_memory = np.zeros(
                    self.obs_number_of_dimension -
                    self.obs_single_size, dtype=np.float32)
                self.high = np.full(
                    self.obs_number_of_dimension, 1.0,
                    dtype=np.float32)
                self.observation_space = gym.spaces.Box(
                    low=0,
                    high=1,
                    shape=(self.obs_number_of_dimension, ), dtype=np.float32)

            self.action_space = spaces.MultiDiscrete(
                [3, 2])

        # Memory type 5 = None (For LSTM)
        elif self.memory_type == 5 and self.ae_enabled:
            self.obs_number_of_dimension = latent_dims
            self.obs_single_size = latent_dims
            self.action_space = spaces.Discrete(3)

            if self.ae_integer:
                self.observation_space = gym.spaces.Box(
                    low=0,
                    high=1,
                    shape=(self.obs_number_of_dimension, ), dtype=np.int32)
            else:
                self.observation_space = gym.spaces.Box(
                    low=0,
                    high=1,
                    shape=(self.obs_number_of_dimension, ), dtype=np.float32)

        self.observation = self.observation_space.sample()
        self.observation_valid = False
        self.step_count = 0

    # This function gets the current observation.
    # In addition, it calculates the reconstruction loss if ae is enabled.
    # If AE is disabled, then reconstruction loss is zero.
    def _get_observation(self, ):
        if self.observation_valid:
            return self.observation, self.last_recons_loss
        else:
            if self.ae_enabled:
                if self.ae_integer:
                    observation = np.zeros(
                        self.obs_number_of_dimension, dtype=np.int32)
                else:
                    observation = np.zeros(
                        self.obs_number_of_dimension, dtype=np.float32)
                observation_ae_img = Image.fromarray(self.current_state)
                if self.transforms is not None:
                    observation_ae = self.transforms_ae(
                        observation_ae_img)

                observation_ae = observation_ae[None, :]

                recons_obs, observation_ = self.get_ae_result(observation_ae)

                # Calculate reconstruction loss (MAE or MSE)
                # with autoencoders. Default is MSE.
                if self.ae_rcons_err_type == "MAE":
                    recons_loss = (np.abs(
                        observation_ae - recons_obs)).sum()
                else:
                    recons_loss = ((
                        observation_ae - recons_obs)**2).sum()
                # Normalizing the loss with the maximum loss(each rgb pixel
                # density is totally different, error is 1,
                # and total of 1*input_dims*input_dims*in_channels
                # Loss is in range (0, 1)
                recons_loss = recons_loss / (
                    1*input_dims*input_dims*in_channels)

                observation_latent = observation_.cpu().numpy().transpose()
                observation[:self.obs_single_size] = observation_latent[:, 0]
                if self.memory_type != 0 and self.memory_type != 5:
                    observation[self.obs_single_size:] = self.external_memory
                observation_ae_img.close()

            else:
                observation_ae_img = Image.fromarray(self.current_state)
                if self.transforms is not None:
                    observation_ae = self.transforms(
                        observation_ae_img)
                # CxHxW with [0, 255] uint8
                observation = observation_ae.cpu().numpy()
                observation_ae_img.close()
                recons_loss = 0.0

            self.observation = observation
            self.observation_valid = True
            self.last_recons_loss = recons_loss
            return self.observation, self.last_recons_loss

    def add_observation_to_memory(self, memoryAction, action):
        # The memory update strategy could be changed, for now, it is set to
        # First In First Out Strategy

        # In each add step, shift memory by self.mem_single_size and
        # add the new element into the starting of the array.

        # Memory type 2 = Bk
        if self.memory_type == 2:

            string_binary = str(bin(memoryAction))

            # Converting int value to binary and splitting each
            # bit in a list.
            # Ex: For value 5, binary is 101. Char split is [1, 0, 1].
            int_list = list(map(int, [char for char in string_binary][2:]))
            if len(int_list) < self.memory_length:
                int_list = [0]*(self.memory_length - len(int_list)) + int_list
            if (self.external_memory != np.array(
                    int_list, dtype=np.float32)).any():
                self.memory_change_counter += 1
            self.external_memory = np.array(int_list, dtype=np.float32)
        else:

            # Memory type 1 = Kk
            if self.memory_type == 1:
                self.memory_change_counter += 1
                self.external_memory = np.roll(
                    self.external_memory, self.mem_single_size)
                self.external_memory[0:self.mem_single_size] = \
                    self._get_observation()[0][:self.obs_single_size]

            # Memory type 3 = Ok
            elif self.memory_type == 3 and memoryAction == 1:
                self.memory_change_counter += 1
                self.external_memory = np.roll(
                    self.external_memory, self.mem_single_size)
                self.external_memory_recons_losses = np.roll(
                    self.external_memory_recons_losses, 1)

                observation, recons_loss = self._get_observation()
                self.external_memory[0:self.mem_single_size] = \
                    observation[:self.obs_single_size]
                self.external_memory_recons_losses[0] = recons_loss

            # Memory type 4 = OAk
            elif self.memory_type == 4 and memoryAction == 1:
                self.memory_change_counter += 1
                self.external_memory = np.roll(
                    self.external_memory, self.mem_single_size)
                self.external_memory_recons_losses = np.roll(
                    self.external_memory_recons_losses, 1)

                observation, recons_loss = self._get_observation()
                self.external_memory[0:self.mem_single_size] = \
                    np.append(
                        observation[:self.obs_single_size],
                        [action])
                self.external_memory_recons_losses[0] = recons_loss
            self.observation_valid = False

    def step(self, action):
        done = False
        success = 0
        movementAction = action
        memoryAction = 0

        # Memory type 0 = None
        if self.ae_enabled and self.memory_type == 0:
            movementAction = action
            memoryAction = 0
        # Memory type 1 = Kk
        elif self.ae_enabled and self.memory_type == 1:
            movementAction = action
            memoryAction = 1
        # Memory type 2 = Bk
        elif self.ae_enabled and self.memory_type == 2:
            movementAction = action[0]
            memoryAction = action[1]
        # Memory type 3 = Ok
        elif self.ae_enabled and self.memory_type == 3:
            movementAction = action[0]
            memoryAction = action[1]
        # Memory type 4 = OAk
        elif self.ae_enabled and self.memory_type == 4:
            movementAction = action[0]
            memoryAction = action[1]
        # Memory type 5 = None (For LSTM)
        elif self.ae_enabled and self.memory_type == 5:
            movementAction = action
            memoryAction = 0

        new_state, reward, done, _ = self.env.step(movementAction)
        self.step_count += 1

        # Check if add observation to memory action or not
        if self.ae_enabled and memoryAction != 0:
            self.add_observation_to_memory(memoryAction, movementAction)

        if self.intrinsic_enabled:
            if self.ae_enabled:
                # observations are transformed into tensors with torchvision,
                # which automatically converts
                # PIL images in shape of (H x W x C), range [0, 255] to a
                # uint8 tensor of shape (C x H x W) in the range [0, 255]
                with torch.no_grad():
                    """
                    observation_ae_img = Image.fromarray(new_state)
                    observation_ae = self.transforms_ae(
                        observation_ae_img)
                    observation_ae_img.close()
                    observation_ae = observation_ae[None, :]

                    new_state_gen_tmp, _ = self.get_ae_result(observation_ae)

                    new_state_gen_tmp = np.squeeze(
                        new_state_gen_tmp.cpu().numpy(), 0)

                    new_state_img = Image.fromarray(new_state)
                    new_state_orig_tmp = self.transforms_ae(
                        new_state_img).cpu().numpy()
                    new_state_img.close()

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
                    # and total of 1*input_dims*input_dims*in_channels
                    # Loss is in range (0, 1)
                    loss = loss / (1*input_dims*input_dims*in_channels)

                    # In the code below, if the calculated reconstruction loss
                    # is greater than average, an additional intrinsic
                    # motivation is constructed. The range of the reward is
                    # shaped accordingly to be in the range of [0.5, 1].
                    # (since average reward of this environment due to
                    # the partially observability is 0.5)
                    # Reward shaping steps. (reconstruction loss is normalized,
                    # therefore max. value is 1)
                    # [avg, 1] (subtract avg)
                    # [0, 1-avg] (divide 2*(1-avg))
                    # [0, 0.5] (add 0.5)
                    # [0.5, 1] (result range)
                    if self.intrinsic_avg_loss > 0 and (
                            loss / self.intrinsic_avg_loss) > 1.0:
                        intrinsic_reward = ((
                            (loss-self.intrinsic_avg_loss) /
                            (2*(1-self.intrinsic_avg_loss))
                            ) + 0.5) / (self.step_count**2)
                    else:
                        intrinsic_reward = 0

                    # Calculating cumulative average.
                    self.intrinsic_count += 1
                    self.intrinsic_avg_loss = self.intrinsic_avg_loss + (
                        loss - self.intrinsic_avg_loss
                        )/(self.intrinsic_count)

                    # TODO: Since the difference of the reconstruction loss
                    # between common and uncommon observations is small,
                    # a scaling operation could be useful to get more
                    # distinct and useful intrinsic rewards.
                    """
                    intrinsic_reward = ((np.mean(
                        self.external_memory_recons_losses)) - 1)

                    # Higher loss leads to higher positive reward.
                    # Intrinsic motivation is multiplied with intrinsic beta
                    # to tune the density of the intrinsic reward
                    reward += self.intrinsic_beta * intrinsic_reward
            else:
                print("***Intrinsic reward calculation without autoencoders" +
                      " is not supported in visual environments, " +
                      "stopping the execution.***")
                exit(1)

        if done:
            self.episode_count += 1
            if tuple(self.env.agent_pos) == self.env.success_pos:
                success = 1
                self.success_count += 1

        self.current_state = new_state
        self.episode_reward += reward

        self.observation_valid = False
        return self._get_observation()[0], reward, done, {
            'success': success, 'is_success': bool(success),
            'memory_change_counter': self.memory_change_counter}

    def reset(self, seed=None):
        if self.ae_enabled:
            if self.ae_integer:
                # Memory type 1 = Kk
                if self.memory_type == 1:
                    self.external_memory = np.zeros(
                        self.obs_number_of_dimension -
                        self.obs_single_size, dtype=np.int32)

                # Memory type 2 = Bk
                elif self.memory_type == 2:
                    self.external_memory = np.zeros(
                        self.memory_length, dtype=np.int32)

                # Memory type 3 = Ok
                elif self.memory_type == 3:
                    self.external_memory = np.zeros(
                        self.obs_number_of_dimension - self.obs_single_size,
                        dtype=np.int32)

                # Memory type 4 = OAk
                elif self.memory_type == 4:
                    self.external_memory = np.zeros(
                        self.obs_number_of_dimension -
                        self.obs_single_size, dtype=np.int32)

            else:
                # Memory type 1 = Kk
                if self.memory_type == 1:
                    self.external_memory = np.zeros(
                        self.obs_number_of_dimension -
                        self.obs_single_size, dtype=np.float32)

                # Memory type 2 = Bk
                elif self.memory_type == 2:
                    self.external_memory = np.zeros(
                        self.memory_length, dtype=np.float32)

                # Memory type 3 = Ok
                elif self.memory_type == 3:
                    self.external_memory = np.zeros(
                        self.obs_number_of_dimension - self.obs_single_size,
                        dtype=np.float32)

                # Memory type 4 = OAk
                elif self.memory_type == 4:
                    self.external_memory = np.zeros(
                        self.obs_number_of_dimension -
                        self.obs_single_size, dtype=np.float32)

        obs = self.env.reset(seed=seed)
        self.current_state = obs

        self.intrinsic_avg_loss = 0
        self.intrinsic_count = 0
        self.episode_reward = 0

        self.step_count = 0
        self.memory_change_counter = 0
        self.observation_valid = False
        return self._get_observation()[0]

    def get_ae_result(self, tensor_data):
        if self.ae_shared:
            data = tensor_data.cpu().numpy()
            comm_variable = self.ae_comm_list[self.env_id]
            comm_variable[0].put(data)
            (obs_, latent_) = comm_variable[1].get()
            obs = torch.tensor(
                obs_, dtype=torch.float32, requires_grad=False)
            latent = torch.tensor(
                latent_, dtype=torch.float32, requires_grad=False)
            latent = self.binarize_latent(latent)
        else:
            with torch.no_grad():
                obs, latent = self.ae_model(tensor_data)
                if self.ae_integer:
                    latent = self.binarize_latent(latent)

        return obs, latent

    def binarize_latent(self, latent):
        return (latent > 0.5).int()
