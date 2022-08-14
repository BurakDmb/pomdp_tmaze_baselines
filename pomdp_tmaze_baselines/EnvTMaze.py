import gym
from gym import spaces
import sys
import numpy as np
import random


class TMazeEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, **kwargs):
        self.env_type = "TMazeEnv"
        self.grid_size = (kwargs.get('maze_length'), 3)
        self.grid = ['X' * (self.grid_size[0] - 1) + '_',
                     '_' * (self.grid_size[0]),
                     'X' * (self.grid_size[0] - 1) + "_"]

        # Rewards
        self.fl_default_reward = -0.1
        self.fl_false_goal_reward = -40.0
        self.fl_true_goal_reward = 40.0
        # print("default_tmaze: ", kwargs.get('arg1'))
        self.fl_intended_direction_prob = 1.0

        # State:
        # - coordinates: x, y
        # - y of the true goal location: 0 for north, 2 for south

        # Initial states
        self.li_initial_states = [(0, 1, 0), (0, 1, 2)]

        # Terminal states
        self.li_terminal_states = [
            (self.grid_size[0] - 1, 0, 0), (self.grid_size[0] - 1, 0, 2),
            (self.grid_size[0] - 1, 2, 0), (self.grid_size[0] - 1, 2, 2)]

        # Actions: n e s w
        self.action_space = spaces.Discrete(4)

        # Full observability: (x, y, y of the true goal)
        self.high = np.zeros(3, dtype=int)
        self.high[0] = self.grid_size[0] - 1
        self.high[1] = self.grid_size[1] - 1
        self.high[2] = 2
        self.observation_space = spaces.MultiDiscrete(self.high+1)

        self.episode_reward = 0
        self.success_count = 0
        self.episode_count = 0

    # @staticmethod
    def _is_state_available(self, state):
        x, y, _ = state
        if x < 0 or x >= self.grid_size[0]:
            return False
        if y < 0 or y >= self.grid_size[1]:
            return False
        if self.grid[y][x] != '_':
            return False
        return True

    def _one_agent_step(self, state, action):
        if action not in range(4):
            print(f"Error - Invalid Action:{action}")
            sys.exit(1)

        reward = self.fl_default_reward
        possible_states = {state: 0}
        action_effects = [(0, -1), (1, 0), (0, 1), (-1, 0)]     # n e s w
        for action_index, action_effect in enumerate(action_effects):
            probability = (1.0 - self.fl_intended_direction_prob) / 3.0
            if action_index == action:  # matched action
                probability = self.fl_intended_direction_prob

            possible_next_state = (
                state[0] + action_effect[0], state[1] + action_effect[1],
                state[2])
            if self._is_state_available(possible_next_state):
                possible_states[possible_next_state] = probability
            else:
                possible_states[state] += probability

        assert (sum(possible_states.values()) == 1.0)
        next_state = random.choices(
            list(possible_states.keys()), list(possible_states.values()))[0]
        done = False
        success = 0
        if next_state in self.li_terminal_states:
            done = True
            self.episode_count += 1
            if next_state[1] == next_state[2]:
                reward = self.fl_true_goal_reward
                success = 1
                self.success_count += 1
            else:
                reward = self.fl_false_goal_reward
        return next_state, reward, done, success

    # Actions: n e s w
    def step(self, action):
        new_state, reward, done, success = self._one_agent_step(
            self.current_state, action)

        self.current_state = new_state
        self.episode_reward += reward
        return self._get_observation(), reward, done, {
            'success': success, 'is_success': bool(success)}

    def reset(self):
        self.current_state = random.choice(
                self.li_initial_states)
        self.episode_reward = 0
        return self._get_observation()

    def get_observation_space_size(self):
        return len(self.observation_space.nvec)

    def get_action_space_size(self):
        return self.action_space.n

    def _render(self, mode='human', close=False):
        pass

    def _get_observation(self):
        return self.current_state


class TMazeEnvPOMDP(TMazeEnv):
    def __init__(self, **kwargs):
        super(TMazeEnvPOMDP, self).__init__(**kwargs)
        # print("Tmazev1: ", kwargs.get('arg1'))
        self.light_x = 0

        # Partial observability: (wall_north, wall_east,
        # wall_south, wall_west, y of the true goal)
        self.high = np.full(5, 1.0, dtype=int)
        self.high[-1] = 2
        self.observation_space = spaces.MultiDiscrete(self.high+1)

    def _get_observation(self):
        observation = np.zeros(5, dtype=int)
        state = self.current_state
        # Partial observability

        # north direction
        if state[1] == 0 or self.grid[state[1] - 1][state[0]] == 'X':
            observation[0] = 1

        # east direction
        if (state[0] == (self.grid_size[0] - 1) or
           self.grid[state[1]][state[0] + 1] == 'X'):
            observation[1] = 1

        # south direction
        if (state[1] == (self.grid_size[1] - 1) or
           self.grid[state[1] + 1][state[0]] == 'X'):
            observation[2] = 1

        # west direction
        if state[0] == 0 or self.grid[state[1]][state[0] - 1] == 'X':
            observation[3] = 1

        # the agent can only get the true goal location either in the light
        # (start) location or in the terminal state
        if state[0] == self.light_x or state in self.li_terminal_states:
            observation[-1] = state[2]
        else:
            # otherwise, it gets a neutral direction for the true goal
            observation[-1] = 1
        return observation


# Partial observable,
# Augmented external memory with the options of
# None, Kk, Bk, Ok and OAk which is encoded as
# 0, 1, 2, 3, 4 respectively.
# Small k corrensponds as the memory length.
# Actions are defined seperately
# None: 4 movement actions
# Kk: 4 movement actions (Memory always updates according the history)
# Bk: 4 movement x 2 memory actions (Set memory bit or dont save)
#     = Total of 8 possible actions
# Ok: 4 movement x 2 memory actions (Save observation or dont save)
#     = Total of 8 possible actions
# OAk: 4 movement x 2 memory actions (Save obs-action pair or dont save)
#      = Total of 8 possible actions
# Ref: Icarte, Rodrigo Toro, et al. "The act of remembering: a study in
# partially observable reinforcement learning."
# arXiv preprint arXiv:2010.01753 (2020).
class TMazeEnvMemoryWrapped(TMazeEnvPOMDP):
    def __init__(self, **kwargs):
        super(TMazeEnvMemoryWrapped, self).__init__(**kwargs)
        self.memory_type = kwargs.get('memory_type', 0)
        self.memory_length = kwargs.get('memory_length', 1)
        self.intrinsic_enabled = kwargs.get('intrinsic_enabled', False)
        self.intrinsic_beta = kwargs.get('intrinsic_beta', 0.1)

        # Memory type 0 = None
        if self.memory_type == 0:
            # Partial observability: (wall_north, wall_east,
            # wall_south, wall_west, y of the true goal)
            self.obs_number_of_dimension = 5
            self.high = np.full(5, 1.0, dtype=int)
            self.high[-1] = 2
            self.observation_space = spaces.MultiDiscrete(self.high+1)
            self.action_space = self.action_space

        # Memory type 1 = Kk
        elif self.memory_type == 1:
            # Partial observability: (wall_north, wall_east,
            # wall_south, wall_west, y of the true goal
            # and additional dimensions with size
            # (memory_seq_length*observation_size)
            # Note that max. number of dimensions for a np array is 32.
            # So the maximum fixed sequence size could be 5.

            self.obs_single_size = len(self.high)
            self.mem_single_size = self.obs_single_size
            # Actions are defined n e s w, add observation to the memory.
            self.obs_number_of_dimension = (self.obs_single_size +
                                            self.mem_single_size *
                                            self.memory_length)
            self.external_memory = np.zeros(self.obs_number_of_dimension -
                                            self.obs_single_size, dtype=int)
            self.high = np.full(self.obs_number_of_dimension, 1.0, dtype=int)
            self.high[4::self.obs_single_size] = 2
            self.observation_space = spaces.MultiDiscrete(self.high+1)
            self.action_space = self.action_space

        # Memory type 2 = Bk
        elif self.memory_type == 2:
            # Partial observability: (wall_north, wall_east,
            # wall_south, wall_west, external memory bit, y of the true goal)
            self.obs_single_size = len(self.high)
            self.mem_single_size = 1
            # Actions are defined n e s w, add observation to the memory.
            self.obs_number_of_dimension = (self.obs_single_size +
                                            self.mem_single_size *
                                            self.memory_length)
            high = np.full(self.obs_number_of_dimension, 1.0, dtype=int)
            high[4] = 2
            high[5:] = 2
            self.observation_space = spaces.MultiDiscrete(high+1)
            self.action_space = spaces.Discrete(self.action_space.n * 3)
            self.external_memory = np.zeros(self.memory_length, dtype=int)

        # Memory type 3 = Ok
        elif self.memory_type == 3:
            # Partial observability: (wall_north, wall_east,
            # wall_south, wall_west, y of the true goal
            # and additional dimensions of size
            # memory_seq_length*observation_size)
            # Note that max. number of dimensions for a np array is 32.
            # So the maximum fixed sequence size could be 5.

            self.obs_single_size = len(self.high)
            self.mem_single_size = self.obs_single_size
            # Actions are defined n e s w, add observation to the memory.
            self.obs_number_of_dimension = (self.obs_single_size +
                                            self.mem_single_size *
                                            self.memory_length)
            self.external_memory = np.zeros(self.obs_number_of_dimension -
                                            self.obs_single_size, dtype=int)
            self.high = np.full(self.obs_number_of_dimension, 1.0, dtype=int)
            self.high[4::self.obs_single_size] = 2
            self.observation_space = spaces.MultiDiscrete(self.high+1)
            self.action_space = spaces.Discrete(self.action_space.n * 2)

        # Memory type 4 = OAk
        elif self.memory_type == 4:
            # Partial observability: (wall_north, wall_east,
            # wall_south, wall_west, y of the true goal
            # and additional dimensions of size
            # memory_seq_length*observation_size)
            # Note that max. number of dimensions for a np array is 32.
            # So the maximum fixed sequence size could be 5.

            self.action_space = spaces.Discrete(self.action_space.n * 2)
            self.obs_single_size = len(self.high)
            self.act_single_size = 1
            self.mem_single_size = self.obs_single_size + self.act_single_size

            # Actions are defined n e s w, add observation to the memory.
            self.obs_number_of_dimension = (self.obs_single_size +
                                            self.mem_single_size *
                                            self.memory_length)
            self.external_memory = np.zeros(self.obs_number_of_dimension -
                                            self.obs_single_size, dtype=int)
            self.high = np.full(self.obs_number_of_dimension, 1.0, dtype=int)
            self.high[4] = 2
            self.high[(4+self.obs_single_size)::(self.mem_single_size)] = 2
            self.high[(5+self.obs_single_size)::(
                self.mem_single_size)] = self.action_space.n
            self.observation_space = spaces.MultiDiscrete(self.high+1)

        # Memory type 5 = None (For LSTM)
        elif self.memory_type == 5:
            # Partial observability: (wall_north, wall_east,
            # wall_south, wall_west, y of the true goal)
            self.obs_number_of_dimension = 5
            self.high = np.full(5, 1.0, dtype=int)
            self.high[-1] = 2
            self.observation_space = spaces.MultiDiscrete(self.high+1)
            self.action_space = self.action_space

        # Intrinsic memory initialization
        # Intrinsic memory is only limited to the memory types 1, 3 and 4
        # since observation frequency is only meaningful if the
        # memory includes observations, otherwise there is no way
        # to calculate intrinsic reward.
        if self.memory_type != 0 and self.memory_type != 2\
                and self.memory_type != 5:
            self.intrinsic_dict = {}
            self.intrinsic_total_count = 0
            for i in range(self.memory_length):
                obs_i = np.array_str(self.external_memory[
                    i*self.mem_single_size:(i+1)*self.mem_single_size])
                obs_freq = self.intrinsic_dict.get(obs_i, 0)
                self.intrinsic_dict[obs_i] = obs_freq + 1
                self.intrinsic_total_count += 1

    def _get_observation(self):
        observation = np.zeros(self.obs_number_of_dimension, dtype=int)
        state = self.current_state
        # Partial observability

        # north direction
        if state[1] == 0 or self.grid[state[1] - 1][state[0]] == 'X':
            observation[0] = 1

        # east direction
        if (state[0] == (self.grid_size[0] - 1) or
           self.grid[state[1]][state[0] + 1] == 'X'):
            observation[1] = 1

        # south direction
        if (state[1] == (self.grid_size[1] - 1) or
           self.grid[state[1] + 1][state[0]] == 'X'):
            observation[2] = 1

        # west direction
        if state[0] == 0 or self.grid[state[1]][state[0] - 1] == 'X':
            observation[3] = 1

        # the agent can only get the true goal location either in the light
        # location or in the terminal state
        if state[0] == self.light_x or state in self.li_terminal_states:
            observation[4] = state[2]
        else:
            # otherwise, it gets a neutral direction for the true goal
            observation[4] = 1

        # Memory type 0 = None or = 5 None(For LSTM)
        if self.memory_type != 0 and self.memory_type != 5:
            observation[5:] = self.external_memory

        return observation

    def step(self, action):
        done = False
        success = 0

        # Memory type 0 = None
        if self.memory_type == 0:
            movementAction = action
            memoryAction = 0
        # Memory type 1 = Kk
        elif self.memory_type == 1:
            movementAction = action
            memoryAction = 1
        # Memory type 2 = Bk
        elif self.memory_type == 2:
            movementAction = action // 3
            memoryAction = action % 3
        # Memory type 3 = Ok
        elif self.memory_type == 3:
            movementAction = action // 2
            memoryAction = action % 2
        # Memory type 4 = OAk
        elif self.memory_type == 4:
            movementAction = action // 2
            memoryAction = action % 2
        # Memory type 5 = None (For LSTM)
        elif self.memory_type == 5:
            movementAction = action
            memoryAction = 0

        # Check if add observation to memory action or not
        if memoryAction != 0:
            self.add_observation_to_memory(memoryAction, movementAction)

        new_state, reward, done, success = self._one_agent_step(
                self.current_state, movementAction)

        intrinsic_reward = 0

        if self.memory_type != 0 and self.memory_type != 2\
                and self.memory_type != 5:

            # Memory type 1 = Kk
            if self.memory_type == 1:
                intrinsic_obs = np.array_str(self._get_observation()[
                    0:self.obs_single_size])
            # Memory type 3 = Ok
            elif self.memory_type == 3:
                intrinsic_obs = np.array_str(self._get_observation()[
                    0:self.obs_single_size])
            # Memory type 4 = OAk
            elif self.memory_type == 4:
                intrinsic_obs = np.array_str(np.append(self._get_observation()[
                    0:self.obs_single_size], [action]))

            obs_freq = self.intrinsic_dict.get(intrinsic_obs, 0)
            self.intrinsic_dict[intrinsic_obs] = obs_freq + 1
            self.intrinsic_total_count += 1

            intrinsic_reward = self.calculateIntrinsicMotivationReward()

        reward = reward + intrinsic_reward
        self.current_state = new_state

        self.episode_reward += reward
        return self._get_observation(), reward, done, {
            'success': success, 'is_success': bool(success)}

    def reset(self):
        # Memory type 1 = Kk
        if self.memory_type == 1:
            self.external_memory = np.zeros(self.obs_number_of_dimension -
                                            self.obs_single_size, dtype=int)

        # Memory type 2 = Bk
        elif self.memory_type == 2:
            self.external_memory = np.zeros(self.memory_length, dtype=int)

        # Memory type 3 = Ok
        elif self.memory_type == 3:
            self.external_memory = np.zeros(self.obs_number_of_dimension -
                                            self.obs_single_size, dtype=int)

        # Memory type 4 = OAk
        elif self.memory_type == 4:
            self.external_memory = np.zeros(self.obs_number_of_dimension -
                                            self.obs_single_size, dtype=int)

        self.current_state = random.choice(
                self.li_initial_states)
        self.episode_reward = 0

        return self._get_observation()

    def add_observation_to_memory(self, memoryAction, action):
        # The memory update strategy could be changed, for now, it is set to
        # First In First Out Strategy

        # In each add step, shift memory by self.mem_single_size and
        # add the new element into the starting of the array.

        self.external_memory = np.roll(self.external_memory,
                                       self.mem_single_size)

        # Memory type 1 = Kk
        if self.memory_type == 1:
            self.external_memory[0:self.mem_single_size] = \
                self._get_observation()[:self.obs_single_size]

        # Memory type 2 = Bk
        elif self.memory_type == 2:
            binaryAction = 1 if memoryAction == 1 else 0
            self.external_memory[0:self.mem_single_size] = binaryAction
        # Memory type 3 = Ok
        elif self.memory_type == 3:
            self.external_memory[0:self.mem_single_size] = \
                self._get_observation()[:self.obs_single_size]

        # Memory type 4 = OAk
        elif self.memory_type == 4:
            self.external_memory[0:self.mem_single_size] = \
                np.append(self._get_observation()[:self.obs_single_size],
                          [action])
        pass

    def calculateIntrinsicMotivationReward(self):
        # beta is between [0,1]
        # c is the memory capacity.
        # p_obs is the probability of gathering observation obs
        # r_int_m is bounded to [-c, 0]
        if self.intrinsic_enabled:
            total_frequencies = 0
            for i in range(self.memory_length):
                obs_i = np.array_str(self.external_memory[
                    i*self.mem_single_size:(i+1)*self.mem_single_size])
                p_obs = self.intrinsic_dict[obs_i] / self.intrinsic_total_count
                total_frequencies += 1-p_obs
            total_frequencies -= self.memory_length
            r_int_m = self.intrinsic_beta * total_frequencies
            return r_int_m
        else:
            return 0
