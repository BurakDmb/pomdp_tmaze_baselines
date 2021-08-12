import gym
from gym import spaces
import sys
import numpy as np
import random


class TMazeEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, **kwargs):

        self.grid_size = (kwargs.get('maze_length'), 3)
        self.grid = ['X' * (self.grid_size[0] - 1) + '_',
                     '_' * (self.grid_size[0]),
                     'X' * (self.grid_size[0] - 1) + "_"]

        # Rewards
        self.fl_default_reward = -0.1
        self.fl_false_goal_reward = -0.1
        self.fl_true_goal_reward = 4.0
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
        low = np.zeros(3, dtype=int)
        high = np.zeros(3, dtype=int)
        high[0] = self.grid_size[0] - 1
        high[1] = self.grid_size[1] - 1
        high[2] = 2
        self.observation_space = spaces.Box(low, high, dtype=np.int32)

        self.episode_reward = 0

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

        assert(sum(possible_states.values()) == 1.0)
        next_state = random.choices(
            list(possible_states.keys()), list(possible_states.values()))[0]
        done = False
        success = 0
        if next_state in self.li_terminal_states:
            done = True
            if next_state[1] == next_state[2]:
                reward = self.fl_true_goal_reward
                success = 1
            else:
                reward = self.fl_false_goal_reward
        return next_state, reward, done, success

    # Actions: n e s w
    def step(self, action):
        new_state, reward, done, success = self._one_agent_step(
            self.current_state, action)

        self.current_state = new_state
        self.episode_reward += reward
        return self._get_observation(), reward, done, {}

    def reset(self):
        self.current_state = random.choice(
                self.li_initial_states)
        self.episode_reward = 0
        return self._get_observation()

    def get_observation_space_size(self):
        return self.observation_space.shape[0]

    def get_action_space_size(self):
        return self.action_space.n

    def _render(self, mode='human', close=False):
        pass

    def _get_observation(self):
        return self.current_state


class TMazeEnvV1(TMazeEnv):
    def __init__(self, **kwargs):
        super(TMazeEnvV1, self).__init__(**kwargs)
        # print("Tmazev1: ", kwargs.get('arg1'))
        self.light_x = 0

        # Partial observability: (wall_north, wall_east,
        # wall_south, wall_west, y of the true goal)
        low = np.zeros(5, dtype=int)
        high = np.full(5, 1.0, dtype=int)
        high[-1] = 2
        self.observation_space = spaces.Box(low, high, dtype=np.int32)

    # Needs Update
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
        # location or in the terminal state
        if state[0] == self.light_x or state in self.li_terminal_states:
            observation[-1] = state[2]
        else:
            # otherwise, it gets a neutral direction for the true goal
            observation[-1] = 1
        return observation


# partial observation version with the exact observations as the paper
class TMazeEnvV2(TMazeEnvV1):
    def __init__(self, **kwargs):
        super(TMazeEnvV2, self).__init__(**kwargs)

        '''
        Partial observability:
        - At the start; 011 or 110
        - The corridor; 101
        - T-junction; 010
        - Upper-corner; 111
        - Lower-corner; 000
        '''
        low = np.zeros(3, dtype=int)
        high = np.full(3, 1.0, dtype=int)
        self.observation_space = spaces.Box(low, high, dtype=np.int32)

    def _get_observation(self):
        observation = np.zeros(3, dtype=int)
        state = self.current_state
        # Partial observability
        if state[0] == 0:   # at the start
            if state[2] == 0:
                observation = np.asarray([0, 1, 1])
            elif state[2] == 2:
                observation = np.asarray([1, 1, 0])
        elif 0 < state[0] < self.grid_size[0] - 1:  # at the corridor
            observation = np.asarray([1, 0, 1])
        elif state[0] == self.grid_size[0] - 1:
            if state[1] == 1:  # at T-junction
                observation = np.asarray([0, 1, 0])
            elif state[1] == 0:  # upper corner
                observation = np.asarray([1, 1, 1])
            elif state[1] == 2:  # lower corner
                observation = np.asarray([0, 0, 0])
        return observation


# the full observable version with one hot vector of the states
class TMazeEnvV3(TMazeEnv):
    def __init__(self, **kwargs):
        super(TMazeEnvV3, self).__init__(**kwargs)

        self.di_state_map = {}
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                for k in [0, 2]:
                    s = (i, j, k)
                    if self._is_state_available(s):
                        self.di_state_map[s] = len(self.di_state_map.keys())

        # Full observability with one hot vectors
        in_number_of_states = len(self.di_state_map.keys())

        low = np.zeros(in_number_of_states, dtype=int)
        high = np.full(in_number_of_states, 1.0, dtype=int)
        self.observation_space = spaces.Box(low, high, dtype=np.int32)

    def _get_observation(self):
        state = self.current_state
        in_number_of_states = len(self.di_state_map.keys())
        observation = np.zeros(in_number_of_states, dtype=int)

        observation[self.di_state_map[state]] = 1   # one hot vector

        return observation

    def _get_observations_dict(self):
        obs = self._get_observation()
        return obs


# the partially observable version with one hot vector of the observations
class TMazeEnvV4(TMazeEnv):
    def __init__(self, **kwargs):
        super(TMazeEnvV4, self).__init__(**kwargs)
        self.light_x = 0
        # Terminal states
        self.li_terminal_states = [
            (self.grid_size[0] - 1, 0, 0), (self.grid_size[0] - 1, 0, 2),
            (self.grid_size[0] - 1, 2, 0), (self.grid_size[0] - 1, 2, 2)]

        self.di_observation_map = {}
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                for k in [0, 2]:
                    s = (i, j, k)
                    o = tuple(self._get_observation_o(s))
                    if self._is_state_available(s) and o not in \
                            self.di_observation_map.keys():

                        self.di_observation_map[o] = len(
                            self.di_observation_map.keys())

        # Partial observability with one hot vectors
        in_number_of_observations = len(self.di_observation_map.keys())

        low = np.zeros(in_number_of_observations, dtype=int)
        high = np.full(in_number_of_observations, 1.0, dtype=int)
        self.observation_space = spaces.Box(low, high, dtype=np.int32)

    def _get_observation_o(self, state):

        observation = np.zeros(5, dtype=int)

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

        # the agent can only get the true goal location either in the
        #  light location or in the terminal state
        if state[0] == self.light_x or state in self.li_terminal_states:
            observation[-1] = state[2]
        else:
            # otherwise, it gets a neutral direction for the true goal
            observation[-1] = 1

        return observation

    def _get_agent_one_hot_observation(self):
        in_number_of_observations = len(self.di_observation_map.keys())

        o = tuple(self._get_observation_o(self.current_state))

        observation = np.zeros(in_number_of_observations, dtype=int)
        observation[self.di_observation_map[o]] = 1   # one hot vector

        return observation

    def _get_observation(self):
        obs = self._get_agent_one_hot_observation()
        return obs


# Partial observable version with external memory wrapper
class TMazeEnvV5(TMazeEnvV1):
    def __init__(self, **kwargs):
        super(TMazeEnvV5, self).__init__(**kwargs)

        # Partial observability: (wall_north, wall_east,
        # wall_south, wall_west, external memory bit, y of the true goal)
        low = np.zeros(6, dtype=int)
        high = np.full(6, 1.0, dtype=int)
        high[4] = 1
        high[-1] = 2
        self.observation_space = spaces.Box(low, high, dtype=np.int32)
        self.action_space = spaces.Discrete(6)
        self.memory_bit = 0

    def _get_observation(self):
        observation = np.zeros(6, dtype=int)
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
            observation[-1] = state[2]
        else:
            # otherwise, it gets a neutral direction for the true goal
            observation[-1] = 1

        observation[4] = self.memory_bit

        return observation

    def _get_observations_dict(self):
        obs = self._get_observation()
        return obs

    def step(self, action):
        new_state = self.current_state
        reward = 0
        # if the action is movement action then do the
        # same as original environment
        if action in range(4):

            new_state, reward, done, success = self._one_agent_step(
                self.current_state, action)
            self.current_state = new_state
            self.episode_reward += reward

        # if the action is not movement action, it is bit clear/set actions.
        # then do not move the agent, set the reward as zero and
        # clear/set the bit.
        else:
            # TODO: add a condition for punishing the agents
            # for not going the end.

            # if self.memory_bit == 1:
            #     reward = -2 * self.fl_default_reward
            # else:
            #     reward = 5 * self.fl_default_reward
            done = False

            if action == 4:
                self.memory_bit = 0
            elif action == 5:
                self.memory_bit = 1

        self.current_state = new_state
        self.episode_reward += reward
        return self._get_observations_dict(), reward, done, {}

    def reset(self):
        self.memory_bit = 0
        self.current_state = random.choice(
                self.li_initial_states)
        self.episode_reward = 0
        return self._get_observation()


# Partial observable version with external memory wrapper
# (with cross product of memory actions)
class TMazeEnvV6(TMazeEnvV1):
    def __init__(self, **kwargs):
        super(TMazeEnvV6, self).__init__(**kwargs)

        # Partial observability: (wall_north, wall_east,
        # wall_south, wall_west, external memory bit, y of the true goal)
        low = np.zeros(6, dtype=int)
        high = np.full(6, 1.0, dtype=int)
        high[4] = 1
        high[-1] = 2
        self.observation_space = spaces.Box(low, high, dtype=np.int32)
        self.cl_action_space = spaces.Discrete(12)
        self.memory_bit = 0

    def _get_observation(self):
        observation = np.zeros(6, dtype=int)
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
            observation[-1] = state[2]
        else:
            # otherwise, it gets a neutral direction for the true goal
            observation[-1] = 1

        observation[4] = self.memory_bit
        return observation

    def _get_observations_dict(self):
        obs = self._get_observation()
        return obs

    def step(self, action):
        new_state = self.current_state
        reward = 0

        # if the action is movement action then do the same
        # as original environment
        movementAction = action // 3
        memoryAction = action % 3

        new_state, reward, done, success = self._one_agent_step(
                self.current_state, movementAction)

        # Memory actions - 0: No Operation 1: clear bit, 2: Set bit.
        if memoryAction == 1:
            self.memory_bit = 0
        elif memoryAction == 2:
            self.memory_bit = 1

        return self._get_observations_dict(), reward, done, {}

    def reset(self):
        self.memory_bit = 0
        self.current_state = random.choice(
                self.li_initial_states)
        self.episode_reward = 0
        return self._get_observation()
