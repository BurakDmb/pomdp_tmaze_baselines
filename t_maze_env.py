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

        self._initialize_observation_space()

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

        return self._get_observation(), reward, done, {}

    def reset(self):
        self.current_state = random.choice(
                self.li_initial_states)
        return self._get_observation()

    def _initialize_observation_space(self):
        # Full observability: (x, y, y of the true goal)
        low = np.zeros(3, dtype=int)
        high = np.zeros(3, dtype=int)
        high[0] = self.grid_size[0] - 1
        high[1] = self.grid_size[1] - 1
        high[2] = 2
        self.observation_space = spaces.Box(low, high, dtype=np.int32)

    def get_observation_space_size(self):
        return self.observation_space.shape[0]

    def get_action_space_size(self):
        return self.action_space.n

    def _render(self, mode='human', close=False):
        pass

    def _get_observation(self):
        return self.current_state
