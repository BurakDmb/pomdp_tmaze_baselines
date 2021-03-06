import numpy as np
from torch.utils.tensorboard import SummaryWriter


class QAgent:
    '''
    Q Learning Algorithm For Discrete Action, Discrete Observation space
    '''

    def __init__(self, env, learning_setting):
        self.learning_rate = learning_setting['learning_rate']
        self.discount_rate = learning_setting['discount_rate']
        self.epsilon_start = learning_setting['epsilon_start']
        self.epsilon_end = learning_setting['epsilon_end']
        self.epsilon = self.epsilon_start
        self.seed = learning_setting['seed']
        np.random.seed(self.seed)

        self.env = env
        self.action_size = env.action_space.n
        self.observation_space = env.observation_space
        self.log_dir = learning_setting['tb_log_dir']

        self.q_table = {}
        # observation_dims = env.observation_space.nvec
        # q_shape = np.append(observation_dims, self.action_size)
        # self.q_table = np.zeros(q_shape)

    def learn(self, total_timesteps, tb_log_name):
        self.writer = None
        if self.log_dir:
            self.writer = SummaryWriter(log_dir=self.log_dir+tb_log_name)
        self.time_step = 0
        self.episode = 0
        self.timeStepLimit = False
        self.total_timestep = total_timesteps

        while not self.timeStepLimit:
            obs = self.env.reset()
            done = False
            self.episode_reward = 0
            self.episode_step = 0
            while not done:
                action = self.pre_action(obs)
                next_obs, reward, done, _ = self.env.step(action)
                self.post_action(obs, action, reward, next_obs, done)
                obs = next_obs
                self.time_step += 1
                self.episode_step += 1
                if self.time_step > self.total_timestep:
                    self.timeStepLimit = True
                    break
                if done:
                    self.episode += 1
                    break
            if not self.timeStepLimit:
                self.post_episode()

    def pre_action(self, observation):
        self.init_q_value(observation)

        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)
        return self.get_action(observation)

    def post_action(self, observation, action, reward, next_observation, done):
        q_value = self.get_q_values(observation)[action]
        max_q_of_next = self.get_max_q_value(next_observation)
        q_update = self.learning_rate * \
            (reward + self.discount_rate * max_q_of_next - q_value)
        self.set_q_value(observation, action, q_value + q_update)

        # Setting the cumulative reward for the episode
        self.episode_reward += reward

    def get_action(self, observation):
        return np.random.choice(np.flatnonzero(self.get_q_values(observation)
                                == self.get_q_values(observation).max()))

    def init_q_value(self, observation):
        obs_key = tuple(observation)
        if obs_key not in self.q_table:
            self.q_table[obs_key] = np.zeros((self.action_size,))

    def get_q_values(self, observation):
        obs_key = tuple(observation)
        if obs_key not in self.q_table:
            self.q_table[obs_key] = np.zeros((self.action_size,))
        return self.q_table[obs_key]

    def get_max_q_value(self, observation):
        return self.get_q_values(observation).max()

    def set_q_value(self, observation, action, q_value):
        self.get_q_values(observation)[action] = q_value

    def post_episode(self):
        self.epsilon = self.epsilon_start - \
                        (self.epsilon_start - self.epsilon_end) * \
                        self.time_step / self.total_timestep
        self.success_ratio = (self.env.success_count /
                              self.env.episode_count) * 100
        if self.writer is not None:
            self.writer.add_scalar("_tmaze/Reward per episode",
                                   self.episode_reward, self.episode)
            self.writer.add_scalar("_tmaze/Episode length per episode",
                                   self.episode_step, self.episode)
            self.writer.add_scalar("_tmaze/Success Ratio per episode",
                                   self.success_ratio,
                                   self.episode)
