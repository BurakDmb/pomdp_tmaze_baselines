import numpy as np
import datetime
from torch.utils.tensorboard import SummaryWriter


class SarsaLambdaAgent:
    '''
    Sarsa(Lambda) Learning Algorithm For Discrete Action,
    Discrete Observation Space
    Algorithm: Sutton, Barto, "Reinforcement Learning: An Introduction"
    http://incompleteideas.net/book/first/ebook/node77.html
    https://www.researchgate.net/publication/221656193_Robustness_Analysis_of_SARSAlambda_Different_Models_of_Reward_and_Initialisation
    '''

    def __init__(self, env, learning_setting):
        self.learning_rate = learning_setting['learning_rate']
        self.lambda_value = learning_setting['lambda_value']
        self.discount_rate = learning_setting['discount_rate']
        self.epsilon_start = learning_setting['epsilon_start']
        self.epsilon_end = learning_setting['epsilon_end']
        self.epsilon = self.epsilon_start
        self.seed = learning_setting['seed']
        np.random.seed(self.seed)

        self.env = env
        self.action_size = env.action_space.n
        self.observation_space = env.get_observation_space_size()
        self.log_dir = learning_setting['tb_log_dir']
        # self.q_table = {}
        # self.e_table = {}
        observation_dims = env.observation_space.nvec
        q_shape = np.append(observation_dims, self.action_size)
        self.q_table = np.zeros(q_shape)
        self.e_table = np.zeros(q_shape)

    def learn(self, total_timesteps, tb_log_name):
        self.writer = SummaryWriter(log_dir=self.log_dir+tb_log_name
                                    + "-"
                                    + str(datetime.datetime.now()))
        self.time_step = 0
        self.episode = 0
        self.timeStepLimit = False
        self.total_timestep = total_timesteps

        while not self.timeStepLimit:
            obs = self.env.reset()
            done = False
            self.episode_reward = 0
            self.episode_step = 0
            action = self.pre_action(obs)
            while not done:
                next_obs, reward, done, _ = self.env.step(action)
                next_action = self.pre_action(next_obs)
                self.post_action(obs, action, reward, next_obs,
                                 next_action, done)
                obs = next_obs
                action = next_action
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

        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)
        return self.get_action(observation)

    def post_action(self, observation, action, reward, next_observation,
                    next_action, done):

        # Initially all e(s,a) values starts as zero, so if specific s,a
        # tuple has been not explored, the value of the e keeps zero,
        # therefore the update does not change anything. So, we can only
        # update for the items in e_table in order to calculate for all s,a

        delta = reward + self.discount_rate * \
            self.get_q_values(next_observation)[next_action] - \
            self.get_q_values(observation)[action]

        # self.set_e_value(observation, action,
        #                  self.get_e_values(observation)[action] + 1)

        self.set_e_value(observation, action, 1)

        it = np.nditer(self.e_table, flags=['multi_index'])
        for x in it:
            if self.e_table[it.multi_index] == 0:
                continue

            self.q_table[it.multi_index] = (self.q_table[it.multi_index] +
                                            self.learning_rate * delta *
                                            self.e_table[it.multi_index])
            self.e_table[it.multi_index] = (self.discount_rate *
                                            self.lambda_value *
                                            self.e_table[it.multi_index])

        # Setting the cumulative reward for the episode
        self.episode_reward += reward

    def get_action(self, observation):
        return np.random.choice(np.flatnonzero(self.get_q_values(observation)
                                == self.get_q_values(observation).max()))

    def get_q_values(self, observation):
        return self.q_table[tuple(observation)]

    def get_e_values(self, observation):
        return self.e_table[tuple(observation)]

    def get_max_q_value(self, observation):
        return self.get_q_values(observation).max()

    def set_q_value(self, observation, action, q_value):
        self.q_table[tuple(observation)+tuple([action])] = q_value

    def set_e_value(self, observation, action, e_value):
        self.e_table[tuple(observation)+tuple([action])] = e_value

    def post_episode(self):
        self.epsilon = self.epsilon_start - \
                        (self.epsilon_start - self.epsilon_end) * \
                        self.time_step / self.total_timestep
        self.success_ratio = (self.env.success_count /
                              self.env.episode_count) * 100
        self.writer.add_scalar("_tmaze/Reward per episode",
                               self.episode_reward, self.episode)
        self.writer.add_scalar("_tmaze/Episode length per episode",
                               self.episode_step, self.episode)
        self.writer.add_scalar("_tmaze/Success Ratio per episode",
                               self.success_ratio,
                               self.episode)
