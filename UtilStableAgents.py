from stable_baselines3 import PPO, DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import TensorBoardOutputFormat
from ClassQAgent import QAgent
from ClassSarsaLambdaAgent import SarsaLambdaAgent
import datetime


def train_q_agent(learning_setting):
    """
    Q-Learning Algorithm, Must-Have Learning Settings And
    Example Parameter Values:
        q_learning_setting = {}
        q_learning_setting['envClass'] = envClass
        q_learning_setting['learning_rate'] = 0.1
        q_learning_setting['discount_rate'] = 0.99
        q_learning_setting['epsilon_start'] = 0.33
        q_learning_setting['epsilon_end'] = 0.33
        q_learning_setting['tb_log_name'] = "q-tmazev0"
        q_learning_setting['tb_log_dir'] = "./logs/t_maze_tensorboard/"
        q_learning_setting['maze_length'] = maze_length
        q_learning_setting['total_timesteps'] = total_timesteps
        q_learning_setting['seed'] = None
        q_learning_setting['save'] = False
    """
    envClass = learning_setting['envClass']
    env = envClass(maze_length=learning_setting['maze_length'])

    model = QAgent(env, learning_setting=learning_setting)

    model.learn(total_timesteps=learning_setting['total_timesteps'],
                tb_log_name=learning_setting['tb_log_name'])

    # TODO: Implement saving q model
    if learning_setting['save']:
        pass

    return model


def train_sarsa_lambda_agent(learning_setting):
    """
    SARSA(Lambda) Algorithm, Must-Have Learning Settings And
    Example Parameter Values:
        sarsa_learning_setting = {}
        sarsa_learning_setting['envClass'] = envClass
        sarsa_learning_setting['learning_rate'] = 0.1
        sarsa_learning_setting['lambda_value'] = 0.9
        sarsa_learning_setting['discount_rate'] = 0.99
        sarsa_learning_setting['epsilon_start'] = 0.33
        sarsa_learning_setting['epsilon_end'] = 0.33
        sarsa_learning_setting['tb_log_name'] = "sarsa-l-tmazev0"
        sarsa_learning_setting['tb_log_dir'] = "./logs/t_maze_tensorboard/"
        sarsa_learning_setting['maze_length'] = maze_length
        sarsa_learning_setting['total_timesteps'] = total_timesteps
        sarsa_learning_setting['seed'] = None
        sarsa_learning_setting['save'] = False
    """
    envClass = learning_setting['envClass']
    env = envClass(maze_length=learning_setting['maze_length'])

    model = SarsaLambdaAgent(env, learning_setting=learning_setting)

    model.learn(total_timesteps=learning_setting['total_timesteps'],
                tb_log_name=learning_setting['tb_log_name'])

    # TODO: Implement saving sarsa model
    if learning_setting['save']:
        pass

    return model


def train_dqn_agent(learning_setting):
    """
    Deep Q-Learning Algorithm, Must-Have Learning Settings And
    Example Parameter Values:
        dqn_learning_setting = {}
        dqn_learning_setting['envClass'] = envClass
        dqn_learning_setting['learning_rate'] = 1e-3
        dqn_learning_setting['discount_rate'] = 0.99
        dqn_learning_setting['epsilon_start'] = 0.9
        dqn_learning_setting['epsilon_end'] = 0.01
        dqn_learning_setting['exploration_fraction'] = 0.5
        dqn_learning_setting['update_interval'] = 100
        dqn_learning_setting['nn_layer_size'] = 8
        dqn_learning_setting['tb_log_name'] = "dqn-tmazev0"
        dqn_learning_setting['tb_log_dir'] = "./logs/t_maze_tensorboard/"
        dqn_learning_setting['maze_length'] = maze_length
        dqn_learning_setting['total_timesteps'] = total_timesteps
        dqn_learning_setting['seed'] = None
        dqn_learning_setting['policy'] = "MlpPolicy"
        dqn_learning_setting['save'] = False
    """
    envClass = learning_setting['envClass']
    env = envClass(maze_length=learning_setting['maze_length'])

    policy_kwargs = dict(net_arch=[learning_setting['nn_layer_size'],
                                   learning_setting['nn_layer_size']])

    model = DQN(learning_setting['policy'], env, verbose=0,
                tensorboard_log=learning_setting['tb_log_dir'],
                seed=learning_setting['seed'],
                policy_kwargs=policy_kwargs,
                learning_rate=learning_setting['learning_rate'],
                gamma=learning_setting['discount_rate'],
                exploration_initial_eps=learning_setting['epsilon_start'],
                exploration_final_eps=learning_setting['epsilon_end'],
                target_update_interval=learning_setting['update_interval'],
                exploration_fraction=learning_setting['exploration_fraction'])

    model.learn(total_timesteps=learning_setting['total_timesteps'],
                tb_log_name=learning_setting['tb_log_name'],
                callback=TensorboardCallback())

    if learning_setting['save']:
        model.save("saves/dqn_agent_"+str(datetime.datetime.now()))

    return model


def train_ppo_agent(learning_setting):
    """
    PPO(Proximal Policy Optimization Algorithm, Must-Have Learning Settings And
    Example Parameter Values:
        ppo_learning_setting = {}
        ppo_learning_setting['envClass'] = envClass
        ppo_learning_setting['learning_rate'] = 1e-3
        ppo_learning_setting['discount_rate'] = 0.99
        ppo_learning_setting['nn_layer_size'] = 8
        ppo_learning_setting['tb_log_name'] = "ppo-tmazev0"
        ppo_learning_setting['tb_log_dir'] = "./logs/t_maze_tensorboard/"
        ppo_learning_setting['maze_length'] = maze_length
        ppo_learning_setting['total_timesteps'] = total_timesteps
        ppo_learning_setting['seed'] = None
        ppo_learning_setting['policy'] = "MlpPolicy"
        ppo_learning_setting['save'] = False
    """
    envClass = learning_setting['envClass']
    env = envClass(maze_length=learning_setting['maze_length'])

    policy_kwargs = dict(net_arch=[
                         dict(pi=[learning_setting['nn_layer_size'],
                                  learning_setting['nn_layer_size']],
                              vf=[learning_setting['nn_layer_size'],
                                  learning_setting['nn_layer_size']])])

    model = PPO(learning_setting['policy'], env, verbose=0,
                tensorboard_log=learning_setting['tb_log_dir'],
                seed=learning_setting['seed'],
                policy_kwargs=policy_kwargs,
                learning_rate=learning_setting['learning_rate'],
                gamma=learning_setting['discount_rate'])

    model.learn(total_timesteps=learning_setting['total_timesteps'],
                tb_log_name=learning_setting['tb_log_name'],
                callback=TensorboardCallback())

    if learning_setting['save']:
        model.save("saves/ppo_agent_" + str(datetime.datetime.now()))

    return model


class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)

    def _on_training_start(self):
        output_formats = self.logger.output_formats
        # Save reference to tensorboard formatter object
        # note: the failure case (not formatter found) is
        # not handled here, should be done with try/except.
        self.tb_formatter = next(formatter for formatter in output_formats
                                 if isinstance(formatter,
                                               TensorBoardOutputFormat))

    def _on_step(self) -> bool:
        if(self.locals['self'].env.buf_dones[-1]):
            epi_reward = self.model.env.unwrapped.envs[0].episode_returns[-1]
            epi_number = len(self.locals['self'].env.unwrapped.envs[0].
                             episode_lengths)
            success_ratio = (self.locals['self'].env.unwrapped.envs[0].
                             success_count /
                             self.locals['self'].env.unwrapped.envs[0].
                             episode_count) * 100
            self.tb_formatter.writer.add_scalar("_tmaze/Reward per episode",
                                                epi_reward, epi_number)
            self.tb_formatter.writer.flush()
            self.tb_formatter.writer.\
                add_scalar("_tmaze/Episode length per episode",
                           self.locals['self'].env.
                           unwrapped.envs[0].
                           episode_lengths[-1],
                           epi_number)
            self.tb_formatter.writer.\
                add_scalar("_tmaze/Success Ratio per episode",
                           success_ratio, epi_number)
            self.tb_formatter.writer.flush()
        return True
