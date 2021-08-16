from stable_baselines3 import PPO, DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import TensorBoardOutputFormat
from UtilPolicies import MultiLayerActorCriticPolicy
from ClassQAgent import QAgent
from ClassSarsaLambdaAgent import SarsaLambdaAgent
import datetime


def train_q_agent(learning_setting):
    envClass = learning_setting['envClass']
    env = envClass(maze_length=learning_setting['maze_length'])

    model = QAgent(env, tensorboard_log=learning_setting['tb_log_dir'],
                   learning_setting=learning_setting)

    model.learn(total_timesteps=learning_setting['total_timesteps'],
                tb_log_name=learning_setting['tb_log_name'])


def train_sarsa_lambda_agent(learning_setting):
    envClass = learning_setting['envClass']
    env = envClass(maze_length=learning_setting['maze_length'])

    model = SarsaLambdaAgent(env,
                             tensorboard_log=learning_setting['tb_log_dir'],
                             learning_setting=learning_setting)

    model.learn(total_timesteps=learning_setting['total_timesteps'],
                tb_log_name=learning_setting['tb_log_name'])


def train_dqn_agent(learning_setting):
    envClass = learning_setting['envClass']
    env = envClass(maze_length=learning_setting['maze_length'])

    policy_kwargs = dict(net_arch=[learning_setting['nn_layer_size'],
                                   learning_setting['nn_layer_size']])

    model = DQN("MlpPolicy", env, verbose=0,
                tensorboard_log=learning_setting['tb_log_dir'],
                seed=learning_setting['seed'],
                policy_kwargs=policy_kwargs,
                learning_rate=learning_setting['learning_rate'],
                gamma=learning_setting['discount_rate'],
                exploration_initial_eps=learning_setting['epsilon_start'],
                exploration_final_eps=learning_setting['epsilon_end'],
                target_update_interval=learning_setting['update_interval'])

    model.learn(total_timesteps=learning_setting['total_timesteps'],
                tb_log_name=learning_setting['tb_log_name'],
                callback=TensorboardCallback())
    model.save("saves/dqn_agent_"+str(datetime.datetime.now()))


def train_ppo_agent(learning_setting):
    envClass = learning_setting['envClass']
    env = envClass(maze_length=learning_setting['maze_length'])

    policy_kwargs = dict(net_arch=[
                         dict(pi=[learning_setting['nn_layer_size'],
                                  learning_setting['nn_layer_size']],
                              vf=[learning_setting['nn_layer_size'],
                                  learning_setting['nn_layer_size']])])

    model = PPO(MultiLayerActorCriticPolicy, env, verbose=0,
                tensorboard_log=learning_setting['tb_log_dir'],
                seed=learning_setting['seed'],
                policy_kwargs=policy_kwargs,
                learning_rate=learning_setting['learning_rate'],
                gamma=learning_setting['discount_rate'])

    model.learn(total_timesteps=learning_setting['total_timesteps'],
                tb_log_name=learning_setting['tb_log_name'],
                callback=TensorboardCallback())
    model.save("saves/ppo_agent_" + str(datetime.datetime.now()))


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
            self.tb_formatter.writer.add_scalar("_tmaze/Reward per episode",
                                                epi_reward, epi_number)
            self.tb_formatter.writer.flush()
            self.tb_formatter.writer.\
                add_scalar("_tmaze/Episode length per episode",
                           self.locals['self'].env.
                           unwrapped.envs[0].
                           episode_lengths[-1],
                           epi_number)
            self.tb_formatter.writer.flush()
        return True
