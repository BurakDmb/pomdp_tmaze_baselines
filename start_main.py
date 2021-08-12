from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import TensorBoardOutputFormat
from MultiLayerPolicyClass import MultiLayerActorCriticPolicy
import torch.multiprocessing as mp
from t_maze_env import TMazeEnv
# import numpy as np


def train():
    env = TMazeEnv(maze_length=6)
    model = PPO(MultiLayerActorCriticPolicy, env, verbose=0,
                tensorboard_log="./logs/t_maze_tensorboard/", seed=0,)
    model.learn(total_timesteps=50000, tb_log_name="T-Maze-v0",
                callback=TensorboardCallback())


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

    # def _on_step(self):
    #     return True

    def _on_step(self) -> bool:
        # Log scalar value (here a random variable)
        # TODO: Check the right done flag for getting the episode reward.
        if(self.locals['dones'][-1]):
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


if __name__ == '__main__':

    mp.set_start_method('spawn')
    number_of_parallel_experiments = 1
    processes = []
    for rank in range(number_of_parallel_experiments):
        p = mp.Process(target=train)
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
