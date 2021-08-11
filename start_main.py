from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import TensorBoardOutputFormat
from custom_policy import CustomActorCriticPolicy
import torch.multiprocessing as mp
from t_maze_env import TMazeEnv
import numpy as np


def train():
    env = TMazeEnv(maze_length=6)
    model = PPO(CustomActorCriticPolicy, env, verbose=0,
                tensorboard_log="./logs/t_maze_tensorboard/", seed=0)
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
        # note: the failure case (not formatter found) is not handled here, should be done with try/except.
        self.tb_formatter = next(formatter for formatter in output_formats if isinstance(formatter, TensorBoardOutputFormat))

    # def _on_step(self):
    #     return True

    def _on_step(self) -> bool:
        # Log scalar value (here a random variable)
        # TODO: Check the right done flag for getting the episode reward.
        if(self.locals['dones'][-1]):
            epi_reward = self.model.env.unwrapped.unwrapped.unwrapped.envs[0].\
                            unwrapped.episode_reward
            
            self.logger.record('episode_reward', epi_reward)
            self.tb_formatter.writer.add_text("direct_access", "this is a value", self.num_timesteps)
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
