from stable_baselines3 import PPO
from custom_policy import CustomActorCriticPolicy
import torch.multiprocessing as mp
from t_maze_env import TMazeEnv


def train():
    env = TMazeEnv(maze_length=6)
    model = PPO(CustomActorCriticPolicy, env, verbose=0,
                tensorboard_log="./logs/t_maze_tensorboard/", seed=0)
    model.learn(total_timesteps=50000, tb_log_name="T-Maze-v0")


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
