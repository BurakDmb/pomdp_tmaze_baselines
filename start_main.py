import torch.multiprocessing as mp
from EnvTMaze import TMazeEnv
from UtilStableAgents import train_ppo_agent, train_q_agent, train_dqn_agent
# import numpy as np


def train(nn_layer_size):
    # train_q_agent(TMazeEnv, nn_layer_size=8, total_timesteps=500000,
    #               maze_length=6, tb_log_name="q-tmazev0-")
    train_dqn_agent(TMazeEnv, nn_layer_size=8, total_timesteps=500000,
                    maze_length=6, tb_log_name="dqn-tmazev0")
    # train_ppo_agent(TMazeEnv, nn_layer_size=8, total_timesteps=500000,
    #                 maze_length=6, tb_log_name="ppo-tmazev0")


if __name__ == '__main__':

    mp.set_start_method('spawn')
    number_of_parallel_experiments = 1
    processes = []
    for rank in range(number_of_parallel_experiments):
        p = mp.Process(target=train, args=(8,))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
