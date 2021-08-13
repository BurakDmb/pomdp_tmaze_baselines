import torch.multiprocessing as mp
from EnvTMaze import TMazeEnv
from UtilStableAgents import train_ppo_agent, train_q_agent
from UtilStableAgents import train_dqn_agent, train_sarsa_lambda_agent


if __name__ == '__main__':

    mp.set_start_method('spawn')
    number_of_parallel_experiments = 1
    processes = []
    for rank in range(number_of_parallel_experiments):
        p1 = mp.Process(target=train_q_agent,
                        kwargs={'envClass': TMazeEnv,
                                'total_timesteps': 500000, 'maze_length': 6,
                                'tb_log_name': "q-tmazev0"})
        p2 = mp.Process(target=train_sarsa_lambda_agent,
                        kwargs={'envClass': TMazeEnv,
                                'total_timesteps': 500000, 'maze_length': 6,
                                'tb_log_name': "sarsalambda-tmazev0"})
        p3 = mp.Process(target=train_dqn_agent,
                        kwargs={'envClass': TMazeEnv, 'nn_layer_size': 8,
                                'total_timesteps': 500000, 'maze_length': 6,
                                'tb_log_name': "dqn-tmazev0"})
        p4 = mp.Process(target=train_ppo_agent,
                        kwargs={'envClass': TMazeEnv, 'nn_layer_size': 8,
                                'total_timesteps': 500000, 'maze_length': 6,
                                'tb_log_name': "ppo-tmazev0"})
        p1.start()
        p2.start()
        p3.start()
        p4.start()
        processes.append(p1)
        processes.append(p2)
        processes.append(p3)
        processes.append(p4)
    for p in processes:
        p.join()
