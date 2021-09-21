import torch.multiprocessing as mp
from EnvTMaze import TMazeEnv
from UtilStableAgents import train_ppo_agent, train_q_agent
from UtilStableAgents import train_dqn_agent, train_sarsa_lambda_agent
from UtilStableAgents import train_a2c_agent
from UtilPolicies import MlpACPolicy
from UtilPolicies import MlpDQNPolicy
from UtilPolicies import QLSTMPolicy
from UtilPolicies import LSTMACPolicy


if __name__ == '__main__':

    mp.set_start_method('spawn')
    number_of_parallel_experiments = 1
    processes = []
    total_timesteps = 500000
    maze_length = 6
    envClass = TMazeEnv

    q_learning_setting = {}
    q_learning_setting['envClass'] = envClass
    q_learning_setting['learning_rate'] = 0.1
    q_learning_setting['discount_rate'] = 0.99
    q_learning_setting['epsilon_start'] = 0.99
    q_learning_setting['epsilon_end'] = 0.01
    q_learning_setting['tb_log_name'] = "q-tmazev0"
    q_learning_setting['tb_log_dir'] = "./logs/t_maze_tensorboard/"
    q_learning_setting['maze_length'] = maze_length
    q_learning_setting['total_timesteps'] = total_timesteps
    q_learning_setting['seed'] = None
    q_learning_setting['save'] = False

    sarsa_learning_setting = {}
    sarsa_learning_setting['envClass'] = envClass
    sarsa_learning_setting['learning_rate'] = 0.1
    sarsa_learning_setting['lambda_value'] = 0.9
    sarsa_learning_setting['discount_rate'] = 0.99
    sarsa_learning_setting['epsilon_start'] = 0.99
    sarsa_learning_setting['epsilon_end'] = 0.01
    sarsa_learning_setting['tb_log_name'] = "sarsa-l-tmazev0"
    sarsa_learning_setting['tb_log_dir'] = "./logs/t_maze_tensorboard/"
    sarsa_learning_setting['maze_length'] = maze_length
    sarsa_learning_setting['total_timesteps'] = total_timesteps
    sarsa_learning_setting['seed'] = None
    sarsa_learning_setting['save'] = False

    dqn_learning_setting = {}
    dqn_learning_setting['envClass'] = envClass
    dqn_learning_setting['learning_rate'] = 1e-3
    dqn_learning_setting['discount_rate'] = 0.99
    dqn_learning_setting['epsilon_start'] = 0.9
    dqn_learning_setting['epsilon_end'] = 0.01
    dqn_learning_setting['exploration_fraction'] = 0.5
    dqn_learning_setting['update_interval'] = 100
    dqn_learning_setting['learning_starts'] = 5000
    dqn_learning_setting['buffer_size'] = 10000
    dqn_learning_setting['nn_layer_size'] = 8
    dqn_learning_setting['tb_log_name'] = "dqn-tmazev0"
    dqn_learning_setting['tb_log_dir'] = "./logs/t_maze_tensorboard/"
    dqn_learning_setting['maze_length'] = maze_length
    dqn_learning_setting['total_timesteps'] = total_timesteps
    dqn_learning_setting['seed'] = None
    dqn_learning_setting['policy'] = MlpDQNPolicy
    dqn_learning_setting['save'] = False
    dqn_learning_setting['device'] = 'cuda:0'

    qlstm_learning_setting = {}
    qlstm_learning_setting['envClass'] = envClass
    qlstm_learning_setting['learning_rate'] = 1e-3
    qlstm_learning_setting['discount_rate'] = 0.99
    qlstm_learning_setting['epsilon_start'] = 0.9
    qlstm_learning_setting['epsilon_end'] = 0.01
    qlstm_learning_setting['exploration_fraction'] = 0.5
    qlstm_learning_setting['update_interval'] = 100
    qlstm_learning_setting['learning_starts'] = 5000
    qlstm_learning_setting['buffer_size'] = 10000
    qlstm_learning_setting['nn_layer_size'] = 8
    qlstm_learning_setting['tb_log_name'] = "qlstm-tmazev0"
    qlstm_learning_setting['tb_log_dir'] = "./logs/t_maze_tensorboard/"
    qlstm_learning_setting['maze_length'] = maze_length
    qlstm_learning_setting['total_timesteps'] = total_timesteps
    qlstm_learning_setting['seed'] = None
    qlstm_learning_setting['policy'] = QLSTMPolicy
    qlstm_learning_setting['save'] = False
    qlstm_learning_setting['device'] = 'cuda:0'

    ppo_learning_setting = {}
    ppo_learning_setting['envClass'] = envClass
    ppo_learning_setting['learning_rate'] = 1e-3
    ppo_learning_setting['discount_rate'] = 0.99
    ppo_learning_setting['nn_layer_size'] = 8
    ppo_learning_setting['n_steps'] = 2048
    ppo_learning_setting['tb_log_name'] = "ppo-tmazev0"
    ppo_learning_setting['tb_log_dir'] = "./logs/t_maze_tensorboard/"
    ppo_learning_setting['maze_length'] = maze_length
    ppo_learning_setting['total_timesteps'] = total_timesteps
    ppo_learning_setting['seed'] = None
    ppo_learning_setting['policy'] = MlpACPolicy
    ppo_learning_setting['save'] = False
    ppo_learning_setting['device'] = 'cuda:1'

    ppoLSTM_learning_setting = {}
    ppoLSTM_learning_setting['envClass'] = envClass
    ppoLSTM_learning_setting['learning_rate'] = 1e-3
    ppoLSTM_learning_setting['discount_rate'] = 0.99
    ppoLSTM_learning_setting['nn_layer_size'] = 8
    ppoLSTM_learning_setting['n_steps'] = 2048
    ppoLSTM_learning_setting['tb_log_name'] = "ppoLSTM-tmazev0"
    ppoLSTM_learning_setting['tb_log_dir'] = "./logs/t_maze_tensorboard/"
    ppoLSTM_learning_setting['maze_length'] = maze_length
    ppoLSTM_learning_setting['total_timesteps'] = total_timesteps
    ppoLSTM_learning_setting['seed'] = None
    ppoLSTM_learning_setting['policy'] = LSTMACPolicy
    ppoLSTM_learning_setting['save'] = False
    ppoLSTM_learning_setting['device'] = 'cuda:2'

    a2c_learning_setting = {}
    a2c_learning_setting['envClass'] = envClass
    a2c_learning_setting['learning_rate'] = 1e-3
    a2c_learning_setting['discount_rate'] = 0.99
    a2c_learning_setting['nn_layer_size'] = 8
    a2c_learning_setting['n_steps'] = 50
    a2c_learning_setting['tb_log_name'] = "a2c-tmazev0"
    a2c_learning_setting['tb_log_dir'] = "./logs/t_maze_tensorboard/"
    a2c_learning_setting['maze_length'] = maze_length
    a2c_learning_setting['total_timesteps'] = total_timesteps
    a2c_learning_setting['seed'] = None
    a2c_learning_setting['policy'] = "MlpPolicy"
    a2c_learning_setting['save'] = False
    a2c_learning_setting['device'] = 'cuda:3'

    # Change the flags to True/False for only running specific agents
    start_q,\
        start_sarsa,\
        start_dqn,\
        start_qlstm,\
        start_ppo,\
        start_ppoLSTM,\
        start_a2c = True, True, True, True, True, True, True

    for rank in range(number_of_parallel_experiments):

        if start_q:
            p1 = mp.Process(target=train_q_agent,
                            kwargs={'learning_setting': q_learning_setting})
            p1.start()
            processes.append(p1)

        if start_sarsa:
            p2 = mp.Process(target=train_sarsa_lambda_agent,
                            kwargs={'learning_setting': sarsa_learning_setting}
                            )
            p2.start()
            processes.append(p2)

        if start_dqn:
            p3 = mp.Process(target=train_dqn_agent,
                            kwargs={'learning_setting': dqn_learning_setting})
            p3.start()
            processes.append(p3)

        if start_qlstm:
            p4 = mp.Process(target=train_dqn_agent,
                            kwargs={'learning_setting': qlstm_learning_setting}
                            )
            p4.start()
            processes.append(p4)

        if start_ppo:
            p5 = mp.Process(target=train_ppo_agent,
                            kwargs={'learning_setting': ppo_learning_setting})
            p5.start()
            processes.append(p5)

        if start_ppoLSTM:
            p6 = mp.Process(target=train_ppo_agent,
                            kwargs={'learning_setting':
                                    ppoLSTM_learning_setting})
            p6.start()
            processes.append(p6)

        if start_a2c:
            p7 = mp.Process(target=train_a2c_agent,
                            kwargs={'learning_setting':
                                    a2c_learning_setting})
            p7.start()
            processes.append(p7)

    for p in processes:
        p.join()
