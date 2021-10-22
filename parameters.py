from EnvTMaze import TMazeEnvV8
from UtilPolicies import MlpACPolicy
from UtilPolicies import MlpDQNPolicy
from UtilPolicies import QLSTMPolicy
from UtilPolicies import LSTMACPolicy

total_timesteps = 5_000_000
maze_length = 10
envClass = TMazeEnvV8

number_of_parallel_experiments = 3

# Change the flags to True/False for only running specific agents
start_q = True
start_sarsa_low = True
start_sarsa_high = True
start_dqn = True
start_qlstm = True
start_ppo = True
start_ppoLSTM = True
start_a2c = True

q_learning_setting = {}
q_learning_setting['envClass'] = envClass
q_learning_setting['learning_rate'] = 1e-1
q_learning_setting['discount_rate'] = 0.99
q_learning_setting['epsilon_start'] = 0.3
q_learning_setting['epsilon_end'] = 0.1
q_learning_setting['memory_seq_length'] = 1
q_learning_setting['tb_log_name'] = "q-tmazev0"
q_learning_setting['tb_log_dir'] = "./logs/t_maze_tensorboard/"
q_learning_setting['maze_length'] = maze_length
q_learning_setting['total_timesteps'] = total_timesteps
q_learning_setting['seed'] = None
q_learning_setting['save'] = False

sarsa_low_l_learning_setting = {}
sarsa_low_l_learning_setting['envClass'] = envClass
sarsa_low_l_learning_setting['learning_rate'] = 1e-2
sarsa_low_l_learning_setting['lambda_value'] = 0.8
sarsa_low_l_learning_setting['discount_rate'] = 0.99
sarsa_low_l_learning_setting['epsilon_start'] = 0.3
sarsa_low_l_learning_setting['epsilon_end'] = 0.1
sarsa_low_l_learning_setting['memory_seq_length'] = 1
sarsa_low_l_learning_setting['tb_log_name'] = "sarsa_low_l-tmazev0"
sarsa_low_l_learning_setting['tb_log_dir'] = "./logs/t_maze_tensorboard/"
sarsa_low_l_learning_setting['maze_length'] = maze_length
sarsa_low_l_learning_setting['total_timesteps'] = total_timesteps
sarsa_low_l_learning_setting['seed'] = None
sarsa_low_l_learning_setting['save'] = False


sarsa_high_l_learning_setting = {}
sarsa_high_l_learning_setting['envClass'] = envClass
sarsa_high_l_learning_setting['learning_rate'] = 1e-2
sarsa_high_l_learning_setting['lambda_value'] = 0.99
sarsa_high_l_learning_setting['discount_rate'] = 0.99
sarsa_high_l_learning_setting['epsilon_start'] = 0.3
sarsa_high_l_learning_setting['epsilon_end'] = 0.1
sarsa_high_l_learning_setting['memory_seq_length'] = 1
sarsa_high_l_learning_setting['tb_log_name'] = "sarsa_high_l-tmazev0"
sarsa_high_l_learning_setting['tb_log_dir'] = "./logs/t_maze_tensorboard/"
sarsa_high_l_learning_setting['maze_length'] = maze_length
sarsa_high_l_learning_setting['total_timesteps'] = total_timesteps
sarsa_high_l_learning_setting['seed'] = None
sarsa_high_l_learning_setting['save'] = False

dqn_learning_setting = {}
dqn_learning_setting['envClass'] = envClass
dqn_learning_setting['learning_rate'] = 1e-3
dqn_learning_setting['discount_rate'] = 0.99
dqn_learning_setting['epsilon_start'] = 0.3
dqn_learning_setting['epsilon_end'] = 0.1
dqn_learning_setting['memory_seq_length'] = 1
dqn_learning_setting['exploration_fraction'] = 0.8
dqn_learning_setting['update_interval'] = 256
dqn_learning_setting['learning_starts'] = 512
dqn_learning_setting['buffer_size'] = 2048
dqn_learning_setting['nn_layer_size'] = 8
dqn_learning_setting['tb_log_name'] = "dqn-tmazev0"
dqn_learning_setting['tb_log_dir'] = "./logs/t_maze_tensorboard/"
dqn_learning_setting['maze_length'] = maze_length
dqn_learning_setting['total_timesteps'] = total_timesteps
dqn_learning_setting['seed'] = None
dqn_learning_setting['policy'] = MlpDQNPolicy
dqn_learning_setting['save'] = False
dqn_learning_setting['device'] = 'cpu'

qlstm_learning_setting = {}
qlstm_learning_setting['envClass'] = envClass
qlstm_learning_setting['learning_rate'] = 1e-3
qlstm_learning_setting['discount_rate'] = 0.99
qlstm_learning_setting['epsilon_start'] = 0.3
qlstm_learning_setting['epsilon_end'] = 0.1
qlstm_learning_setting['memory_seq_length'] = 1
qlstm_learning_setting['exploration_fraction'] = 0.8
qlstm_learning_setting['update_interval'] = 256
qlstm_learning_setting['learning_starts'] = 512
qlstm_learning_setting['buffer_size'] = 2048
qlstm_learning_setting['nn_layer_size'] = 8
qlstm_learning_setting['tb_log_name'] = "qlstm-tmazev0"
qlstm_learning_setting['tb_log_dir'] = "./logs/t_maze_tensorboard/"
qlstm_learning_setting['maze_length'] = maze_length
qlstm_learning_setting['total_timesteps'] = total_timesteps
qlstm_learning_setting['seed'] = None
qlstm_learning_setting['policy'] = QLSTMPolicy
qlstm_learning_setting['save'] = False
qlstm_learning_setting['device'] = 'cpu'

ppo_learning_setting = {}
ppo_learning_setting['envClass'] = envClass
ppo_learning_setting['learning_rate'] = 1e-2
ppo_learning_setting['discount_rate'] = 0.99
ppo_learning_setting['nn_layer_size'] = 8
ppo_learning_setting['n_steps'] = 256
ppo_learning_setting['memory_seq_length'] = 1
ppo_learning_setting['tb_log_name'] = "ppo-tmazev0"
ppo_learning_setting['tb_log_dir'] = "./logs/t_maze_tensorboard/"
ppo_learning_setting['maze_length'] = maze_length
ppo_learning_setting['total_timesteps'] = total_timesteps
ppo_learning_setting['seed'] = None
ppo_learning_setting['policy'] = MlpACPolicy
ppo_learning_setting['save'] = False
ppo_learning_setting['device'] = 'cpu'

ppoLSTM_learning_setting = {}
ppoLSTM_learning_setting['envClass'] = envClass
ppoLSTM_learning_setting['learning_rate'] = 1e-2
ppoLSTM_learning_setting['discount_rate'] = 0.99
ppoLSTM_learning_setting['nn_layer_size'] = 2
ppoLSTM_learning_setting['n_steps'] = 256
ppoLSTM_learning_setting['memory_seq_length'] = 1
ppoLSTM_learning_setting['tb_log_name'] = "ppoLSTM-tmazev0"
ppoLSTM_learning_setting['tb_log_dir'] = "./logs/t_maze_tensorboard/"
ppoLSTM_learning_setting['maze_length'] = maze_length
ppoLSTM_learning_setting['total_timesteps'] = total_timesteps
ppoLSTM_learning_setting['seed'] = None
ppoLSTM_learning_setting['policy'] = LSTMACPolicy
ppoLSTM_learning_setting['save'] = False
ppoLSTM_learning_setting['device'] = 'cpu'

a2c_learning_setting = {}
a2c_learning_setting['envClass'] = envClass
a2c_learning_setting['learning_rate'] = 1e-2
a2c_learning_setting['discount_rate'] = 0.99
a2c_learning_setting['nn_layer_size'] = 8
a2c_learning_setting['n_steps'] = 256
a2c_learning_setting['memory_seq_length'] = 1
a2c_learning_setting['tb_log_name'] = "a2c-tmazev0"
a2c_learning_setting['tb_log_dir'] = "./logs/t_maze_tensorboard/"
a2c_learning_setting['maze_length'] = maze_length
a2c_learning_setting['total_timesteps'] = total_timesteps
a2c_learning_setting['seed'] = None
a2c_learning_setting['policy'] = "MlpPolicy"
a2c_learning_setting['save'] = False
a2c_learning_setting['device'] = 'cpu'
