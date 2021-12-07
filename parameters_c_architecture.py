from EnvTMaze import TMazeEnvV9
from UtilPolicies import MlpDQNPolicy
from UtilPolicies import QLSTMPolicy
from UtilStableAgents import train_dqn_agent

total_timesteps = 1_000_000
maze_length = 10
envClass = TMazeEnvV9

number_of_parallel_experiments = 1
# 0: No memory, 1: Kk, 2: Bk, 3: Ok, 4:OAk

# Change the flags to True/False for only running specific agents
start_no_memory = False
start_no_memory_intr = False
start_o_k_memory = True
start_o_k_intr_memory = True
start_lstm = True
start_lstm_intr = True

learning_rate = 1e-4
discount_rate = 0.99
epsilon_start = 0.1
epsilon_end = 0.1
exploration_fraction = 1.0
nn_num_layers = 4
nn_layer_size = 128

no_memory_learning_setting = {}
no_memory_learning_setting['envClass'] = envClass
no_memory_learning_setting['learning_rate'] = learning_rate
no_memory_learning_setting['discount_rate'] = discount_rate
no_memory_learning_setting['epsilon_start'] = epsilon_start
no_memory_learning_setting['epsilon_end'] = epsilon_end
no_memory_learning_setting['memory_type'] = 0
no_memory_learning_setting['memory_length'] = 1
no_memory_learning_setting['intrinsic_enabled'] = 0
no_memory_learning_setting['exploration_fraction'] = exploration_fraction
no_memory_learning_setting['update_interval'] = 128
no_memory_learning_setting['learning_starts'] = 512
no_memory_learning_setting['buffer_size'] = 100000
no_memory_learning_setting['nn_num_layers'] = nn_num_layers
no_memory_learning_setting['nn_layer_size'] = nn_layer_size
no_memory_learning_setting['tb_log_name'] = "no_memory"
no_memory_learning_setting['tb_log_dir'] = "./logs/c_architectures_tb/"
no_memory_learning_setting['maze_length'] = maze_length
no_memory_learning_setting['total_timesteps'] = total_timesteps
no_memory_learning_setting['seed'] = None
no_memory_learning_setting['policy'] = MlpDQNPolicy
no_memory_learning_setting['save'] = False
no_memory_learning_setting['device'] = 'cuda:0'
no_memory_learning_setting['train_func'] = train_dqn_agent

no_memory_intr_learning_setting = {}
no_memory_intr_learning_setting['envClass'] = envClass
no_memory_intr_learning_setting['learning_rate'] = learning_rate
no_memory_intr_learning_setting['discount_rate'] = discount_rate
no_memory_intr_learning_setting['epsilon_start'] = epsilon_start
no_memory_intr_learning_setting['epsilon_end'] = epsilon_end
no_memory_intr_learning_setting['memory_type'] = 0
no_memory_intr_learning_setting['memory_length'] = 1
no_memory_intr_learning_setting['intrinsic_enabled'] = 0
no_memory_intr_learning_setting['exploration_fraction'] = exploration_fraction
no_memory_intr_learning_setting['update_interval'] = 128
no_memory_intr_learning_setting['learning_starts'] = 512
no_memory_intr_learning_setting['buffer_size'] = 100000
no_memory_intr_learning_setting['nn_num_layers'] = nn_num_layers
no_memory_intr_learning_setting['nn_layer_size'] = nn_layer_size
no_memory_intr_learning_setting['tb_log_name'] = "no_memory_intr"
no_memory_intr_learning_setting['tb_log_dir'] = "./logs/c_architectures_tb/"
no_memory_intr_learning_setting['maze_length'] = maze_length
no_memory_intr_learning_setting['total_timesteps'] = total_timesteps
no_memory_intr_learning_setting['seed'] = None
no_memory_intr_learning_setting['policy'] = MlpDQNPolicy
no_memory_intr_learning_setting['save'] = False
no_memory_intr_learning_setting['device'] = 'cuda:0'
no_memory_intr_learning_setting['train_func'] = train_dqn_agent

o_k_memory_learning_setting = {}
o_k_memory_learning_setting['envClass'] = envClass
o_k_memory_learning_setting['learning_rate'] = learning_rate
o_k_memory_learning_setting['discount_rate'] = discount_rate
o_k_memory_learning_setting['epsilon_start'] = epsilon_start
o_k_memory_learning_setting['epsilon_end'] = epsilon_end
o_k_memory_learning_setting['memory_type'] = 3
o_k_memory_learning_setting['memory_length'] = 1
o_k_memory_learning_setting['intrinsic_enabled'] = 0
o_k_memory_learning_setting['exploration_fraction'] = exploration_fraction
o_k_memory_learning_setting['update_interval'] = 128
o_k_memory_learning_setting['learning_starts'] = 512
o_k_memory_learning_setting['buffer_size'] = 100000
o_k_memory_learning_setting['nn_num_layers'] = nn_num_layers
o_k_memory_learning_setting['nn_layer_size'] = nn_layer_size
o_k_memory_learning_setting['tb_log_name'] = "o_k"
o_k_memory_learning_setting['tb_log_dir'] = "./logs/c_architectures_tb/"
o_k_memory_learning_setting['maze_length'] = maze_length
o_k_memory_learning_setting['total_timesteps'] = total_timesteps
o_k_memory_learning_setting['seed'] = None
o_k_memory_learning_setting['policy'] = MlpDQNPolicy
o_k_memory_learning_setting['save'] = False
o_k_memory_learning_setting['device'] = 'cuda:0'
o_k_memory_learning_setting['train_func'] = train_dqn_agent


o_k_intr_memory_learning_setting = {}
o_k_intr_memory_learning_setting['envClass'] = envClass
o_k_intr_memory_learning_setting['learning_rate'] = learning_rate
o_k_intr_memory_learning_setting['discount_rate'] = discount_rate
o_k_intr_memory_learning_setting['epsilon_start'] = epsilon_start
o_k_intr_memory_learning_setting['epsilon_end'] = epsilon_end
o_k_intr_memory_learning_setting['memory_type'] = 3
o_k_intr_memory_learning_setting['memory_length'] = 1
o_k_intr_memory_learning_setting['intrinsic_enabled'] = 1
o_k_intr_memory_learning_setting['exploration_fraction'] = exploration_fraction
o_k_intr_memory_learning_setting['update_interval'] = 128
o_k_intr_memory_learning_setting['learning_starts'] = 512
o_k_intr_memory_learning_setting['buffer_size'] = 100000
o_k_intr_memory_learning_setting['nn_num_layers'] = nn_num_layers
o_k_intr_memory_learning_setting['nn_layer_size'] = nn_layer_size
o_k_intr_memory_learning_setting['tb_log_name'] = "o_k_intr"
o_k_intr_memory_learning_setting['tb_log_dir'] = "./logs/c_architectures_tb/"
o_k_intr_memory_learning_setting['maze_length'] = maze_length
o_k_intr_memory_learning_setting['total_timesteps'] = total_timesteps
o_k_intr_memory_learning_setting['seed'] = None
o_k_intr_memory_learning_setting['policy'] = MlpDQNPolicy
o_k_intr_memory_learning_setting['save'] = False
o_k_intr_memory_learning_setting['device'] = 'cuda:0'
o_k_intr_memory_learning_setting['train_func'] = train_dqn_agent


lstm_learning_setting = {}
lstm_learning_setting['envClass'] = envClass
lstm_learning_setting['learning_rate'] = learning_rate
lstm_learning_setting['discount_rate'] = discount_rate
lstm_learning_setting['epsilon_start'] = epsilon_start
lstm_learning_setting['epsilon_end'] = epsilon_end
lstm_learning_setting['memory_type'] = 0
lstm_learning_setting['memory_length'] = 1
lstm_learning_setting['intrinsic_enabled'] = 0
lstm_learning_setting['exploration_fraction'] = exploration_fraction
lstm_learning_setting['update_interval'] = 128
lstm_learning_setting['learning_starts'] = 512
lstm_learning_setting['buffer_size'] = 100000
lstm_learning_setting['nn_num_layers'] = nn_num_layers
lstm_learning_setting['nn_layer_size'] = nn_layer_size
lstm_learning_setting['tb_log_name'] = "lstm"
lstm_learning_setting['tb_log_dir'] = "./logs/c_architectures_tb/"
lstm_learning_setting['maze_length'] = maze_length
lstm_learning_setting['total_timesteps'] = total_timesteps
lstm_learning_setting['seed'] = None
lstm_learning_setting['policy'] = QLSTMPolicy
lstm_learning_setting['save'] = False
lstm_learning_setting['device'] = 'cuda:0'
lstm_learning_setting['train_func'] = train_dqn_agent

lstm_intr_learning_setting = {}
lstm_intr_learning_setting['envClass'] = envClass
lstm_intr_learning_setting['learning_rate'] = learning_rate
lstm_intr_learning_setting['discount_rate'] = discount_rate
lstm_intr_learning_setting['epsilon_start'] = epsilon_start
lstm_intr_learning_setting['epsilon_end'] = epsilon_end
lstm_intr_learning_setting['memory_type'] = 0
lstm_intr_learning_setting['memory_length'] = 1
lstm_intr_learning_setting['intrinsic_enabled'] = 0
lstm_intr_learning_setting['exploration_fraction'] = exploration_fraction
lstm_intr_learning_setting['update_interval'] = 128
lstm_intr_learning_setting['learning_starts'] = 512
lstm_intr_learning_setting['buffer_size'] = 100000
lstm_intr_learning_setting['nn_num_layers'] = nn_num_layers
lstm_intr_learning_setting['nn_layer_size'] = nn_layer_size
lstm_intr_learning_setting['tb_log_name'] = "lstm_intr"
lstm_intr_learning_setting['tb_log_dir'] = "./logs/c_architectures_tb/"
lstm_intr_learning_setting['maze_length'] = maze_length
lstm_intr_learning_setting['total_timesteps'] = total_timesteps
lstm_intr_learning_setting['seed'] = None
lstm_intr_learning_setting['policy'] = QLSTMPolicy
lstm_intr_learning_setting['save'] = False
lstm_intr_learning_setting['device'] = 'cuda:0'
lstm_intr_learning_setting['train_func'] = train_dqn_agent
