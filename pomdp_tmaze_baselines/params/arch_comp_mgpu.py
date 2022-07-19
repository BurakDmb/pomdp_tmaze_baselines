from pomdp_tmaze_baselines.EnvTMaze import TMazeEnvMemoryWrapped
from pomdp_tmaze_baselines.utils.UtilPolicies import MlpACPolicy
from pomdp_tmaze_baselines.utils.UtilStableAgents import train_ppo_lstm_agent,\
    train_ppo_agent

total_timesteps = 1_000_000
maze_length = 10
envClass = TMazeEnvMemoryWrapped

number_of_parallel_experiments = 8
# 0: No memory, 1: Kk, 2: Bk, 3: Ok, 4:OAk

# Change the flags to True/False for only running specific agents
start_no_memory = True
start_no_memory_intr = False
start_o_k_memory = True
start_o_k_intr_memory = False
start_oa_k_memory = True
start_oa_k_intr_memory = False
start_lstm = True
start_lstm_intr = False

learning_rate = 1e-3
discount_rate = 0.99
nn_num_layers = 3
nn_layer_size = 16
n_steps = 2048
batch_size = 64

no_memory_learning_setting = {}
no_memory_learning_setting['envClass'] = envClass
no_memory_learning_setting['learning_rate'] = learning_rate
no_memory_learning_setting['discount_rate'] = discount_rate
no_memory_learning_setting['nn_num_layers'] = nn_num_layers
no_memory_learning_setting['nn_layer_size'] = nn_layer_size
no_memory_learning_setting['n_steps'] = n_steps
no_memory_learning_setting['batch_size'] = batch_size
no_memory_learning_setting['memory_type'] = 0
no_memory_learning_setting['memory_length'] = 1
no_memory_learning_setting['intrinsic_enabled'] = False
no_memory_learning_setting['intrinsic_beta'] = 0.1
no_memory_learning_setting['tb_log_name'] = "no_memory"
no_memory_learning_setting['tb_log_dir'] = "./logs/c_architectures_tb/"
no_memory_learning_setting['maze_length'] = maze_length
no_memory_learning_setting['total_timesteps'] = total_timesteps
no_memory_learning_setting['seed'] = None
no_memory_learning_setting['policy'] = MlpACPolicy
no_memory_learning_setting['save'] = False
no_memory_learning_setting['device'] = 'cuda:3'
no_memory_learning_setting['train_func'] = train_ppo_agent

no_memory_intr_learning_setting = {}
no_memory_intr_learning_setting['envClass'] = envClass
no_memory_intr_learning_setting['learning_rate'] = learning_rate
no_memory_intr_learning_setting['discount_rate'] = discount_rate
no_memory_intr_learning_setting['nn_num_layers'] = nn_num_layers
no_memory_intr_learning_setting['nn_layer_size'] = nn_layer_size
no_memory_intr_learning_setting['n_steps'] = n_steps
no_memory_intr_learning_setting['batch_size'] = batch_size
no_memory_intr_learning_setting['memory_type'] = 0
no_memory_intr_learning_setting['memory_length'] = 1
no_memory_intr_learning_setting['intrinsic_enabled'] = True
no_memory_intr_learning_setting['intrinsic_beta'] = 0.1
no_memory_intr_learning_setting['tb_log_name'] = "no_memory_intr"
no_memory_intr_learning_setting['tb_log_dir'] = "./logs/c_architectures_tb/"
no_memory_intr_learning_setting['maze_length'] = maze_length
no_memory_intr_learning_setting['total_timesteps'] = total_timesteps
no_memory_intr_learning_setting['seed'] = None
no_memory_intr_learning_setting['policy'] = MlpACPolicy
no_memory_intr_learning_setting['save'] = False
no_memory_intr_learning_setting['device'] = 'cuda:3'
no_memory_intr_learning_setting['train_func'] = train_ppo_agent

o_k_memory_learning_setting = {}
o_k_memory_learning_setting['envClass'] = envClass
o_k_memory_learning_setting['learning_rate'] = learning_rate
o_k_memory_learning_setting['discount_rate'] = discount_rate
o_k_memory_learning_setting['nn_num_layers'] = nn_num_layers
o_k_memory_learning_setting['nn_layer_size'] = nn_layer_size
o_k_memory_learning_setting['n_steps'] = n_steps
o_k_memory_learning_setting['batch_size'] = batch_size
o_k_memory_learning_setting['memory_type'] = 3
o_k_memory_learning_setting['memory_length'] = 1
o_k_memory_learning_setting['intrinsic_enabled'] = False
o_k_memory_learning_setting['intrinsic_beta'] = 0.1
o_k_memory_learning_setting['tb_log_name'] = "o_k"
o_k_memory_learning_setting['tb_log_dir'] = "./logs/c_architectures_tb/"
o_k_memory_learning_setting['maze_length'] = maze_length
o_k_memory_learning_setting['total_timesteps'] = total_timesteps
o_k_memory_learning_setting['seed'] = None
o_k_memory_learning_setting['policy'] = MlpACPolicy
o_k_memory_learning_setting['save'] = False
o_k_memory_learning_setting['device'] = 'cuda:2'
o_k_memory_learning_setting['train_func'] = train_ppo_agent

o_k_intr_memory_learning_setting = {}
o_k_intr_memory_learning_setting['envClass'] = envClass
o_k_intr_memory_learning_setting['learning_rate'] = learning_rate
o_k_intr_memory_learning_setting['discount_rate'] = discount_rate
o_k_intr_memory_learning_setting['nn_num_layers'] = nn_num_layers
o_k_intr_memory_learning_setting['nn_layer_size'] = nn_layer_size
o_k_intr_memory_learning_setting['n_steps'] = n_steps
o_k_intr_memory_learning_setting['batch_size'] = batch_size
o_k_intr_memory_learning_setting['memory_type'] = 3
o_k_intr_memory_learning_setting['memory_length'] = 1
o_k_intr_memory_learning_setting['intrinsic_enabled'] = True
o_k_intr_memory_learning_setting['intrinsic_beta'] = 0.1
o_k_intr_memory_learning_setting['tb_log_name'] = "o_k_intr"
o_k_intr_memory_learning_setting['tb_log_dir'] = "./logs/c_architectures_tb/"
o_k_intr_memory_learning_setting['maze_length'] = maze_length
o_k_intr_memory_learning_setting['total_timesteps'] = total_timesteps
o_k_intr_memory_learning_setting['seed'] = None
o_k_intr_memory_learning_setting['policy'] = MlpACPolicy
o_k_intr_memory_learning_setting['save'] = False
o_k_intr_memory_learning_setting['device'] = 'cuda:2'
o_k_intr_memory_learning_setting['train_func'] = train_ppo_agent


oa_k_memory_learning_setting = {}
oa_k_memory_learning_setting['envClass'] = envClass
oa_k_memory_learning_setting['learning_rate'] = learning_rate
oa_k_memory_learning_setting['discount_rate'] = discount_rate
oa_k_memory_learning_setting['nn_num_layers'] = nn_num_layers
oa_k_memory_learning_setting['nn_layer_size'] = nn_layer_size
oa_k_memory_learning_setting['n_steps'] = n_steps
oa_k_memory_learning_setting['batch_size'] = batch_size
oa_k_memory_learning_setting['memory_type'] = 4
oa_k_memory_learning_setting['memory_length'] = 1
oa_k_memory_learning_setting['intrinsic_enabled'] = False
oa_k_memory_learning_setting['intrinsic_beta'] = 0.1
oa_k_memory_learning_setting['tb_log_name'] = "oa_k"
oa_k_memory_learning_setting['tb_log_dir'] = "./logs/c_architectures_tb/"
oa_k_memory_learning_setting['maze_length'] = maze_length
oa_k_memory_learning_setting['total_timesteps'] = total_timesteps
oa_k_memory_learning_setting['seed'] = None
oa_k_memory_learning_setting['policy'] = MlpACPolicy
oa_k_memory_learning_setting['save'] = False
oa_k_memory_learning_setting['device'] = 'cuda:1'
oa_k_memory_learning_setting['train_func'] = train_ppo_agent

oa_k_intr_memory_learning_setting = {}
oa_k_intr_memory_learning_setting['envClass'] = envClass
oa_k_intr_memory_learning_setting['learning_rate'] = learning_rate
oa_k_intr_memory_learning_setting['discount_rate'] = discount_rate
oa_k_intr_memory_learning_setting['nn_num_layers'] = nn_num_layers
oa_k_intr_memory_learning_setting['nn_layer_size'] = nn_layer_size
oa_k_intr_memory_learning_setting['n_steps'] = n_steps
oa_k_intr_memory_learning_setting['batch_size'] = batch_size
oa_k_intr_memory_learning_setting['memory_type'] = 4
oa_k_intr_memory_learning_setting['memory_length'] = 1
oa_k_intr_memory_learning_setting['intrinsic_enabled'] = True
oa_k_intr_memory_learning_setting['intrinsic_beta'] = 0.1
oa_k_intr_memory_learning_setting['tb_log_name'] = "oa_k_intr"
oa_k_intr_memory_learning_setting['tb_log_dir'] = "./logs/c_architectures_tb/"
oa_k_intr_memory_learning_setting['maze_length'] = maze_length
oa_k_intr_memory_learning_setting['total_timesteps'] = total_timesteps
oa_k_intr_memory_learning_setting['seed'] = None
oa_k_intr_memory_learning_setting['policy'] = MlpACPolicy
oa_k_intr_memory_learning_setting['save'] = False
oa_k_intr_memory_learning_setting['device'] = 'cuda:1'
oa_k_intr_memory_learning_setting['train_func'] = train_ppo_agent

lstm_learning_setting = {}
lstm_learning_setting['envClass'] = envClass
lstm_learning_setting['learning_rate'] = learning_rate
lstm_learning_setting['discount_rate'] = discount_rate
lstm_learning_setting['nn_num_layers'] = nn_num_layers
lstm_learning_setting['nn_layer_size'] = nn_layer_size
lstm_learning_setting['n_steps'] = n_steps
lstm_learning_setting['batch_size'] = batch_size
lstm_learning_setting['memory_type'] = 0
lstm_learning_setting['memory_length'] = 1
lstm_learning_setting['intrinsic_enabled'] = False
lstm_learning_setting['intrinsic_beta'] = 0.1
lstm_learning_setting['tb_log_name'] = "lstm"
lstm_learning_setting['tb_log_dir'] = "./logs/c_architectures_tb/"
lstm_learning_setting['maze_length'] = maze_length
lstm_learning_setting['total_timesteps'] = total_timesteps
lstm_learning_setting['seed'] = None
lstm_learning_setting['policy'] = "MlpLstmPolicy"
lstm_learning_setting['save'] = False
lstm_learning_setting['device'] = 'cuda:0'
lstm_learning_setting['train_func'] = train_ppo_lstm_agent

lstm_intr_learning_setting = {}
lstm_intr_learning_setting['envClass'] = envClass
lstm_intr_learning_setting['learning_rate'] = learning_rate
lstm_intr_learning_setting['discount_rate'] = discount_rate
lstm_intr_learning_setting['nn_num_layers'] = nn_num_layers
lstm_intr_learning_setting['nn_layer_size'] = nn_layer_size
lstm_intr_learning_setting['n_steps'] = n_steps
lstm_intr_learning_setting['batch_size'] = batch_size
lstm_intr_learning_setting['memory_type'] = 0
lstm_intr_learning_setting['memory_length'] = 1
lstm_intr_learning_setting['intrinsic_enabled'] = True
lstm_intr_learning_setting['intrinsic_beta'] = 0.1
lstm_intr_learning_setting['tb_log_name'] = "lstm_intr"
lstm_intr_learning_setting['tb_log_dir'] = "./logs/c_architectures_tb/"
lstm_intr_learning_setting['maze_length'] = maze_length
lstm_intr_learning_setting['total_timesteps'] = total_timesteps
lstm_intr_learning_setting['seed'] = None
lstm_intr_learning_setting['policy'] = "MlpLstmPolicy"
lstm_intr_learning_setting['save'] = False
lstm_intr_learning_setting['device'] = 'cuda:0'
lstm_intr_learning_setting['train_func'] = train_ppo_lstm_agent
