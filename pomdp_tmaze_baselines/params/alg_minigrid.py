from pomdp_tmaze_baselines.EnvMinigrid import MinigridEnv
from pomdp_tmaze_baselines.utils.UtilStableAgents import train_ppo_lstm_agent,\
    train_ppo_agent

from pomdp_tmaze_baselines.utils.UtilPolicies import MlpACPolicy, CNNACPolicy

total_timesteps = 1_000_000
maze_length = 10
envClass = MinigridEnv

number_of_parallel_experiments = 1
# 0: No memory, 1: Kk, 2: Bk, 3: Ok, 4:OAk

# Change the flags to True/False for only running specific agents
start_ae_ppo = True
start_cnn_ppo = True
start_ae_smm = True
start_cnn_smm = True
start_ae_smm_intr = True
start_ae_lstm = True
start_cnn_lstm = True


learning_rate = 1e-3
discount_rate = 0.99
nn_num_layers = 4
nn_layer_size = 4
n_steps = 256
batch_size = 256


mgrid_ae_ppo_setting = {}
mgrid_ae_ppo_setting['envClass'] = envClass
mgrid_ae_ppo_setting['learning_rate'] = learning_rate
mgrid_ae_ppo_setting['discount_rate'] = discount_rate
mgrid_ae_ppo_setting['nn_num_layers'] = nn_num_layers
mgrid_ae_ppo_setting['nn_layer_size'] = nn_layer_size
mgrid_ae_ppo_setting['n_steps'] = n_steps
mgrid_ae_ppo_setting['batch_size'] = batch_size
mgrid_ae_ppo_setting['memory_type'] = 0
mgrid_ae_ppo_setting['memory_length'] = 1
mgrid_ae_ppo_setting['intrinsic_enabled'] = False
mgrid_ae_ppo_setting['intrinsic_beta'] = 0.01
mgrid_ae_ppo_setting['ae_enabled'] = True
mgrid_ae_ppo_setting['ae_path'] = "models/ae.torch"
mgrid_ae_ppo_setting['ae_rcons_err_type'] = "MSE"
mgrid_ae_ppo_setting['tb_log_name'] = "ae_ppo"
mgrid_ae_ppo_setting['tb_log_dir'] = "./logs/c_minigrid_tb/"
mgrid_ae_ppo_setting['maze_length'] = maze_length
mgrid_ae_ppo_setting['total_timesteps'] = total_timesteps
mgrid_ae_ppo_setting['seed'] = None
mgrid_ae_ppo_setting['policy'] = MlpACPolicy
mgrid_ae_ppo_setting['save'] = True
mgrid_ae_ppo_setting['device'] = 'cuda:0'
mgrid_ae_ppo_setting['train_func'] = train_ppo_agent


mgrid_cnn_ppo_setting = {}
mgrid_cnn_ppo_setting['envClass'] = envClass
mgrid_cnn_ppo_setting['learning_rate'] = learning_rate
mgrid_cnn_ppo_setting['discount_rate'] = discount_rate
mgrid_cnn_ppo_setting['nn_num_layers'] = nn_num_layers
mgrid_cnn_ppo_setting['nn_layer_size'] = nn_layer_size
mgrid_cnn_ppo_setting['n_steps'] = n_steps
mgrid_cnn_ppo_setting['batch_size'] = batch_size
mgrid_cnn_ppo_setting['memory_type'] = 0
mgrid_cnn_ppo_setting['memory_length'] = 1
mgrid_cnn_ppo_setting['intrinsic_enabled'] = False
mgrid_cnn_ppo_setting['intrinsic_beta'] = 0.01
mgrid_cnn_ppo_setting['ae_enabled'] = False
mgrid_cnn_ppo_setting['ae_path'] = "models/ae.torch"
mgrid_cnn_ppo_setting['ae_rcons_err_type'] = "MSE"
mgrid_cnn_ppo_setting['tb_log_name'] = "cnn_ppo"
mgrid_cnn_ppo_setting['tb_log_dir'] = "./logs/c_minigrid_tb/"
mgrid_cnn_ppo_setting['maze_length'] = maze_length
mgrid_cnn_ppo_setting['total_timesteps'] = total_timesteps
mgrid_cnn_ppo_setting['seed'] = None
mgrid_cnn_ppo_setting['policy'] = CNNACPolicy
mgrid_cnn_ppo_setting['save'] = True
mgrid_cnn_ppo_setting['device'] = 'cuda:0'
mgrid_cnn_ppo_setting['train_func'] = train_ppo_agent


mgrid_ae_smm_setting = {}
mgrid_ae_smm_setting['envClass'] = envClass
mgrid_ae_smm_setting['learning_rate'] = learning_rate
mgrid_ae_smm_setting['discount_rate'] = discount_rate
mgrid_ae_smm_setting['nn_num_layers'] = nn_num_layers
mgrid_ae_smm_setting['nn_layer_size'] = nn_layer_size
mgrid_ae_smm_setting['n_steps'] = n_steps
mgrid_ae_smm_setting['batch_size'] = batch_size
mgrid_ae_smm_setting['memory_type'] = 3
mgrid_ae_smm_setting['memory_length'] = 1
mgrid_ae_smm_setting['intrinsic_enabled'] = False
mgrid_ae_smm_setting['intrinsic_beta'] = 0.01
mgrid_ae_smm_setting['ae_enabled'] = True
mgrid_ae_smm_setting['ae_path'] = "models/ae.torch"
mgrid_ae_smm_setting['ae_rcons_err_type'] = "MSE"
mgrid_ae_smm_setting['tb_log_name'] = "ae_smm"
mgrid_ae_smm_setting['tb_log_dir'] = "./logs/c_minigrid_tb/"
mgrid_ae_smm_setting['maze_length'] = maze_length
mgrid_ae_smm_setting['total_timesteps'] = total_timesteps
mgrid_ae_smm_setting['seed'] = None
mgrid_ae_smm_setting['policy'] = MlpACPolicy
mgrid_ae_smm_setting['save'] = True
mgrid_ae_smm_setting['device'] = 'cuda:0'
mgrid_ae_smm_setting['train_func'] = train_ppo_agent


mgrid_cnn_smm_setting = {}
mgrid_cnn_smm_setting['envClass'] = envClass
mgrid_cnn_smm_setting['learning_rate'] = learning_rate
mgrid_cnn_smm_setting['discount_rate'] = discount_rate
mgrid_cnn_smm_setting['nn_num_layers'] = nn_num_layers
mgrid_cnn_smm_setting['nn_layer_size'] = nn_layer_size
mgrid_cnn_smm_setting['n_steps'] = n_steps
mgrid_cnn_smm_setting['batch_size'] = batch_size
mgrid_cnn_smm_setting['memory_type'] = 3
mgrid_cnn_smm_setting['memory_length'] = 1
mgrid_cnn_smm_setting['intrinsic_enabled'] = False
mgrid_cnn_smm_setting['intrinsic_beta'] = 0.01
mgrid_cnn_smm_setting['ae_enabled'] = False
mgrid_cnn_smm_setting['ae_path'] = "models/ae.torch"
mgrid_cnn_smm_setting['ae_rcons_err_type'] = "MSE"
mgrid_cnn_smm_setting['tb_log_name'] = "cnn_smm"
mgrid_cnn_smm_setting['tb_log_dir'] = "./logs/c_minigrid_tb/"
mgrid_cnn_smm_setting['maze_length'] = maze_length
mgrid_cnn_smm_setting['total_timesteps'] = total_timesteps
mgrid_cnn_smm_setting['seed'] = None
mgrid_cnn_smm_setting['policy'] = CNNACPolicy
mgrid_cnn_smm_setting['save'] = True
mgrid_cnn_smm_setting['device'] = 'cuda:0'
mgrid_cnn_smm_setting['train_func'] = train_ppo_agent


mgrid_ae_smm_intr_setting = {}
mgrid_ae_smm_intr_setting['envClass'] = envClass
mgrid_ae_smm_intr_setting['learning_rate'] = learning_rate
mgrid_ae_smm_intr_setting['discount_rate'] = discount_rate
mgrid_ae_smm_intr_setting['nn_num_layers'] = nn_num_layers
mgrid_ae_smm_intr_setting['nn_layer_size'] = nn_layer_size
mgrid_ae_smm_intr_setting['n_steps'] = n_steps
mgrid_ae_smm_intr_setting['batch_size'] = batch_size
mgrid_ae_smm_intr_setting['memory_type'] = 3
mgrid_ae_smm_intr_setting['memory_length'] = 1
mgrid_ae_smm_intr_setting['intrinsic_enabled'] = True
mgrid_ae_smm_intr_setting['intrinsic_beta'] = 0.01
mgrid_ae_smm_intr_setting['ae_enabled'] = True
mgrid_ae_smm_intr_setting['ae_path'] = "models/ae.torch"
mgrid_ae_smm_intr_setting['ae_rcons_err_type'] = "MSE"
mgrid_ae_smm_intr_setting['tb_log_name'] = "ae_smm_intr"
mgrid_ae_smm_intr_setting['tb_log_dir'] = "./logs/c_minigrid_tb/"
mgrid_ae_smm_intr_setting['maze_length'] = maze_length
mgrid_ae_smm_intr_setting['total_timesteps'] = total_timesteps
mgrid_ae_smm_intr_setting['seed'] = None
mgrid_ae_smm_intr_setting['policy'] = MlpACPolicy
mgrid_ae_smm_intr_setting['save'] = True
mgrid_ae_smm_intr_setting['device'] = 'cuda:0'
mgrid_ae_smm_intr_setting['train_func'] = train_ppo_agent


mgrid_ae_lstm_setting = {}
mgrid_ae_lstm_setting['envClass'] = envClass
mgrid_ae_lstm_setting['learning_rate'] = learning_rate
mgrid_ae_lstm_setting['discount_rate'] = discount_rate
mgrid_ae_lstm_setting['nn_num_layers'] = nn_num_layers
mgrid_ae_lstm_setting['nn_layer_size'] = nn_layer_size
mgrid_ae_lstm_setting['n_steps'] = n_steps
mgrid_ae_lstm_setting['batch_size'] = batch_size
mgrid_ae_lstm_setting['memory_type'] = 0
mgrid_ae_lstm_setting['memory_length'] = 1
mgrid_ae_lstm_setting['intrinsic_enabled'] = False
mgrid_ae_lstm_setting['intrinsic_beta'] = 0.01
mgrid_ae_lstm_setting['ae_enabled'] = True
mgrid_ae_lstm_setting['ae_path'] = "models/ae.torch"
mgrid_ae_lstm_setting['ae_rcons_err_type'] = "MSE"
mgrid_ae_lstm_setting['tb_log_name'] = "ae_lstm"
mgrid_ae_lstm_setting['tb_log_dir'] = "./logs/c_minigrid_tb/"
mgrid_ae_lstm_setting['maze_length'] = maze_length
mgrid_ae_lstm_setting['total_timesteps'] = total_timesteps
mgrid_ae_lstm_setting['seed'] = None
mgrid_ae_lstm_setting['policy'] = "MlpLstmPolicy"
mgrid_ae_lstm_setting['save'] = True
mgrid_ae_lstm_setting['device'] = 'cuda:0'
mgrid_ae_lstm_setting['train_func'] = train_ppo_lstm_agent


mgrid_cnn_lstm_setting = {}
mgrid_cnn_lstm_setting['envClass'] = envClass
mgrid_cnn_lstm_setting['learning_rate'] = learning_rate
mgrid_cnn_lstm_setting['discount_rate'] = discount_rate
mgrid_cnn_lstm_setting['nn_num_layers'] = nn_num_layers
mgrid_cnn_lstm_setting['nn_layer_size'] = nn_layer_size
mgrid_cnn_lstm_setting['n_steps'] = n_steps
mgrid_cnn_lstm_setting['batch_size'] = batch_size
mgrid_cnn_lstm_setting['memory_type'] = 0
mgrid_cnn_lstm_setting['memory_length'] = 1
mgrid_cnn_lstm_setting['intrinsic_enabled'] = False
mgrid_cnn_lstm_setting['intrinsic_beta'] = 0.01
mgrid_cnn_lstm_setting['ae_enabled'] = False
mgrid_cnn_lstm_setting['ae_path'] = "models/ae.torch"
mgrid_cnn_lstm_setting['ae_rcons_err_type'] = "MSE"
mgrid_cnn_lstm_setting['tb_log_name'] = "cnn_lstm"
mgrid_cnn_lstm_setting['tb_log_dir'] = "./logs/c_minigrid_tb/"
mgrid_cnn_lstm_setting['maze_length'] = maze_length
mgrid_cnn_lstm_setting['total_timesteps'] = total_timesteps
mgrid_cnn_lstm_setting['seed'] = None
mgrid_cnn_lstm_setting['policy'] = "CnnLstmPolicy"
mgrid_cnn_lstm_setting['save'] = True
mgrid_cnn_lstm_setting['device'] = 'cuda:0'
mgrid_cnn_lstm_setting['train_func'] = train_ppo_lstm_agent
