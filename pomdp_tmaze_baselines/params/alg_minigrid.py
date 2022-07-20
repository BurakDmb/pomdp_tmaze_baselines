from pomdp_tmaze_baselines.EnvMinigrid import MinigridEnv
from pomdp_tmaze_baselines.utils.UtilStableAgents import train_ppo_lstm_agent,\
    train_ppo_agent

from pomdp_tmaze_baselines.utils.UtilPolicies import MlpACPolicy

total_timesteps = 1_000_000
maze_length = 10
envClass = MinigridEnv

number_of_parallel_experiments = 1
# 0: No memory, 1: Kk, 2: Bk, 3: Ok, 4:OAk

# Change the flags to True/False for only running specific agents
start_ppo = True
start_smm = False
start_smm_intr = False
start_lstm = False

learning_rate = 1e-3
discount_rate = 0.99
nn_num_layers = 4
nn_layer_size = 4
n_steps = 1024
batch_size = 1024


mgrid_ppo_learning_setting = {}
mgrid_ppo_learning_setting['envClass'] = envClass
mgrid_ppo_learning_setting['learning_rate'] = learning_rate
mgrid_ppo_learning_setting['discount_rate'] = discount_rate
mgrid_ppo_learning_setting['nn_num_layers'] = nn_num_layers
mgrid_ppo_learning_setting['nn_layer_size'] = nn_layer_size
mgrid_ppo_learning_setting['n_steps'] = n_steps
mgrid_ppo_learning_setting['batch_size'] = batch_size
mgrid_ppo_learning_setting['memory_type'] = 0
mgrid_ppo_learning_setting['memory_length'] = 1
mgrid_ppo_learning_setting['intrinsic_enabled'] = False
mgrid_ppo_learning_setting['intrinsic_beta'] = 0.1
mgrid_ppo_learning_setting['ae_enabled'] = True
mgrid_ppo_learning_setting['ae_path'] = "models/ae.torch"
mgrid_ppo_learning_setting['ae_rcons_err_type'] = "MSE"
mgrid_ppo_learning_setting['tb_log_name'] = "o_k"
mgrid_ppo_learning_setting['tb_log_dir'] = None
mgrid_ppo_learning_setting['maze_length'] = 10
mgrid_ppo_learning_setting['total_timesteps'] = 50
mgrid_ppo_learning_setting['seed'] = None
mgrid_ppo_learning_setting['policy'] = MlpACPolicy
mgrid_ppo_learning_setting['save'] = False
mgrid_ppo_learning_setting['device'] = 'cuda:0'
mgrid_ppo_learning_setting['train_func'] = train_ppo_agent


mgrid_smm_learning_setting = {}
mgrid_smm_learning_setting['envClass'] = envClass
mgrid_smm_learning_setting['learning_rate'] = learning_rate
mgrid_smm_learning_setting['discount_rate'] = discount_rate
mgrid_smm_learning_setting['nn_num_layers'] = nn_num_layers
mgrid_smm_learning_setting['nn_layer_size'] = nn_layer_size
mgrid_smm_learning_setting['n_steps'] = n_steps
mgrid_smm_learning_setting['batch_size'] = batch_size
mgrid_smm_learning_setting['memory_type'] = 3
mgrid_smm_learning_setting['memory_length'] = 1
mgrid_smm_learning_setting['intrinsic_enabled'] = False
mgrid_smm_learning_setting['intrinsic_beta'] = 0.1
mgrid_smm_learning_setting['ae_enabled'] = True
mgrid_smm_learning_setting['ae_path'] = "models/ae.torch"
mgrid_smm_learning_setting['ae_rcons_err_type'] = "MSE"
mgrid_smm_learning_setting['tb_log_name'] = "o_k"
mgrid_smm_learning_setting['tb_log_dir'] = None
mgrid_smm_learning_setting['maze_length'] = 10
mgrid_smm_learning_setting['total_timesteps'] = 50
mgrid_smm_learning_setting['seed'] = None
mgrid_smm_learning_setting['policy'] = MlpACPolicy
mgrid_smm_learning_setting['save'] = False
mgrid_smm_learning_setting['device'] = 'cuda:0'
mgrid_smm_learning_setting['train_func'] = train_ppo_agent


mgrid_smm_intr_learning_setting = {}
mgrid_smm_intr_learning_setting['envClass'] = envClass
mgrid_smm_intr_learning_setting['learning_rate'] = learning_rate
mgrid_smm_intr_learning_setting['discount_rate'] = discount_rate
mgrid_smm_intr_learning_setting['nn_num_layers'] = nn_num_layers
mgrid_smm_intr_learning_setting['nn_layer_size'] = nn_layer_size
mgrid_smm_intr_learning_setting['n_steps'] = n_steps
mgrid_smm_intr_learning_setting['batch_size'] = batch_size
mgrid_smm_intr_learning_setting['memory_type'] = 3
mgrid_smm_intr_learning_setting['memory_length'] = 1
mgrid_smm_intr_learning_setting['intrinsic_enabled'] = True
mgrid_smm_intr_learning_setting['intrinsic_beta'] = 0.1
mgrid_smm_intr_learning_setting['ae_enabled'] = True
mgrid_smm_intr_learning_setting['ae_path'] = "models/ae.torch"
mgrid_smm_intr_learning_setting['ae_rcons_err_type'] = "MSE"
mgrid_smm_intr_learning_setting['tb_log_name'] = "o_k"
mgrid_smm_intr_learning_setting['tb_log_dir'] = None
mgrid_smm_intr_learning_setting['maze_length'] = 10
mgrid_smm_intr_learning_setting['total_timesteps'] = 50
mgrid_smm_intr_learning_setting['seed'] = None
mgrid_smm_intr_learning_setting['policy'] = MlpACPolicy
mgrid_smm_intr_learning_setting['save'] = False
mgrid_smm_intr_learning_setting['device'] = 'cuda:0'
mgrid_smm_intr_learning_setting['train_func'] = train_ppo_agent


mgrid_lstm_learning_setting = {}
mgrid_lstm_learning_setting['envClass'] = envClass
mgrid_lstm_learning_setting['learning_rate'] = learning_rate
mgrid_lstm_learning_setting['discount_rate'] = discount_rate
mgrid_lstm_learning_setting['nn_num_layers'] = nn_num_layers
mgrid_lstm_learning_setting['nn_layer_size'] = nn_layer_size
mgrid_lstm_learning_setting['n_steps'] = n_steps
mgrid_lstm_learning_setting['batch_size'] = batch_size
mgrid_lstm_learning_setting['memory_type'] = 0
mgrid_lstm_learning_setting['memory_length'] = 1
mgrid_lstm_learning_setting['intrinsic_enabled'] = False
mgrid_lstm_learning_setting['intrinsic_beta'] = 0.1
mgrid_lstm_learning_setting['ae_enabled'] = True
mgrid_lstm_learning_setting['ae_path'] = "models/ae.torch"
mgrid_lstm_learning_setting['ae_rcons_err_type'] = "MSE"
mgrid_lstm_learning_setting['tb_log_name'] = "o_k"
mgrid_lstm_learning_setting['tb_log_dir'] = None
mgrid_lstm_learning_setting['maze_length'] = maze_length
mgrid_lstm_learning_setting['total_timesteps'] = total_timesteps
mgrid_lstm_learning_setting['seed'] = None
mgrid_lstm_learning_setting['policy'] = "MlpLstmPolicy"
mgrid_lstm_learning_setting['save'] = False
mgrid_lstm_learning_setting['device'] = 'cuda:0'
mgrid_lstm_learning_setting['train_func'] = train_ppo_lstm_agent
