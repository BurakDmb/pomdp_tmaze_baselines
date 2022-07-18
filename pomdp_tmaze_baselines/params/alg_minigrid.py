from pomdp_tmaze_baselines.EnvMinigrid import MinigridEnv
from pomdp_tmaze_baselines.utils.UtilStableAgents import\
    train_ppo_agent
from pomdp_tmaze_baselines.utils.UtilPolicies import MlpACPolicy

total_timesteps = 1_000_000
maze_length = 10
envClass = MinigridEnv

number_of_parallel_experiments = 1
# 0: No memory, 1: Kk, 2: Bk, 3: Ok, 4:OAk

# Change the flags to True/False for only running specific agents
start_no_memory = True
start_no_memory_intr = False
start_o_k_memory = True
start_o_k_intr_memory = True
start_oa_k_memory = True
start_oa_k_intr_memory = True
start_lstm = True
start_lstm_intr = False

learning_rate = 1e-3
discount_rate = 0.99
nn_num_layers = 4
nn_layer_size = 4
n_steps = 1024
batch_size = 1024


minigrid_learning_setting = {}
minigrid_learning_setting['envClass'] = envClass
minigrid_learning_setting['learning_rate'] = learning_rate
minigrid_learning_setting['discount_rate'] = discount_rate
minigrid_learning_setting['nn_num_layers'] = nn_num_layers
minigrid_learning_setting['nn_layer_size'] = nn_layer_size
minigrid_learning_setting['n_steps'] = n_steps
minigrid_learning_setting['batch_size'] = batch_size
minigrid_learning_setting['memory_type'] = 3
minigrid_learning_setting['memory_length'] = 1
minigrid_learning_setting['intrinsic_enabled'] = 0
minigrid_learning_setting['intrinsic_beta'] = 0.1
minigrid_learning_setting['ae_enabled'] = True
minigrid_learning_setting['ae_path'] = "models/ae.torch"
minigrid_learning_setting['ae_rcons_err_type'] = "MSE"
minigrid_learning_setting['tb_log_name'] = "o_k"
minigrid_learning_setting['tb_log_dir'] = None
minigrid_learning_setting['maze_length'] = 10
minigrid_learning_setting['total_timesteps'] = 50
minigrid_learning_setting['seed'] = None
minigrid_learning_setting['policy'] = MlpACPolicy
minigrid_learning_setting['save'] = False
minigrid_learning_setting['device'] = 'cpu'
minigrid_learning_setting['train_func'] = train_ppo_agent
