from pomdp_tmaze_baselines.EnvMinigrid import MinigridEnv
from pomdp_tmaze_baselines.utils.UtilStableAgents import train_ppo_lstm_agent,\
    train_ppo_agent

from pomdp_tmaze_baselines.utils.UtilPolicies import MlpACPolicy, CNNACPolicy
# from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import SubprocVecEnv

from pomdp_tmaze_baselines.utils.ConsumerAE import generateCommDict
from pomdp_tmaze_baselines.utils.ConsumerAE import ae_consumer
from multiprocessing import Process

total_timesteps = 1_000_000
maze_length = 10  # Not used in minigrid.
envClass = MinigridEnv

number_of_parallel_experiments = 1
# 0: No memory, 1: Kk, 2: Bk, 3: Ok, 4:OAk

# Change the flags to True/False for only running specific agents
start_ae_no_mem = True
start_ae_smm_lastk = True
start_ae_smm_bk = True
start_ae_smm_ok = True
start_ae_smm_ok_intr = True
start_ae_smm_oak = True
start_ae_smm_oak_intr = True
start_ae_lstm = True
start_cnn_no_mem = False

experiment_count = (
    int(start_ae_no_mem) +
    int(start_ae_smm_lastk) +
    int(start_ae_smm_bk) +
    int(start_ae_smm_ok) +
    int(start_ae_smm_ok_intr) +
    int(start_ae_smm_oak) +
    int(start_ae_smm_oak_intr) +
    int(start_ae_lstm) +
    int(start_cnn_no_mem)
)

# Dict keys are equal to tb_log_dir
start_dict = {
    'ae_no_mem': start_ae_no_mem,
    'ae_smm_lastk': start_ae_smm_lastk,
    'ae_smm_bk': start_ae_smm_bk,
    'ae_smm_ok': start_ae_smm_ok,
    'ae_smm_ok_intr': start_ae_smm_ok_intr,
    'ae_smm_oak': start_ae_smm_oak,
    'ae_smm_oak_intr': start_ae_smm_oak_intr,
    'ae_lstm': start_ae_lstm,
    'cnn_no_mem': start_cnn_no_mem,
}

learning_rate = 1e-4
discount_rate = 0.99
nn_num_layers = 2
nn_layer_size = 16
env_n_proc = 32 // (
    experiment_count*number_of_parallel_experiments
    ) if experiment_count > 0 else 1
vec_env_cls = SubprocVecEnv

n_steps = 1024
batch_size = n_steps  # Full batch iteration

ae_path = "models/ae.torch"
comm_dict = generateCommDict(start_dict, env_n_proc)
ae_consumer_process = Process(
    target=ae_consumer, args=(comm_dict, ae_path, "cuda:0"))
ae_consumer_process.start()


mgrid_ae_no_mem_setting = {}
mgrid_ae_no_mem_setting['envClass'] = envClass
mgrid_ae_no_mem_setting['learning_rate'] = learning_rate
mgrid_ae_no_mem_setting['discount_rate'] = discount_rate
mgrid_ae_no_mem_setting['nn_num_layers'] = nn_num_layers
mgrid_ae_no_mem_setting['nn_layer_size'] = nn_layer_size
mgrid_ae_no_mem_setting['n_steps'] = n_steps
mgrid_ae_no_mem_setting['batch_size'] = batch_size
mgrid_ae_no_mem_setting['memory_type'] = 0
mgrid_ae_no_mem_setting['memory_length'] = 1
mgrid_ae_no_mem_setting['intrinsic_enabled'] = False
mgrid_ae_no_mem_setting['intrinsic_beta'] = 0.01
mgrid_ae_no_mem_setting['ae_enabled'] = True
mgrid_ae_no_mem_setting['ae_comm_list'] = comm_dict['ae_no_mem']
mgrid_ae_no_mem_setting['ae_path'] = "models/ae.torch"
mgrid_ae_no_mem_setting['ae_rcons_err_type'] = "MSE"
mgrid_ae_no_mem_setting['eval_enabled'] = True
mgrid_ae_no_mem_setting['eval_freq'] = total_timesteps//200
mgrid_ae_no_mem_setting['eval_path'] = None
mgrid_ae_no_mem_setting['eval_episodes'] = 5
mgrid_ae_no_mem_setting['env_n_proc'] = env_n_proc
mgrid_ae_no_mem_setting['vec_env_cls'] = vec_env_cls
mgrid_ae_no_mem_setting['tb_log_name'] = "ae_no_mem"
mgrid_ae_no_mem_setting['tb_log_dir'] = "./logs/c_minigrid_tb/"
mgrid_ae_no_mem_setting['maze_length'] = maze_length
mgrid_ae_no_mem_setting['total_timesteps'] = total_timesteps
mgrid_ae_no_mem_setting['seed'] = None
mgrid_ae_no_mem_setting['policy'] = MlpACPolicy
mgrid_ae_no_mem_setting['save'] = True
mgrid_ae_no_mem_setting['device'] = 'cuda:0'
mgrid_ae_no_mem_setting['train_func'] = train_ppo_agent


mgrid_ae_smm_lastk_setting = {}
mgrid_ae_smm_lastk_setting['envClass'] = envClass
mgrid_ae_smm_lastk_setting['learning_rate'] = learning_rate
mgrid_ae_smm_lastk_setting['discount_rate'] = discount_rate
mgrid_ae_smm_lastk_setting['nn_num_layers'] = nn_num_layers
mgrid_ae_smm_lastk_setting['nn_layer_size'] = nn_layer_size
mgrid_ae_smm_lastk_setting['n_steps'] = n_steps
mgrid_ae_smm_lastk_setting['batch_size'] = batch_size
mgrid_ae_smm_lastk_setting['memory_type'] = 1
mgrid_ae_smm_lastk_setting['memory_length'] = 1
mgrid_ae_smm_lastk_setting['intrinsic_enabled'] = False
mgrid_ae_smm_lastk_setting['intrinsic_beta'] = 0.01
mgrid_ae_smm_lastk_setting['ae_enabled'] = True
mgrid_ae_smm_lastk_setting['ae_comm_list'] = comm_dict['ae_smm_lastk']
mgrid_ae_smm_lastk_setting['ae_path'] = "models/ae.torch"
mgrid_ae_smm_lastk_setting['ae_rcons_err_type'] = "MSE"
mgrid_ae_smm_lastk_setting['eval_enabled'] = True
mgrid_ae_smm_lastk_setting['eval_freq'] = total_timesteps//200
mgrid_ae_smm_lastk_setting['eval_episodes'] = 5
mgrid_ae_smm_lastk_setting['eval_path'] = None
mgrid_ae_smm_lastk_setting['env_n_proc'] = env_n_proc
mgrid_ae_smm_lastk_setting['vec_env_cls'] = vec_env_cls
mgrid_ae_smm_lastk_setting['tb_log_name'] = "ae_smm_lastk"
mgrid_ae_smm_lastk_setting['tb_log_dir'] = "./logs/c_minigrid_tb/"
mgrid_ae_smm_lastk_setting['maze_length'] = maze_length
mgrid_ae_smm_lastk_setting['total_timesteps'] = total_timesteps
mgrid_ae_smm_lastk_setting['seed'] = None
mgrid_ae_smm_lastk_setting['policy'] = MlpACPolicy
mgrid_ae_smm_lastk_setting['save'] = True
mgrid_ae_smm_lastk_setting['device'] = 'cuda:0'
mgrid_ae_smm_lastk_setting['train_func'] = train_ppo_agent


mgrid_ae_smm_bk_setting = {}
mgrid_ae_smm_bk_setting['envClass'] = envClass
mgrid_ae_smm_bk_setting['learning_rate'] = learning_rate
mgrid_ae_smm_bk_setting['discount_rate'] = discount_rate
mgrid_ae_smm_bk_setting['nn_num_layers'] = nn_num_layers
mgrid_ae_smm_bk_setting['nn_layer_size'] = nn_layer_size
mgrid_ae_smm_bk_setting['n_steps'] = n_steps
mgrid_ae_smm_bk_setting['batch_size'] = batch_size
mgrid_ae_smm_bk_setting['memory_type'] = 2
mgrid_ae_smm_bk_setting['memory_length'] = 1
mgrid_ae_smm_bk_setting['intrinsic_enabled'] = False
mgrid_ae_smm_bk_setting['intrinsic_beta'] = 0.01
mgrid_ae_smm_bk_setting['ae_enabled'] = True
mgrid_ae_smm_bk_setting['ae_comm_list'] = comm_dict['ae_smm_bk']
mgrid_ae_smm_bk_setting['ae_path'] = "models/ae.torch"
mgrid_ae_smm_bk_setting['ae_rcons_err_type'] = "MSE"
mgrid_ae_smm_bk_setting['eval_enabled'] = True
mgrid_ae_smm_bk_setting['eval_freq'] = total_timesteps//200
mgrid_ae_smm_bk_setting['eval_episodes'] = 5
mgrid_ae_smm_bk_setting['eval_path'] = None
mgrid_ae_smm_bk_setting['env_n_proc'] = env_n_proc
mgrid_ae_smm_bk_setting['vec_env_cls'] = vec_env_cls
mgrid_ae_smm_bk_setting['tb_log_name'] = "ae_smm_bk"
mgrid_ae_smm_bk_setting['tb_log_dir'] = "./logs/c_minigrid_tb/"
mgrid_ae_smm_bk_setting['maze_length'] = maze_length
mgrid_ae_smm_bk_setting['total_timesteps'] = total_timesteps
mgrid_ae_smm_bk_setting['seed'] = None
mgrid_ae_smm_bk_setting['policy'] = MlpACPolicy
mgrid_ae_smm_bk_setting['save'] = True
mgrid_ae_smm_bk_setting['device'] = 'cuda:0'
mgrid_ae_smm_bk_setting['train_func'] = train_ppo_agent


mgrid_ae_smm_ok_setting = {}
mgrid_ae_smm_ok_setting['envClass'] = envClass
mgrid_ae_smm_ok_setting['learning_rate'] = learning_rate
mgrid_ae_smm_ok_setting['discount_rate'] = discount_rate
mgrid_ae_smm_ok_setting['nn_num_layers'] = nn_num_layers
mgrid_ae_smm_ok_setting['nn_layer_size'] = nn_layer_size
mgrid_ae_smm_ok_setting['n_steps'] = n_steps
mgrid_ae_smm_ok_setting['batch_size'] = batch_size
mgrid_ae_smm_ok_setting['memory_type'] = 3
mgrid_ae_smm_ok_setting['memory_length'] = 1
mgrid_ae_smm_ok_setting['intrinsic_enabled'] = False
mgrid_ae_smm_ok_setting['intrinsic_beta'] = 0.01
mgrid_ae_smm_ok_setting['ae_enabled'] = True
mgrid_ae_smm_ok_setting['ae_comm_list'] = comm_dict['ae_smm_ok']
mgrid_ae_smm_ok_setting['ae_path'] = "models/ae.torch"
mgrid_ae_smm_ok_setting['ae_rcons_err_type'] = "MSE"
mgrid_ae_smm_ok_setting['eval_enabled'] = True
mgrid_ae_smm_ok_setting['eval_freq'] = total_timesteps//200
mgrid_ae_smm_ok_setting['eval_episodes'] = 5
mgrid_ae_smm_ok_setting['eval_path'] = None
mgrid_ae_smm_ok_setting['env_n_proc'] = env_n_proc
mgrid_ae_smm_ok_setting['vec_env_cls'] = vec_env_cls
mgrid_ae_smm_ok_setting['tb_log_name'] = "ae_smm_ok"
mgrid_ae_smm_ok_setting['tb_log_dir'] = "./logs/c_minigrid_tb/"
mgrid_ae_smm_ok_setting['maze_length'] = maze_length
mgrid_ae_smm_ok_setting['total_timesteps'] = total_timesteps
mgrid_ae_smm_ok_setting['seed'] = None
mgrid_ae_smm_ok_setting['policy'] = MlpACPolicy
mgrid_ae_smm_ok_setting['save'] = True
mgrid_ae_smm_ok_setting['device'] = 'cuda:0'
mgrid_ae_smm_ok_setting['train_func'] = train_ppo_agent


mgrid_ae_smm_ok_intr_setting = {}
mgrid_ae_smm_ok_intr_setting['envClass'] = envClass
mgrid_ae_smm_ok_intr_setting['learning_rate'] = learning_rate
mgrid_ae_smm_ok_intr_setting['discount_rate'] = discount_rate
mgrid_ae_smm_ok_intr_setting['nn_num_layers'] = nn_num_layers
mgrid_ae_smm_ok_intr_setting['nn_layer_size'] = nn_layer_size
mgrid_ae_smm_ok_intr_setting['n_steps'] = n_steps
mgrid_ae_smm_ok_intr_setting['batch_size'] = batch_size
mgrid_ae_smm_ok_intr_setting['memory_type'] = 3
mgrid_ae_smm_ok_intr_setting['memory_length'] = 1
mgrid_ae_smm_ok_intr_setting['intrinsic_enabled'] = True
mgrid_ae_smm_ok_intr_setting['intrinsic_beta'] = 0.01
mgrid_ae_smm_ok_intr_setting['ae_enabled'] = True
mgrid_ae_smm_ok_intr_setting['ae_comm_list'] = comm_dict['ae_smm_ok_intr']
mgrid_ae_smm_ok_intr_setting['ae_path'] = "models/ae.torch"
mgrid_ae_smm_ok_intr_setting['ae_rcons_err_type'] = "MSE"
mgrid_ae_smm_ok_intr_setting['eval_enabled'] = True
mgrid_ae_smm_ok_intr_setting['eval_freq'] = total_timesteps//200
mgrid_ae_smm_ok_intr_setting['eval_episodes'] = 5
mgrid_ae_smm_ok_intr_setting['eval_path'] = None
mgrid_ae_smm_ok_intr_setting['env_n_proc'] = env_n_proc
mgrid_ae_smm_ok_intr_setting['vec_env_cls'] = vec_env_cls
mgrid_ae_smm_ok_intr_setting['tb_log_name'] = "ae_smm_ok_intr"
mgrid_ae_smm_ok_intr_setting['tb_log_dir'] = "./logs/c_minigrid_tb/"
mgrid_ae_smm_ok_intr_setting['maze_length'] = maze_length
mgrid_ae_smm_ok_intr_setting['total_timesteps'] = total_timesteps
mgrid_ae_smm_ok_intr_setting['seed'] = None
mgrid_ae_smm_ok_intr_setting['policy'] = MlpACPolicy
mgrid_ae_smm_ok_intr_setting['save'] = True
mgrid_ae_smm_ok_intr_setting['device'] = 'cuda:0'
mgrid_ae_smm_ok_intr_setting['train_func'] = train_ppo_agent


mgrid_ae_smm_oak_setting = {}
mgrid_ae_smm_oak_setting['envClass'] = envClass
mgrid_ae_smm_oak_setting['learning_rate'] = learning_rate
mgrid_ae_smm_oak_setting['discount_rate'] = discount_rate
mgrid_ae_smm_oak_setting['nn_num_layers'] = nn_num_layers
mgrid_ae_smm_oak_setting['nn_layer_size'] = nn_layer_size
mgrid_ae_smm_oak_setting['n_steps'] = n_steps
mgrid_ae_smm_oak_setting['batch_size'] = batch_size
mgrid_ae_smm_oak_setting['memory_type'] = 4
mgrid_ae_smm_oak_setting['memory_length'] = 1
mgrid_ae_smm_oak_setting['intrinsic_enabled'] = False
mgrid_ae_smm_oak_setting['intrinsic_beta'] = 0.01
mgrid_ae_smm_oak_setting['ae_enabled'] = True
mgrid_ae_smm_oak_setting['ae_comm_list'] = comm_dict['ae_smm_oak']
mgrid_ae_smm_oak_setting['ae_path'] = "models/ae.torch"
mgrid_ae_smm_oak_setting['ae_rcons_err_type'] = "MSE"
mgrid_ae_smm_oak_setting['eval_enabled'] = True
mgrid_ae_smm_oak_setting['eval_freq'] = total_timesteps//200
mgrid_ae_smm_oak_setting['eval_episodes'] = 5
mgrid_ae_smm_oak_setting['eval_path'] = None
mgrid_ae_smm_oak_setting['env_n_proc'] = env_n_proc
mgrid_ae_smm_oak_setting['vec_env_cls'] = vec_env_cls
mgrid_ae_smm_oak_setting['tb_log_name'] = "ae_smm_oak"
mgrid_ae_smm_oak_setting['tb_log_dir'] = "./logs/c_minigrid_tb/"
mgrid_ae_smm_oak_setting['maze_length'] = maze_length
mgrid_ae_smm_oak_setting['total_timesteps'] = total_timesteps
mgrid_ae_smm_oak_setting['seed'] = None
mgrid_ae_smm_oak_setting['policy'] = MlpACPolicy
mgrid_ae_smm_oak_setting['save'] = True
mgrid_ae_smm_oak_setting['device'] = 'cuda:0'
mgrid_ae_smm_oak_setting['train_func'] = train_ppo_agent


mgrid_ae_smm_oak_intr_setting = {}
mgrid_ae_smm_oak_intr_setting['envClass'] = envClass
mgrid_ae_smm_oak_intr_setting['learning_rate'] = learning_rate
mgrid_ae_smm_oak_intr_setting['discount_rate'] = discount_rate
mgrid_ae_smm_oak_intr_setting['nn_num_layers'] = nn_num_layers
mgrid_ae_smm_oak_intr_setting['nn_layer_size'] = nn_layer_size
mgrid_ae_smm_oak_intr_setting['n_steps'] = n_steps
mgrid_ae_smm_oak_intr_setting['batch_size'] = batch_size
mgrid_ae_smm_oak_intr_setting['memory_type'] = 4
mgrid_ae_smm_oak_intr_setting['memory_length'] = 1
mgrid_ae_smm_oak_intr_setting['intrinsic_enabled'] = True
mgrid_ae_smm_oak_intr_setting['intrinsic_beta'] = 0.01
mgrid_ae_smm_oak_intr_setting['ae_enabled'] = True
mgrid_ae_smm_oak_intr_setting['ae_comm_list'] = comm_dict['ae_smm_oak_intr']
mgrid_ae_smm_oak_intr_setting['ae_path'] = "models/ae.torch"
mgrid_ae_smm_oak_intr_setting['ae_rcons_err_type'] = "MSE"
mgrid_ae_smm_oak_intr_setting['eval_enabled'] = True
mgrid_ae_smm_oak_intr_setting['eval_freq'] = total_timesteps//200
mgrid_ae_smm_oak_intr_setting['eval_episodes'] = 5
mgrid_ae_smm_oak_intr_setting['eval_path'] = None
mgrid_ae_smm_oak_intr_setting['env_n_proc'] = env_n_proc
mgrid_ae_smm_oak_intr_setting['vec_env_cls'] = vec_env_cls
mgrid_ae_smm_oak_intr_setting['tb_log_name'] = "ae_smm_oak_intr"
mgrid_ae_smm_oak_intr_setting['tb_log_dir'] = "./logs/c_minigrid_tb/"
mgrid_ae_smm_oak_intr_setting['maze_length'] = maze_length
mgrid_ae_smm_oak_intr_setting['total_timesteps'] = total_timesteps
mgrid_ae_smm_oak_intr_setting['seed'] = None
mgrid_ae_smm_oak_intr_setting['policy'] = MlpACPolicy
mgrid_ae_smm_oak_intr_setting['save'] = True
mgrid_ae_smm_oak_intr_setting['device'] = 'cuda:0'
mgrid_ae_smm_oak_intr_setting['train_func'] = train_ppo_agent


mgrid_ae_lstm_setting = {}
mgrid_ae_lstm_setting['envClass'] = envClass
mgrid_ae_lstm_setting['learning_rate'] = learning_rate
mgrid_ae_lstm_setting['discount_rate'] = discount_rate
mgrid_ae_lstm_setting['nn_num_layers'] = nn_num_layers
mgrid_ae_lstm_setting['nn_layer_size'] = nn_layer_size
mgrid_ae_lstm_setting['n_steps'] = n_steps
mgrid_ae_lstm_setting['batch_size'] = batch_size
mgrid_ae_lstm_setting['memory_type'] = 5
mgrid_ae_lstm_setting['memory_length'] = 1
mgrid_ae_lstm_setting['intrinsic_enabled'] = False
mgrid_ae_lstm_setting['intrinsic_beta'] = 0.01
mgrid_ae_lstm_setting['ae_enabled'] = True
mgrid_ae_lstm_setting['ae_comm_list'] = comm_dict['ae_lstm']
mgrid_ae_lstm_setting['ae_path'] = "models/ae.torch"
mgrid_ae_lstm_setting['ae_rcons_err_type'] = "MSE"
mgrid_ae_lstm_setting['eval_enabled'] = True
mgrid_ae_lstm_setting['eval_freq'] = total_timesteps//200
mgrid_ae_lstm_setting['eval_episodes'] = 5
mgrid_ae_lstm_setting['eval_path'] = None
mgrid_ae_lstm_setting['env_n_proc'] = env_n_proc
mgrid_ae_lstm_setting['vec_env_cls'] = vec_env_cls
mgrid_ae_lstm_setting['tb_log_name'] = "ae_lstm"
mgrid_ae_lstm_setting['tb_log_dir'] = "./logs/c_minigrid_tb/"
mgrid_ae_lstm_setting['maze_length'] = maze_length
mgrid_ae_lstm_setting['total_timesteps'] = total_timesteps
mgrid_ae_lstm_setting['seed'] = None
mgrid_ae_lstm_setting['policy'] = "MlpLstmPolicy"
mgrid_ae_lstm_setting['save'] = True
mgrid_ae_lstm_setting['device'] = 'cuda:0'
mgrid_ae_lstm_setting['train_func'] = train_ppo_lstm_agent


mgrid_cnn_no_mem_setting = {}
mgrid_cnn_no_mem_setting['envClass'] = envClass
mgrid_cnn_no_mem_setting['learning_rate'] = learning_rate
mgrid_cnn_no_mem_setting['discount_rate'] = discount_rate
mgrid_cnn_no_mem_setting['nn_num_layers'] = nn_num_layers
mgrid_cnn_no_mem_setting['nn_layer_size'] = nn_layer_size
mgrid_cnn_no_mem_setting['n_steps'] = n_steps
mgrid_cnn_no_mem_setting['batch_size'] = batch_size
mgrid_cnn_no_mem_setting['memory_type'] = 0
mgrid_cnn_no_mem_setting['memory_length'] = 1
mgrid_cnn_no_mem_setting['intrinsic_enabled'] = False
mgrid_cnn_no_mem_setting['intrinsic_beta'] = 0.01
mgrid_cnn_no_mem_setting['ae_enabled'] = False
mgrid_cnn_no_mem_setting['ae_comm_list'] = comm_dict['cnn_no_mem']
mgrid_cnn_no_mem_setting['ae_path'] = "models/ae.torch"
mgrid_cnn_no_mem_setting['ae_rcons_err_type'] = "MSE"
mgrid_cnn_no_mem_setting['eval_enabled'] = True
mgrid_cnn_no_mem_setting['eval_freq'] = total_timesteps//200
mgrid_cnn_no_mem_setting['eval_episodes'] = 5
mgrid_cnn_no_mem_setting['eval_path'] = None
mgrid_cnn_no_mem_setting['env_n_proc'] = env_n_proc
mgrid_cnn_no_mem_setting['vec_env_cls'] = vec_env_cls
mgrid_cnn_no_mem_setting['tb_log_name'] = "cnn_no_mem"
mgrid_cnn_no_mem_setting['tb_log_dir'] = "./logs/c_minigrid_tb/"
mgrid_cnn_no_mem_setting['maze_length'] = maze_length
mgrid_cnn_no_mem_setting['total_timesteps'] = total_timesteps
mgrid_cnn_no_mem_setting['seed'] = None
mgrid_cnn_no_mem_setting['policy'] = CNNACPolicy
mgrid_cnn_no_mem_setting['save'] = True
mgrid_cnn_no_mem_setting['device'] = 'cuda:0'
mgrid_cnn_no_mem_setting['train_func'] = train_ppo_agent
