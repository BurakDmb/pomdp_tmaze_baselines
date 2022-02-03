import os
import sys
import numpy as np
import torch.multiprocessing as mp
import optuna
from EnvTMaze import TMazeEnvMemoryWrapped
from utils.UtilPolicies import MlpACPolicy
from utils.UtilStableAgents import train_ppo_lstm_agent, train_ppo_agent
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv

# Static parameters
number_of_parallel_jobs = 8
study_name = "architecture_search"
storage = "mysql://root:1234@127.0.0.1/pomdp"

total_timesteps = 1_000_000
maze_length = 10
envClass = TMazeEnvMemoryWrapped
log_dir = "./logs/hyper_param_search/"

# Number of each hyperparameter is required to be in 2's power. (Ex: 2,4,8,...)
hyper_parameters = {}
hyper_parameters['learning_rate'] = [1e-7, 1e-5, 1e-4, 1e-3]
hyper_parameters['nn_num_layers'] = [4, 8]
hyper_parameters['nn_layer_size'] = [4, 8, 32, 128]
hyper_parameters['batch_size'] = [32, 128, 256, 512]
hyper_parameters['memory_type'] = [0, 3, 4, 5]
hyper_parameters['memory_length'] = [1, 3, 10, 20]
hyper_parameters['intrinsic_enabled'] = [0, 1]

# Total combination of 4*2*4*4*4*4*2 = 4096
# possible hyperparameter combination.
list_of_dict_lengths = [len(v) for k, v in hyper_parameters.items()]
total_number_of_trials = np.prod(np.array(list_of_dict_lengths))


def stop_callback(study, frozen_trial):
    if len(study.trials) >= total_number_of_trials:
        study.stop()


def start_optimization(cuda_device_id: str = "0",
                       study_name="architecture_search",
                       storage="mysql://root:1234@127.0.0.1/pomdp"):

    # Turn off optuna logging.
    optuna.logging.set_verbosity(optuna.logging.WARN)

    # With this, the visible devices seen from this process is set to be the
    # device with cuda_device_id.

    # By default, the 0th index cuda device is used.
    # For multi gpu purposes, one can pass the cuda_device_id to make this
    # process uses only that gpu device.
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device_id

    # Debug line, remove later:
    # print("CUDA_VISIBLE_DEVICES: ", os.getenv('CUDA_VISIBLE_DEVICES', '0'))
    # print("Torch Current Device: ", torch.cuda.current_device())

    study = optuna.load_study(study_name=study_name, storage=storage)

    study.optimize(objective, callbacks=[stop_callback])


def objective(trial):
    # For parallel and multi gpu search, CUDA_VISIBLE_DEVICES
    # environment variable is used.
    # So at any time, there will be only one gpu is set and
    # the default device can be used with the name "cuda"
    device = "cuda"

    learning_rate = trial.suggest_categorical(
        "learning_rate", hyper_parameters['learning_rate'])

    nn_num_layers = trial.suggest_categorical(
        "nn_num_layers", hyper_parameters['nn_num_layers'])

    nn_layer_size = trial.suggest_categorical(
        "nn_layer_size", hyper_parameters['nn_layer_size'])

    # n_steps and batch size is related with each other, need to check.
    # n_steps = trial.suggest_int("n_steps", 32, 512, 32)
    batch_size = trial.suggest_categorical(
        "batch_size", hyper_parameters['batch_size'])

    memory_type = trial.suggest_categorical(
        "memory_type", hyper_parameters['memory_type'])

    memory_length = trial.suggest_categorical(
        "memory_length", hyper_parameters['memory_length'])

    intrinsic_enabled = trial.suggest_categorical(
        "intrinsic_enabled", hyper_parameters['intrinsic_enabled'])

    if memory_type == 5:
        train_func = train_ppo_lstm_agent
        policy = "MlpLstmPolicy"
        memory_type_category = 0
    else:
        train_func = train_ppo_agent
        policy = MlpACPolicy
        memory_type_category = memory_type

    log_name = str(trial.number)+"."+str(trial.params).replace(" ", "")\
        .replace("'", "").replace(":", "_").replace(",", ".")\
        .replace("{", "").replace("}", "")

    learning_setting = {}
    learning_setting['envClass'] = envClass
    learning_setting['learning_rate'] = learning_rate
    learning_setting['discount_rate'] = 0.99
    learning_setting['nn_num_layers'] = nn_num_layers
    learning_setting['nn_layer_size'] = nn_layer_size
    learning_setting['n_steps'] = batch_size  # assumed equal with n_steps
    learning_setting['batch_size'] = batch_size
    learning_setting['memory_type'] = memory_type_category
    learning_setting['memory_length'] = memory_length
    learning_setting['intrinsic_enabled'] = intrinsic_enabled
    learning_setting['intrinsic_beta'] = 0.5
    learning_setting['tb_log_name'] = log_name
    learning_setting['tb_log_dir'] = log_dir
    learning_setting['maze_length'] = maze_length
    learning_setting['total_timesteps'] = total_timesteps
    learning_setting['seed'] = None
    learning_setting['policy'] = policy
    learning_setting['save'] = False
    learning_setting['device'] = device
    learning_setting['train_func'] = train_func

    model = learning_setting['train_func'](learning_setting=learning_setting)

    # For evaluation metric, success ratio of this model is calculated by
    # dividing total success count to the total episode count.
    # Range of this metric is 0-100 and higher is better.
    # The optimization is aimed to maximize this evaluation metric.
    if isinstance(model.env, DummyVecEnv):
        if model.env.envs[0].episode_count == 0:
            success_ratio = 0.0
        else:
            success_ratio = (
                model.env.envs[0].success_count
                / model.env.envs[0].episode_count
                ) * 100
    else:
        success_ratio = (
            model.env.success_count / model.env.episode_count) * 100
    print("Trial: ", trial.number, ", Score: ", success_ratio)
    return success_ratio


# hp_search_architecture.py main function
# Usage:
# Arg1: multigpu or singlegpu
# Arg2: GPU count, used if multigpu exists.

# Parallel jobs with 4 gpu:
# python hp_search_architecture.py multigpu 4

# No parallelization, single gpu:
# python hp_search_architecture.py

# Deleting existing study in the database, not in logs file.:
# python hp_search_architecture.py delete
def main():
    # Turn off optuna logging.
    optuna.logging.set_verbosity(optuna.logging.WARN)

    gpu_count = 1
    params = len(sys.argv)
    if params == 2 and sys.argv[1] == 'delete':
        try:
            optuna.delete_study(
                study_name=study_name,
                storage=storage)
        except KeyError:
            print(
                "There is no study defined with name: ", study_name,
                ". No deletion required.")
        else:
            print("Successfully deleted the study")
        return
    if params == 3 and sys.argv[1] == 'multigpu':
        # Multi gpu and its count is converted into an arange array
        # ex: [0, 1, 2, 3]
        # For single gpu this list is only [0]
        gpu_count = int(sys.argv[2])

    cuda_devices = np.arange(gpu_count).tolist()

    optuna.create_study(
        storage=storage,
        study_name=study_name,
        direction="maximize",
        load_if_exists=True
        )

    mp.set_start_method('spawn')
    processes = []
    for device_id in cuda_devices:
        for _ in range(number_of_parallel_jobs):
            p = mp.Process(
                    target=start_optimization,
                    kwargs={'cuda_device_id': str(device_id),
                            'study_name': study_name,
                            'storage': storage})
            p.start()
            processes.append(p)

    for p in processes:
        p.join()

    study = optuna.load_study(study_name=study_name, storage=storage)
    print("Best params: ", study.best_params)
    print("Best value: ", study.best_value)


if __name__ == "__main__":
    main()
