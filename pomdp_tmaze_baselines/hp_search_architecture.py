import os
import sys
import signal
import numpy as np
import torch.multiprocessing as mp
import optuna
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv

from pomdp_tmaze_baselines.EnvTMaze import TMazeEnvMemoryWrapped
from pomdp_tmaze_baselines.utils.UtilPolicies import MlpACPolicy
from pomdp_tmaze_baselines.utils.UtilStableAgents import train_ppo_lstm_agent,\
    train_ppo_agent


# Select the hyperparameter search script:
# from pomdp_tmaze_baselines.params.hp_ppo_params import hyper_parameters
from pomdp_tmaze_baselines.params.hp_comp_arch import hyper_parameters
from pomdp_tmaze_baselines.params.hp_comp_arch import study_name


# Static parameters
number_of_parallel_jobs = 8
storage_url = "mysql://root:1234@127.0.0.1/pomdp"

# 1- Mysql RDB, used by default.
storage = optuna.storages.RDBStorage(
    url=storage_url,
    engine_kwargs={
        'pool_size': 128,
        'max_overflow': 0
    })


# 2- Sqlite file RDB
# storage = optuna.storages.RDBStorage(
#     url="sqlite:///pomdp.db",
#     engine_kwargs={
#         'pool_size': 128,
#         'max_overflow': 0
#     })

envClass = TMazeEnvMemoryWrapped
log_dir = "./logs/" + study_name + "/"

list_of_dict_lengths = [len(v) for k, v in hyper_parameters.items()]
total_number_of_trials = np.prod(np.array(list_of_dict_lengths))

interrupt_max_count = 3


def signal_handler(sig, frame):
    print(
        "Keyboard interrupt detected, sending stop signal " +
        "to the study and waiting to existing methods finish.")
    study = optuna.load_study(study_name=study_name, storage=storage)
    study.set_user_attr('stop_signal', True)

    if "stop_signal_count" in study.user_attrs:
        count = 1 + study.user_attrs['stop_signal_count']
        study.set_user_attr(
            'stop_signal_count', count)
        if count == interrupt_max_count:
            print(
                str(interrupt_max_count) +
                " consequtive Ctrl-C detected, exiting now.")
            sys.exit(0)
        else:
            print(
                "Please Ctrl-C "+str(interrupt_max_count-count) +
                " more times to exit the execution immediately.")
    else:
        study.set_user_attr('stop_signal_count', 1)
        print(
            "Please Ctrl-C "+str(interrupt_max_count-1) +
            " more times to exit the execution immediately.")


def stop_callback(study, frozen_trial):
    if len(study.trials) >= total_number_of_trials:
        study.stop()
    if "stop_signal" in study.user_attrs:
        if study.user_attrs["stop_signal"]:
            study.stop()


def start_optimization(
        storage, cuda_device_id: str = "0",
        study_name="architecture_search"):

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
    study.optimize(objective, callbacks=[stop_callback], gc_after_trial=True)


def objective(trial):
    # For parallel and multi gpu search, CUDA_VISIBLE_DEVICES
    # environment variable is used.
    # So at any time, there will be only one gpu is set and
    # the default device can be used with the name "cuda"
    device = "cuda"

    trial.suggest_categorical(
        "experiment_no", hyper_parameters['experiment_no'])

    memory_type = trial.suggest_categorical(
        "memory_type", hyper_parameters['memory_type'])

    if memory_type == 5:
        train_func = train_ppo_lstm_agent
        policy = "MlpLstmPolicy"
    else:
        train_func = train_ppo_agent
        policy = MlpACPolicy

    if memory_type != 0 and memory_type != 2 and memory_type != 5:
        intrinsic_enabled = trial.suggest_categorical(
            "intrinsic_enabled", hyper_parameters['intrinsic_enabled'])
    else:
        intrinsic_enabled = 0

    memory_length = trial.suggest_categorical(
        "memory_length", hyper_parameters['memory_length'])

    total_timesteps = trial.suggest_categorical(
        "total_timesteps", hyper_parameters['total_timesteps'])

    maze_length = trial.suggest_categorical(
        "maze_length", hyper_parameters['maze_length'])

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
    learning_setting['memory_type'] = memory_type
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
            if model.env.envs[0].episode_count >= (
                    total_timesteps / maze_length)*0.25:
                success_ratio = (
                    model.env.envs[0].success_count
                    / model.env.envs[0].episode_count
                    ) * 100
            else:
                success_ratio = 0.0
    else:
        if model.env.envs[0].episode_count == 0:
            success_ratio = 0.0
        else:
            if model.env.envs[0].episode_count >= (
                    total_timesteps / maze_length)*0.25:
                success_ratio = (
                    model.env.success_count / model.env.episode_count) * 100
            else:
                success_ratio = 0.0
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
    signal.signal(signal.SIGINT, signal.SIG_IGN)

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
                    kwargs={'storage': storage,
                            'cuda_device_id': str(device_id),
                            'study_name': study_name
                            })
            p.start()
            processes.append(p)

    signal.signal(signal.SIGINT, signal_handler)

    for p in processes:
        p.join()
    print("Execution has been completed.")
    # study = optuna.load_study(study_name=study_name, storage=storage)
    # print("Best params: ", study.best_params)
    # print("Best value: ", study.best_value)


if __name__ == "__main__":
    main()
