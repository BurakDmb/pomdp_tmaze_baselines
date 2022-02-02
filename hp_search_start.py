import os
import sys
import numpy as np
import torch.multiprocessing as mp
import optuna
from EnvTMaze import TMazeEnvMemoryWrapped
from utils.UtilPolicies import MlpACPolicy
from utils.UtilStableAgents import train_ppo_lstm_agent, train_ppo_agent


# Turn off optuna logging.
optuna.logging.set_verbosity(optuna.logging.WARN)


def logging_callback(study, frozen_trial):
    previous_best_value = study.user_attrs.get("previous_best_value", None)
    if previous_best_value != study.best_value:
        study.set_user_attr("previous_best_value", study.best_value)
        print(
            "Trial {} finished with best value: {} and parameters: {}. "
            .format(
                frozen_trial.number,
                frozen_trial.value,
                frozen_trial.params,
            )
        )


def start_optimization(cuda_device_id=0, n_trials=100):
    # With this, the visible devices seen from this process is set to be the
    # device with cuda_device_id.

    # By default, the 0th index cuda device is used.
    # For multi gpu purposes, one can pass the cuda_device_id to make this
    # process uses only that gpu device.
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device_id

    study = optuna.create_study(
        storage="mysql://root:1234@10.1.46.229/example",
        study_name="distributed-example",
        direction="maximize",
        load_if_exists=True
        )

    study.optimize(objective, n_trials=n_trials, callbacks=[logging_callback])
    pass


def objective(trial):
    # Static parameters
    total_timesteps = 1_000_000
    maze_length = 10
    envClass = TMazeEnvMemoryWrapped

    # For parallel and multi gpu search, CUDA_VISIBLE_DEVICES
    # environment variable is used.
    # So at any time, there will be only one gpu is set and
    # the default device can be used with the name "cuda"
    device = "cuda"

    # Total combination of 1 float range and 3*3*3*4*3*2 = 648 discrete
    # possible hyperparameter combination.

    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    nn_num_layers = trial.suggest_categorical("nn_num_layers", [2, 4, 8])
    nn_layer_size = trial.suggest_categorical("nn_layer_size", [4, 8, 32])

    # n_steps and batch size is related with each other, need to check.
    # n_steps = trial.suggest_int("n_steps", 32, 512, 32)
    batch_size = trial.suggest_int("batch_size", 32, 512, 32)

    memory_type = trial.suggest_categorical("memory_type", [0, 3, 4, 5])
    memory_length = trial.suggest_categorical("memory_length", [1, 3, 10])
    intrinsic_enabled = trial.suggest_categorical("intrinsic_enabled", [0, 1])

    if memory_type == 5:
        train_func = train_ppo_lstm_agent
        policy = "MlpLstmPolicy"
        memory_type_category = 0
    else:
        train_func = train_ppo_agent
        policy = MlpACPolicy
        memory_type_category = memory_type

    # TODO: Update log name with gathering from the param names.
    log_name = "no_memory"

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
    learning_setting['tb_log_dir'] = "./logs/hyper_param_search/"
    learning_setting['maze_length'] = maze_length
    learning_setting['total_timesteps'] = total_timesteps
    learning_setting['seed'] = None
    learning_setting['policy'] = policy
    learning_setting['save'] = False
    learning_setting['device'] = device
    learning_setting['train_func'] = train_func

    model = learning_setting['train_func'](learning_setting=learning_setting)

    # TODO: Add evaluation metric and return evaluation result
    # Be aware of the minimization/maximization objective
    # is alligned with optuna.

    evaluation_result = 0
    return evaluation_result


# Usage:
# Arg1: multigpu or cpu
# Arg2: GPU count, used if multigpu exists.
# python hp_search_start.py multigpu 4
def main():
    number_of_parallel_experiments = 1
    n_trials = 100
    params = len(sys.argv)
    if params == 3 and sys.argv[1] == 'multigpu':
        # multi gpu and its count is converted into an arange array
        # ex: [0, 1, 2, 3]
        cuda_device = np.arange(sys.argv[2]).tolist()
    else:
        # single gpu
        cuda_device = [0]

    mp.set_start_method('spawn')
    processes = []
    for device_id in cuda_device:
        for _ in range(number_of_parallel_experiments):
            p = mp.Process(
                    target=start_optimization,
                    kwargs={'cuda_device_id': device_id,
                            'n_trials': n_trials})
            p.start()
            processes.append(p)

    for p in processes:
        p.join()


if __name__ == "__main__":
    main()
