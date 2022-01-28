import optuna

# WORK IN PROGRESS

# Turn off optuna log notes.
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


def objective(trial):
    x = trial.suggest_float("x", -10, 10)
    return (x - 2) ** 2


if __name__ == "__main__":
    study = optuna.load_study(
        study_name="distributed-example",
        storage="mysql://root:1234@10.1.46.229/example"
    )
    study.optimize(objective, n_trials=100, callbacks=[logging_callback])


# import sys
# import torch.multiprocessing as mp

# if __name__ == '__main__':
#     # from test_all import unittest_main
#     # unittest_main()

#     params = len(sys.argv)
#     if params == 2 and sys.argv[1] == 'multigpu':
#         pass
#     else:
#         from params.arch_comp_gpu import number_of_parallel_experiments
#         from params.arch_comp_gpu import start_no_memory
#         from params.arch_comp_gpu import start_no_memory_intr
#         from params.arch_comp_gpu import start_o_k_memory
#         from params.arch_comp_gpu import start_o_k_intr_memory
#         from params.arch_comp_gpu import start_oa_k_memory
#         from params.arch_comp_gpu import start_oa_k_intr_memory
#         from params.arch_comp_gpu import start_lstm
#         from params.arch_comp_gpu import start_lstm_intr
#         from params.arch_comp_gpu import no_memory_learning_setting
#         from params.arch_comp_gpu import no_memory_intr_learning_setting
#         from params.arch_comp_gpu import o_k_memory_learning_setting
#         from params.arch_comp_gpu import o_k_intr_memory_learning_setting
#         from params.arch_comp_gpu import oa_k_memory_learning_setting
#         from params.arch_comp_gpu import oa_k_intr_memory_learning_setting
#         from params.arch_comp_gpu import lstm_learning_setting
#         from params.arch_comp_gpu import lstm_intr_learning_setting

#     mp.set_start_method('spawn')
#     processes = []

#     for _ in range(number_of_parallel_experiments):

#         if start_no_memory:
#             p = mp.Process(
#                 target=no_memory_learning_setting['train_func'],
#                 kwargs={'learning_setting': no_memory_learning_setting})
#             p.start()
#             processes.append(p)

#         if start_no_memory_intr:
#             p = mp.Process(
#                 target=no_memory_intr_learning_setting['train_func'],
#                 kwargs={'learning_setting': no_memory_intr_learning_setting})
#             p.start()
#             processes.append(p)

#         if start_o_k_memory:
#             p = mp.Process(
#                 target=o_k_memory_learning_setting['train_func'],
#                 kwargs={'learning_setting': o_k_memory_learning_setting})
#             p.start()
#             processes.append(p)

#         if start_o_k_intr_memory:
#             p = mp.Process(
#                 target=o_k_intr_memory_learning_setting['train_func'],
#                 kwargs={'learning_setting': o_k_intr_memory_learning_setting})
#             p.start()
#             processes.append(p)

#         if start_oa_k_memory:
#             p = mp.Process(
#                 target=o_k_memory_learning_setting['train_func'],
#                 kwargs={'learning_setting': oa_k_memory_learning_setting})
#             p.start()
#             processes.append(p)

#         if start_oa_k_intr_memory:
#             p = mp.Process(
#                 target=o_k_intr_memory_learning_setting['train_func'],
#                 kwargs={'learning_setting': oa_k_intr_memory_learning_setting})
#             p.start()
#             processes.append(p)

#         if start_lstm:
#             p = mp.Process(
#                 target=lstm_learning_setting['train_func'],
#                 kwargs={'learning_setting': lstm_learning_setting})
#             p.start()
#             processes.append(p)

#         if start_lstm_intr:
#             p = mp.Process(
#                 target=lstm_intr_learning_setting['train_func'],
#                 kwargs={'learning_setting': lstm_intr_learning_setting})
#             p.start()
#             processes.append(p)

#     for p in processes:
#         p.join()
