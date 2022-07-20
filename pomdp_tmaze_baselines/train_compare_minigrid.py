import sys
import torch.multiprocessing as mp
from tqdm.auto import tqdm


if __name__ == '__main__':
    # from test_all import unittest_main
    # unittest_main()

    params = len(sys.argv)
    if params == 2 and sys.argv[1] == 'multigpu':
        pass
        # TODO: minigrid_mgpu param file needs to be created.
        # from pomdp_tmaze_baselines.params.alg_minigrid_mgpu import (
        #     number_of_parallel_experiments,
        #     start_ppo,
        #     start_smm, start_smm_intr,
        #     start_lstm,
        #     mgrid_ppo_learning_setting,
        #     mgrid_smm_learning_setting,
        #     mgrid_smm_intr_learning_setting,
        #     mgrid_lstm_learning_setting,
        # )
    else:
        from pomdp_tmaze_baselines.params.alg_minigrid import (
            number_of_parallel_experiments,
            start_ppo,
            start_smm, start_smm_intr,
            start_lstm,
            mgrid_ppo_learning_setting,
            mgrid_smm_learning_setting,
            mgrid_smm_intr_learning_setting,
            mgrid_lstm_learning_setting,
        )

    mp.set_start_method('spawn')
    processes = []

    for _ in range(number_of_parallel_experiments):

        if start_ppo:
            p = mp.Process(
                target=mgrid_ppo_learning_setting['train_func'],
                kwargs={'learning_setting': mgrid_ppo_learning_setting})
            p.start()
            processes.append(p)

        if start_smm:
            p = mp.Process(
                target=mgrid_smm_learning_setting['train_func'],
                kwargs={'learning_setting': mgrid_smm_learning_setting})
            p.start()
            processes.append(p)

        if start_smm_intr:
            p = mp.Process(
                target=mgrid_smm_intr_learning_setting['train_func'],
                kwargs={'learning_setting': mgrid_smm_intr_learning_setting})
            p.start()
            processes.append(p)

        if start_lstm:
            p = mp.Process(
                target=mgrid_lstm_learning_setting['train_func'],
                kwargs={'learning_setting': mgrid_lstm_learning_setting})
            p.start()
            processes.append(p)

    print("Training started with selected architectures " +
          "(each architecture experiment will run with " +
          str(number_of_parallel_experiments) + " parallel run/runs).")

    for p in tqdm(processes):
        p.join()

# TODO: complete minigrid trainings.
