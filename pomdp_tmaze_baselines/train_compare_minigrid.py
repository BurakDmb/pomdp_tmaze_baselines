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
        #     start_ae_ppo, start_cnn_ppo,
        #     start_ae_smm, start_cnn_smm,
        #     start_ae_smm_intr,
        #     start_ae_lstm, start_cnn_lstm,
        #     mgrid_ae_ppo_setting, mgrid_cnn_ppo_setting,
        #     mgrid_ae_smm_setting, mgrid_cnn_smm_setting,
        #     mgrid_ae_smm_intr_setting,
        #     mgrid_ae_lstm_setting, mgrid_cnn_lstm_setting,
        # )
    else:
        from pomdp_tmaze_baselines.params.alg_minigrid import (
            number_of_parallel_experiments,
            start_ae_ppo, start_cnn_ppo,
            start_ae_smm, start_cnn_smm,
            start_ae_smm_intr,
            start_ae_lstm, start_cnn_lstm,
            mgrid_ae_ppo_setting, mgrid_cnn_ppo_setting,
            mgrid_ae_smm_setting, mgrid_cnn_smm_setting,
            mgrid_ae_smm_intr_setting,
            mgrid_ae_lstm_setting, mgrid_cnn_lstm_setting,
        )

    mp.set_start_method('spawn')
    processes = []

    for _ in range(number_of_parallel_experiments):

        if start_ae_ppo:
            p = mp.Process(
                target=mgrid_ae_ppo_setting['train_func'],
                kwargs={'learning_setting': mgrid_ae_ppo_setting})
            p.start()
            processes.append(p)

        if start_cnn_ppo:
            p = mp.Process(
                target=mgrid_cnn_ppo_setting['train_func'],
                kwargs={'learning_setting': mgrid_cnn_ppo_setting})
            p.start()
            processes.append(p)

        if start_ae_smm:
            p = mp.Process(
                target=mgrid_ae_smm_setting['train_func'],
                kwargs={'learning_setting': mgrid_ae_smm_setting})
            p.start()
            processes.append(p)

        if start_cnn_smm:
            p = mp.Process(
                target=mgrid_cnn_smm_setting['train_func'],
                kwargs={'learning_setting': mgrid_cnn_smm_setting})
            p.start()
            processes.append(p)

        if start_ae_smm_intr:
            p = mp.Process(
                target=mgrid_ae_smm_intr_setting['train_func'],
                kwargs={'learning_setting': mgrid_ae_smm_intr_setting})
            p.start()
            processes.append(p)

        if start_ae_lstm:
            p = mp.Process(
                target=mgrid_ae_lstm_setting['train_func'],
                kwargs={'learning_setting': mgrid_ae_lstm_setting})
            p.start()
            processes.append(p)

        if start_cnn_lstm:
            p = mp.Process(
                target=mgrid_cnn_lstm_setting['train_func'],
                kwargs={'learning_setting': mgrid_cnn_lstm_setting})
            p.start()
            processes.append(p)

    print("Training started with selected  " +
          "(each architecture experiment will run with " +
          str(number_of_parallel_experiments) + " parallel run/runs).")

    for p in tqdm(processes):
        p.join()
