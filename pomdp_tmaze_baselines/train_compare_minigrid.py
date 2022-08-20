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
        #     start_ae_no_mem,
        #     start_ae_smm_lastk,
        #     start_ae_smm_bk,
        #     start_ae_smm_ok,
        #     start_ae_smm_ok_intr,
        #     start_ae_smm_oak,
        #     start_ae_smm_oak_intr,
        #     start_ae_lstm,
        #     start_cnn_no_mem,

        #     mgrid_ae_no_mem_setting, mgrid_ae_smm_lastk_setting,
        #     mgrid_ae_smm_bk_setting, mgrid_ae_smm_ok_setting,
        #     mgrid_ae_smm_ok_intr_setting, mgrid_ae_smm_oak_setting,
        #     mgrid_ae_smm_oak_intr_setting, mgrid_ae_lstm_setting,

        #     mgrid_cnn_no_mem_setting,
        # )
    else:
        from pomdp_tmaze_baselines.params.alg_minigrid import (
            number_of_parallel_experiments,
            start_ae_no_mem,
            start_ae_smm_lastk,
            start_ae_smm_bk,
            start_ae_smm_ok,
            start_ae_smm_ok_intr,
            start_ae_smm_oak,
            start_ae_smm_oak_intr,
            start_ae_lstm,
            start_cnn_no_mem,

            mgrid_ae_no_mem_setting, mgrid_ae_smm_lastk_setting,
            mgrid_ae_smm_bk_setting, mgrid_ae_smm_ok_setting,
            mgrid_ae_smm_ok_intr_setting, mgrid_ae_smm_oak_setting,
            mgrid_ae_smm_oak_intr_setting, mgrid_ae_lstm_setting,

            mgrid_cnn_no_mem_setting,
        )

    mp.set_start_method('forkserver', force=True)
    processes = []

    for _ in range(number_of_parallel_experiments):

        if start_ae_no_mem:
            p = mp.Process(
                target=mgrid_ae_no_mem_setting['train_func'],
                kwargs={'learning_setting': mgrid_ae_no_mem_setting})
            p.start()
            processes.append(p)

        if start_ae_smm_lastk:
            p = mp.Process(
                target=mgrid_ae_smm_lastk_setting['train_func'],
                kwargs={'learning_setting': mgrid_ae_smm_lastk_setting})
            p.start()
            processes.append(p)

        if start_ae_smm_bk:
            p = mp.Process(
                target=mgrid_ae_smm_bk_setting['train_func'],
                kwargs={'learning_setting': mgrid_ae_smm_bk_setting})
            p.start()
            processes.append(p)

        if start_ae_smm_ok:
            p = mp.Process(
                target=mgrid_ae_smm_ok_setting['train_func'],
                kwargs={'learning_setting': mgrid_ae_smm_ok_setting})
            p.start()
            processes.append(p)

        if start_ae_smm_ok_intr:
            p = mp.Process(
                target=mgrid_ae_smm_ok_intr_setting['train_func'],
                kwargs={'learning_setting': mgrid_ae_smm_ok_intr_setting})
            p.start()
            processes.append(p)

        if start_ae_smm_oak:
            p = mp.Process(
                target=mgrid_ae_smm_oak_setting['train_func'],
                kwargs={'learning_setting': mgrid_ae_smm_oak_setting})
            p.start()
            processes.append(p)

        if start_ae_smm_oak_intr:
            p = mp.Process(
                target=mgrid_ae_smm_oak_intr_setting['train_func'],
                kwargs={'learning_setting': mgrid_ae_smm_oak_intr_setting})
            p.start()
            processes.append(p)

        if start_ae_lstm:
            p = mp.Process(
                target=mgrid_ae_lstm_setting['train_func'],
                kwargs={'learning_setting': mgrid_ae_lstm_setting})
            p.start()
            processes.append(p)

        if start_cnn_no_mem:
            p = mp.Process(
                target=mgrid_cnn_no_mem_setting['train_func'],
                kwargs={'learning_setting': mgrid_cnn_no_mem_setting})
            p.start()
            processes.append(p)

    print("Training started with selected  " +
          "(each architecture experiment will run with " +
          str(number_of_parallel_experiments) + " parallel run/runs).")

    for p in tqdm(processes):
        p.join()
