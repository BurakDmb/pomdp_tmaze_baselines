import sys
import torch.multiprocessing as mp
from tqdm.auto import tqdm


if __name__ == '__main__':
    # from test_all import unittest_main
    # unittest_main()

    params = len(sys.argv)
    if params == 2 and sys.argv[1] == 'multigpu':

        from pomdp_tmaze_baselines.params.arch_comp_mgpu import (
            number_of_parallel_experiments,
            start_no_memory, start_no_memory_intr, start_o_k_memory,
            start_o_k_intr_memory, start_oa_k_memory, start_oa_k_intr_memory,
            start_lstm, start_lstm_intr, no_memory_learning_setting,
            no_memory_intr_learning_setting, o_k_memory_learning_setting,
            o_k_intr_memory_learning_setting, oa_k_memory_learning_setting,
            oa_k_intr_memory_learning_setting, lstm_learning_setting,
            lstm_intr_learning_setting
        )
    else:
        from pomdp_tmaze_baselines.params.arch_comp_gpu import (
            number_of_parallel_experiments,
            start_no_memory, start_no_memory_intr, start_o_k_memory,
            start_o_k_intr_memory, start_oa_k_memory, start_oa_k_intr_memory,
            start_lstm, start_lstm_intr, no_memory_learning_setting,
            no_memory_intr_learning_setting, o_k_memory_learning_setting,
            o_k_intr_memory_learning_setting, oa_k_memory_learning_setting,
            oa_k_intr_memory_learning_setting, lstm_learning_setting,
            lstm_intr_learning_setting
        )

    mp.set_start_method('spawn')
    processes = []

    for _ in range(number_of_parallel_experiments):

        if start_no_memory:
            p = mp.Process(
                target=no_memory_learning_setting['train_func'],
                kwargs={'learning_setting': no_memory_learning_setting})
            p.start()
            processes.append(p)

        if start_no_memory_intr:
            p = mp.Process(
                target=no_memory_intr_learning_setting['train_func'],
                kwargs={'learning_setting': no_memory_intr_learning_setting})
            p.start()
            processes.append(p)

        if start_o_k_memory:
            p = mp.Process(
                target=o_k_memory_learning_setting['train_func'],
                kwargs={'learning_setting': o_k_memory_learning_setting})
            p.start()
            processes.append(p)

        if start_o_k_intr_memory:
            p = mp.Process(
                target=o_k_intr_memory_learning_setting['train_func'],
                kwargs={'learning_setting': o_k_intr_memory_learning_setting})
            p.start()
            processes.append(p)

        if start_oa_k_memory:
            p = mp.Process(
                target=o_k_memory_learning_setting['train_func'],
                kwargs={'learning_setting': oa_k_memory_learning_setting})
            p.start()
            processes.append(p)

        if start_oa_k_intr_memory:
            p = mp.Process(
                target=o_k_intr_memory_learning_setting['train_func'],
                kwargs={'learning_setting': oa_k_intr_memory_learning_setting})
            p.start()
            processes.append(p)

        if start_lstm:
            p = mp.Process(
                target=lstm_learning_setting['train_func'],
                kwargs={'learning_setting': lstm_learning_setting})
            p.start()
            processes.append(p)

        if start_lstm_intr:
            p = mp.Process(
                target=lstm_intr_learning_setting['train_func'],
                kwargs={'learning_setting': lstm_intr_learning_setting})
            p.start()
            processes.append(p)

    print("Training started with selected architectures " +
          "(each architecture experiment will run with " +
          str(number_of_parallel_experiments) + " parallel run/runs).")

    for p in tqdm(processes):
        p.join()
