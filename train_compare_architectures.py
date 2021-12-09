import sys
import torch.multiprocessing as mp

if __name__ == '__main__':
    # from test_all import unittest_main
    # unittest_main()

    params = len(sys.argv)
    if params == 1 and sys.argv[0] == 'multigpu':
        pass
    else:
        from parameters_c_architecture import number_of_parallel_experiments
        from parameters_c_architecture import start_no_memory
        from parameters_c_architecture import start_no_memory_intr
        from parameters_c_architecture import start_o_k_memory
        from parameters_c_architecture import start_o_k_intr_memory
        from parameters_c_architecture import start_oa_k_memory
        from parameters_c_architecture import start_oa_k_intr_memory
        from parameters_c_architecture import start_lstm
        from parameters_c_architecture import start_lstm_intr

        from parameters_c_architecture import no_memory_learning_setting
        from parameters_c_architecture import no_memory_intr_learning_setting
        from parameters_c_architecture import o_k_memory_learning_setting
        from parameters_c_architecture import o_k_intr_memory_learning_setting
        from parameters_c_architecture import oa_k_memory_learning_setting
        from parameters_c_architecture import oa_k_intr_memory_learning_setting
        from parameters_c_architecture import lstm_learning_setting
        from parameters_c_architecture import lstm_intr_learning_setting

    mp.set_start_method('spawn')
    processes = []

    for rank in range(number_of_parallel_experiments):

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

    for p in processes:
        p.join()
