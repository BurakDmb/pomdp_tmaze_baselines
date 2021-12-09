import sys
import torch.multiprocessing as mp


if __name__ == '__main__':
    from test_all import unittest_main
    unittest_main()

    params = len(sys.argv)
    if params == 2 and sys.argv[1] == 'multigpu':
        pass
    else:
        from params_gpu_comp_alg import number_of_parallel_experiments
        from params_gpu_comp_alg import start_q, start_sarsa_low
        from params_gpu_comp_alg import start_sarsa_middle
        from params_gpu_comp_alg import start_sarsa_high, start_dqn
        from params_gpu_comp_alg import start_qlstm, start_ppo
        from params_gpu_comp_alg import start_ppoLSTM, start_a2c
        from params_gpu_comp_alg import q_learning_setting
        from params_gpu_comp_alg import sarsa_low_l_learning_setting
        from params_gpu_comp_alg import sarsa_middle_l_learning_setting
        from params_gpu_comp_alg import sarsa_high_l_learning_setting
        from params_gpu_comp_alg import dqn_learning_setting
        from params_gpu_comp_alg import qlstm_learning_setting
        from params_gpu_comp_alg import ppo_learning_setting
        from params_gpu_comp_alg import ppoLSTM_learning_setting
        from params_gpu_comp_alg import a2c_learning_setting

    mp.set_start_method('spawn')
    processes = []

    for rank in range(number_of_parallel_experiments):

        if start_q:
            p = mp.Process(target=q_learning_setting['train_func'],
                           kwargs={'learning_setting': q_learning_setting})
            p.start()
            processes.append(p)

        if start_sarsa_low:
            p = mp.Process(target=sarsa_low_l_learning_setting['train_func'],
                           kwargs={'learning_setting':
                                   sarsa_low_l_learning_setting}
                           )
            p.start()
            processes.append(p)

        if start_sarsa_middle:
            p = mp.Process(
                target=sarsa_middle_l_learning_setting['train_func'],
                kwargs={'learning_setting':
                        sarsa_middle_l_learning_setting}
                )
            p.start()
            processes.append(p)

        if start_sarsa_high:
            p = mp.Process(
                target=sarsa_high_l_learning_setting['train_func'],
                kwargs={'learning_setting':
                        sarsa_high_l_learning_setting}
                )
            p.start()
            processes.append(p)

        if start_dqn:
            p = mp.Process(target=dqn_learning_setting['train_func'],
                           kwargs={'learning_setting': dqn_learning_setting})
            p.start()
            processes.append(p)

        if start_qlstm:
            p = mp.Process(target=qlstm_learning_setting['train_func'],
                           kwargs={'learning_setting': qlstm_learning_setting}
                           )
            p.start()
            processes.append(p)

        if start_ppo:
            p = mp.Process(target=ppo_learning_setting['train_func'],
                           kwargs={'learning_setting': ppo_learning_setting})
            p.start()
            processes.append(p)

        if start_ppoLSTM:
            p = mp.Process(target=ppoLSTM_learning_setting['train_func'],
                           kwargs={'learning_setting':
                                   ppoLSTM_learning_setting})
            p.start()
            processes.append(p)

        if start_a2c:
            p = mp.Process(target=a2c_learning_setting['train_func'],
                           kwargs={'learning_setting':
                                   a2c_learning_setting})
            p.start()
            processes.append(p)

    for p in processes:
        p.join()
