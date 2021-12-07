import sys
import torch.multiprocessing as mp


if __name__ == '__main__':
    from test_all import unittest_main
    unittest_main()

    params = len(sys.argv)
    if params == 1 and sys.argv[0] == 'multigpu':
        pass
    else:
        from parameters_c_algorithms import number_of_parallel_experiments
        from parameters_c_algorithms import start_q, start_sarsa_low
        from parameters_c_algorithms import start_sarsa_middle
        from parameters_c_algorithms import start_sarsa_high, start_dqn
        from parameters_c_algorithms import start_qlstm, start_ppo
        from parameters_c_algorithms import start_ppoLSTM, start_a2c
        from parameters_c_algorithms import q_learning_setting
        from parameters_c_algorithms import sarsa_low_l_learning_setting
        from parameters_c_algorithms import sarsa_middle_l_learning_setting
        from parameters_c_algorithms import sarsa_high_l_learning_setting
        from parameters_c_algorithms import dqn_learning_setting
        from parameters_c_algorithms import qlstm_learning_setting
        from parameters_c_algorithms import ppo_learning_setting
        from parameters_c_algorithms import ppoLSTM_learning_setting
        from parameters_c_algorithms import a2c_learning_setting

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
