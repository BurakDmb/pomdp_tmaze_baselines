import sys
import torch.multiprocessing as mp
from tqdm.auto import tqdm


if __name__ == '__main__':
    # from test_all import unittest_main
    # unittest_main()

    params = len(sys.argv)
    if params == 2 and sys.argv[1] == 'multigpu':
        pass
    else:
        from pomdp_tmaze_baselines.params.alg_comp_gpu import (
            number_of_parallel_experiments,
            start_q, start_sarsa_low, start_sarsa_middle, start_sarsa_high,
            start_dqn, start_qlstm, start_ppo, start_ppoLSTM, start_a2c,
            q_learning_setting, sarsa_low_l_learning_setting,
            sarsa_middle_l_learning_setting,
            sarsa_high_l_learning_setting, dqn_learning_setting,
            qlstm_learning_setting, ppo_learning_setting,
            ppoLSTM_learning_setting, a2c_learning_setting
        )
    mp.set_start_method('forkserver', force=True)
    processes = []

    for _ in range(number_of_parallel_experiments):

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

    print("Training started with selected algorithms " +
          "(each algorithm experiment will run with " +
          str(number_of_parallel_experiments) + " parallel run/runs).")

    for p in tqdm(processes):
        p.join()
