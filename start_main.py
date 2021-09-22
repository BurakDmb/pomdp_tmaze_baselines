import sys
import torch.multiprocessing as mp
from UtilStableAgents import train_ppo_agent, train_q_agent
from UtilStableAgents import train_dqn_agent, train_sarsa_lambda_agent
from UtilStableAgents import train_a2c_agent

if __name__ == '__main__':
    params = len(sys.argv)
    if params == 1 and sys.argv[0] == 'multigpu':
        from parameters_multi_gpu import q_learning_setting
        from parameters_multi_gpu import sarsa_learning_setting
        from parameters_multi_gpu import dqn_learning_setting
        from parameters_multi_gpu import qlstm_learning_setting
        from parameters_multi_gpu import ppo_learning_setting
        from parameters_multi_gpu import ppoLSTM_learning_setting
        from parameters_multi_gpu import a2c_learning_setting
    else:
        from parameters import q_learning_setting
        from parameters import sarsa_learning_setting
        from parameters import dqn_learning_setting
        from parameters import qlstm_learning_setting
        from parameters import ppo_learning_setting
        from parameters import ppoLSTM_learning_setting
        from parameters import a2c_learning_setting

    mp.set_start_method('spawn')
    number_of_parallel_experiments = 4
    processes = []

    # Change the flags to True/False for only running specific agents
    start_q,\
        start_sarsa,\
        start_dqn,\
        start_qlstm,\
        start_ppo,\
        start_ppoLSTM,\
        start_a2c = True, True, True, True, True, True, True

    for rank in range(number_of_parallel_experiments):

        if start_q:
            p1 = mp.Process(target=train_q_agent,
                            kwargs={'learning_setting': q_learning_setting})
            p1.start()
            processes.append(p1)

        if start_sarsa:
            p2 = mp.Process(target=train_sarsa_lambda_agent,
                            kwargs={'learning_setting': sarsa_learning_setting}
                            )
            p2.start()
            processes.append(p2)

        if start_dqn:
            p3 = mp.Process(target=train_dqn_agent,
                            kwargs={'learning_setting': dqn_learning_setting})
            p3.start()
            processes.append(p3)

        if start_qlstm:
            p4 = mp.Process(target=train_dqn_agent,
                            kwargs={'learning_setting': qlstm_learning_setting}
                            )
            p4.start()
            processes.append(p4)

        if start_ppo:
            p5 = mp.Process(target=train_ppo_agent,
                            kwargs={'learning_setting': ppo_learning_setting})
            p5.start()
            processes.append(p5)

        if start_ppoLSTM:
            p6 = mp.Process(target=train_ppo_agent,
                            kwargs={'learning_setting':
                                    ppoLSTM_learning_setting})
            p6.start()
            processes.append(p6)

        if start_a2c:
            p7 = mp.Process(target=train_a2c_agent,
                            kwargs={'learning_setting':
                                    a2c_learning_setting})
            p7.start()
            processes.append(p7)

    for p in processes:
        p.join()
