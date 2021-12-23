import unittest


class TestCode(unittest.TestCase):

    def test_custom_agent(self):
        from UtilStableAgents import train_q_agent
        from EnvTMaze import TMazeEnv

        learning_setting = {}
        learning_setting['envClass'] = TMazeEnv
        learning_setting['learning_rate'] = 0.1
        learning_setting['discount_rate'] = 0.99
        learning_setting['epsilon_start'] = 0.33
        learning_setting['epsilon_end'] = 0.33
        learning_setting['tb_log_name'] = "q-tmazev0"
        learning_setting['tb_log_dir'] = "./logs/test_t_maze_tensorboard/"
        learning_setting['maze_length'] = 6
        learning_setting['total_timesteps'] = 50
        learning_setting['seed'] = None
        learning_setting['save'] = False

        model = train_q_agent(learning_setting)
        obs = model.env.reset()
        action = model.pre_action(obs)

        self.assertIsNotNone(action)
        self.assertEqual(model.learning_rate,
                         learning_setting['learning_rate'])
        self.assertEqual(model.env.grid_size[0],
                         learning_setting['maze_length'])

    def test_custom_env(self):
        from EnvTMaze import TMazeEnv, TMazeEnvV1, TMazeEnvV2, TMazeEnvV3
        from EnvTMaze import TMazeEnvV4, TMazeEnvV5, TMazeEnvV6

        env = TMazeEnv(maze_length=6)
        env1 = TMazeEnvV1(maze_length=6)
        env2 = TMazeEnvV2(maze_length=6)
        env3 = TMazeEnvV3(maze_length=6)
        env4 = TMazeEnvV4(maze_length=6)
        env5 = TMazeEnvV5(maze_length=6)
        env6 = TMazeEnvV6(maze_length=6)

        self.assertIsNotNone(env.reset())
        self.assertIsNotNone(env1.reset())
        self.assertIsNotNone(env2.reset())
        self.assertIsNotNone(env3.reset())
        self.assertIsNotNone(env4.reset())
        self.assertIsNotNone(env5.reset())
        self.assertIsNotNone(env6.reset())

    def test_custom_ppo_policy(self):
        from EnvTMaze import TMazeEnv
        from UtilStableAgents import train_ppo_agent
        from UtilPolicies import MlpACPolicy

        ppo_learning_setting = {}
        ppo_learning_setting['envClass'] = TMazeEnv
        ppo_learning_setting['learning_rate'] = 1e-3
        ppo_learning_setting['discount_rate'] = 0.99
        ppo_learning_setting['nn_num_layers'] = 4
        ppo_learning_setting['nn_layer_size'] = 512
        ppo_learning_setting['n_steps'] = 64
        ppo_learning_setting['batch_size'] = 64
        ppo_learning_setting['tb_log_name'] = "ppo-tmazev0"
        ppo_learning_setting['tb_log_dir'] = "./logs/test_t_maze_tensorboard/"
        ppo_learning_setting['maze_length'] = 6
        ppo_learning_setting['total_timesteps'] = 50
        ppo_learning_setting['seed'] = None
        ppo_learning_setting['policy'] = MlpACPolicy
        ppo_learning_setting['save'] = False
        ppo_learning_setting['device'] = 'cpu'

        train_ppo_agent(learning_setting=ppo_learning_setting)

    def test_custom_dqn_policy(self):
        from EnvTMaze import TMazeEnv
        from UtilStableAgents import train_dqn_agent
        from UtilPolicies import MlpDQNPolicy

        dqn_learning_setting = {}
        dqn_learning_setting['envClass'] = TMazeEnv
        dqn_learning_setting['learning_rate'] = 1e-3
        dqn_learning_setting['discount_rate'] = 0.99
        dqn_learning_setting['epsilon_start'] = 0.9
        dqn_learning_setting['epsilon_end'] = 0.01
        dqn_learning_setting['exploration_fraction'] = 0.5
        dqn_learning_setting['update_interval'] = 100
        dqn_learning_setting['buffer_size'] = 1000000
        dqn_learning_setting['learning_starts'] = 50
        dqn_learning_setting['nn_num_layers'] = 4
        dqn_learning_setting['nn_layer_size'] = 512
        dqn_learning_setting['tb_log_name'] = "dqn-tmazev0"
        dqn_learning_setting['tb_log_dir'] = "./logs/test_t_maze_tensorboard/"
        dqn_learning_setting['maze_length'] = 6
        dqn_learning_setting['total_timesteps'] = 100
        dqn_learning_setting['seed'] = None
        dqn_learning_setting['policy'] = "MlpPolicy"
        dqn_learning_setting['save'] = False
        dqn_learning_setting['device'] = 'cpu'

        train_dqn_agent(learning_setting=dqn_learning_setting)

        dqn_learning_setting['policy'] = MlpDQNPolicy
        train_dqn_agent(learning_setting=dqn_learning_setting)

    def test_qlstm_policy(self):
        from EnvTMaze import TMazeEnv
        from UtilStableAgents import train_dqn_agent
        from UtilPolicies import QLSTMPolicy

        dqn_learning_setting = {}
        dqn_learning_setting['envClass'] = TMazeEnv
        dqn_learning_setting['learning_rate'] = 1e-3
        dqn_learning_setting['discount_rate'] = 0.99
        dqn_learning_setting['epsilon_start'] = 0.9
        dqn_learning_setting['epsilon_end'] = 0.01
        dqn_learning_setting['exploration_fraction'] = 0.5
        dqn_learning_setting['update_interval'] = 100
        dqn_learning_setting['learning_starts'] = 50
        dqn_learning_setting['buffer_size'] = 1000000
        dqn_learning_setting['nn_num_layers'] = 4
        dqn_learning_setting['nn_layer_size'] = 512
        dqn_learning_setting['tb_log_name'] = "qlstm-tmazev0"
        dqn_learning_setting['tb_log_dir'] = "./logs/test_t_maze_tensorboard/"
        dqn_learning_setting['maze_length'] = 6
        dqn_learning_setting['total_timesteps'] = 100
        dqn_learning_setting['seed'] = None
        dqn_learning_setting['policy'] = QLSTMPolicy
        dqn_learning_setting['save'] = False
        dqn_learning_setting['device'] = 'cpu'

        train_dqn_agent(learning_setting=dqn_learning_setting)

    def test_lstm_ppo_policy(self):
        from EnvTMaze import TMazeEnv
        from UtilStableAgents import train_ppo_lstm_agent

        ppoLSTM_learning_setting = {}
        ppoLSTM_learning_setting['envClass'] = TMazeEnv
        ppoLSTM_learning_setting['learning_rate'] = 1e-3
        ppoLSTM_learning_setting['discount_rate'] = 0.99
        ppoLSTM_learning_setting['nn_num_layers'] = 4
        ppoLSTM_learning_setting['nn_layer_size'] = 512
        ppoLSTM_learning_setting['n_steps'] = 64
        ppoLSTM_learning_setting['batch_size'] = 64
        ppoLSTM_learning_setting['memory_type'] = 0
        ppoLSTM_learning_setting['memory_length'] = 1
        ppoLSTM_learning_setting['intrinsic_enabled'] = 0
        ppoLSTM_learning_setting['intrinsic_beta'] = 0.5
        ppoLSTM_learning_setting['tb_log_name'] = "ppolstm-tmazev0"
        ppoLSTM_learning_setting['tb_log_dir'] = \
            "./logs/test_t_maze_tensorboard/"
        ppoLSTM_learning_setting['maze_length'] = 6
        ppoLSTM_learning_setting['total_timesteps'] = 50
        ppoLSTM_learning_setting['seed'] = None
        ppoLSTM_learning_setting['policy'] = "MlpLstmPolicy"
        ppoLSTM_learning_setting['save'] = False
        ppoLSTM_learning_setting['device'] = 'cpu'
        ppoLSTM_learning_setting['train_func'] = train_ppo_lstm_agent

        train_ppo_lstm_agent(learning_setting=ppoLSTM_learning_setting)

    def test_a2c_agent(self):
        from EnvTMaze import TMazeEnv
        from UtilStableAgents import train_a2c_agent

        a2c_learning_setting = {}
        a2c_learning_setting['envClass'] = TMazeEnv
        a2c_learning_setting['learning_rate'] = 1e-3
        a2c_learning_setting['discount_rate'] = 0.99
        a2c_learning_setting['nn_num_layers'] = 4
        a2c_learning_setting['nn_layer_size'] = 512
        a2c_learning_setting['n_steps'] = 256
        a2c_learning_setting['tb_log_name'] = "a2c-tmazev0"
        a2c_learning_setting['tb_log_dir'] = \
            "./logs/test_t_maze_tensorboard/"
        a2c_learning_setting['maze_length'] = 6
        a2c_learning_setting['total_timesteps'] = 50
        a2c_learning_setting['seed'] = None
        a2c_learning_setting['policy'] = "MlpPolicy"
        a2c_learning_setting['save'] = False
        a2c_learning_setting['device'] = 'cpu'

        train_a2c_agent(learning_setting=a2c_learning_setting)

    def test_fixed_size_obs_seq_env(self):
        from EnvTMaze import TMazeEnvV7

        env = TMazeEnvV7(maze_length=6, memory_seq_length=3)
        self.assertIsNotNone(env.reset())
        env.step(4)
        self.assertEqual(1, env.nextMemoryIndex)
        env.step(4)
        self.assertEqual(2, env.nextMemoryIndex)
        env.step(4)
        self.assertEqual(0, env.nextMemoryIndex)


def unittest_main(exit=False):
    print("*** Running unit tests ***")
    unittest.main(__name__, exit=exit)


if __name__ == '__main__':
    unittest.main()
