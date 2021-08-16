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
        learning_setting['tb_log_dir'] = "./logs/t_maze_tensorboard/"
        learning_setting['maze_length'] = 6
        learning_setting['total_timesteps'] = 50
        learning_setting['seed'] = None

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


if __name__ == '__main__':
    unittest.main()
