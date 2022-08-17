import unittest
import numpy as np


class TestCode(unittest.TestCase):
    # TODO: Implement TMaze Env unittest.

    def test_vec_Minigrid(self):
        import pomdp_tmaze_baselines.EnvMinigrid as EnvMinigrid
        from pomdp_tmaze_baselines.utils.UtilStableAgents import\
            train_ppo_agent
        from pomdp_tmaze_baselines.utils.UtilPolicies import MlpACPolicy
        from stable_baselines3.common.vec_env import SubprocVecEnv
        from stable_baselines3.common.env_util import make_vec_env

        learning_setting = {}
        learning_setting['envClass'] = EnvMinigrid.MinigridEnv
        learning_setting['learning_rate'] = 1e-3
        learning_setting['discount_rate'] = 0.99
        learning_setting['nn_num_layers'] = 3
        learning_setting['nn_layer_size'] = 8
        learning_setting['n_steps'] = 32
        learning_setting['batch_size'] = 32
        learning_setting['memory_type'] = 0
        learning_setting['memory_length'] = 1
        learning_setting['intrinsic_enabled'] = True
        learning_setting['intrinsic_beta'] = 0.1
        learning_setting['ae_enabled'] = True
        learning_setting['ae_path'] = "models/ae.torch"
        learning_setting['ae_rcons_err_type'] = "MSE"
        learning_setting['eval_enabled'] = False
        learning_setting['eval_freq'] = 1000
        learning_setting['eval_episodes'] = 0
        learning_setting['eval_path'] = None
        learning_setting['env_n_proc'] = 1
        learning_setting['vec_env_cls'] = SubprocVecEnv
        learning_setting['tb_log_name'] = "o_k"
        learning_setting['tb_log_dir'] = None
        learning_setting['maze_length'] = 10
        learning_setting['total_timesteps'] = 50
        learning_setting['seed'] = None
        learning_setting['policy'] = MlpACPolicy
        learning_setting['save'] = False
        learning_setting['device'] = 'cpu'
        learning_setting['train_func'] = train_ppo_agent

        n_procs = 4
        train_env = make_vec_env(
            env_id=EnvMinigrid.MinigridEnv,
            n_envs=n_procs,
            vec_env_cls=SubprocVecEnv,
            vec_env_kwargs=dict(start_method='forkserver'),
            env_kwargs=dict(**learning_setting))

        obs = train_env.reset()
        self.assertTrue(obs.shape == (n_procs, EnvMinigrid.latent_dims))

        n_procs = 1

        env_n_proc = learning_setting.get('env_n_proc', 1)
        if env_n_proc > 1:
            train_env = make_vec_env(
                env_id=EnvMinigrid.MinigridEnv,
                n_envs=env_n_proc,
                vec_env_cls=SubprocVecEnv,
                vec_env_kwargs=dict(start_method='forkserver'),
                env_kwargs=dict(**learning_setting))
        else:
            train_env = EnvMinigrid.MinigridEnv(**learning_setting)

        obs = train_env.reset()
        self.assertTrue(obs.shape == (EnvMinigrid.latent_dims,))

    def test_memory_types_Minigrid(self):
        import pomdp_tmaze_baselines.EnvMinigrid as EnvMinigrid
        from pomdp_tmaze_baselines.utils.UtilStableAgents import\
            train_ppo_agent
        from pomdp_tmaze_baselines.utils.UtilPolicies import MlpACPolicy
        from stable_baselines3.common.vec_env import DummyVecEnv

        learning_setting = {}
        learning_setting['envClass'] = EnvMinigrid.MinigridEnv
        learning_setting['learning_rate'] = 1e-3
        learning_setting['discount_rate'] = 0.99
        learning_setting['nn_num_layers'] = 3
        learning_setting['nn_layer_size'] = 8
        learning_setting['n_steps'] = 32
        learning_setting['batch_size'] = 32
        learning_setting['memory_type'] = 0
        learning_setting['memory_length'] = 1
        learning_setting['intrinsic_enabled'] = True
        learning_setting['intrinsic_beta'] = 0.1
        learning_setting['ae_enabled'] = True
        learning_setting['ae_path'] = "models/ae.torch"
        learning_setting['ae_rcons_err_type'] = "MSE"
        learning_setting['eval_enabled'] = False
        learning_setting['eval_freq'] = 1000
        learning_setting['eval_episodes'] = 0
        learning_setting['env_n_proc'] = 1
        learning_setting['eval_path'] = None
        learning_setting['vec_env_cls'] = DummyVecEnv
        learning_setting['tb_log_name'] = "o_k"
        learning_setting['tb_log_dir'] = None
        learning_setting['maze_length'] = 10
        learning_setting['total_timesteps'] = 50
        learning_setting['seed'] = None
        learning_setting['policy'] = MlpACPolicy
        learning_setting['save'] = False
        learning_setting['device'] = 'cpu'
        learning_setting['train_func'] = train_ppo_agent

        # Testing observation and action space sizes with
        # different memory lengths
        for mem_len in [1, 3, 5]:
            learning_setting['memory_length'] = mem_len

            learning_setting['memory_type'] = 0
            env_no_mem = EnvMinigrid.MinigridEnv(**learning_setting)

            learning_setting['memory_type'] = 1
            env_lastk_mem = EnvMinigrid.MinigridEnv(**learning_setting)

            learning_setting['memory_type'] = 2
            env_bin_mem = EnvMinigrid.MinigridEnv(**learning_setting)

            learning_setting['memory_type'] = 3
            env_ok_mem = EnvMinigrid.MinigridEnv(**learning_setting)

            learning_setting['memory_type'] = 4
            env_oak_mem = EnvMinigrid.MinigridEnv(**learning_setting)

            learning_setting['memory_type'] = 5
            env_lstm_mem = EnvMinigrid.MinigridEnv(**learning_setting)

            # Checking memory type 0 = None
            self.assertTrue(
                env_no_mem.observation_space.shape == (
                    EnvMinigrid.latent_dims, ))
            self.assertTrue(
                env_no_mem.action_space.n == EnvMinigrid.action_dim)

            # Checking memory type 1 = Kk
            self.assertTrue(
                env_lastk_mem.observation_space.shape == (
                    EnvMinigrid.latent_dims +
                    EnvMinigrid.latent_dims *
                    learning_setting['memory_length'],))
            self.assertTrue(
                env_lastk_mem.action_space.n == EnvMinigrid.action_dim * 1)

            # Checking memory type 2 = Bk
            self.assertTrue(
                env_bin_mem.observation_space.shape == (
                    EnvMinigrid.latent_dims +
                    learning_setting['memory_length'], ))
            self.assertTrue(
                tuple(env_bin_mem.action_space.nvec) == (
                    EnvMinigrid.action_dim,
                    2**learning_setting['memory_length']))

            # Checking memory type 3 = Ok
            self.assertTrue(
                env_ok_mem.observation_space.shape == (
                    EnvMinigrid.latent_dims +
                    EnvMinigrid.latent_dims *
                    learning_setting['memory_length'],))
            self.assertTrue(
                tuple(env_ok_mem.action_space.nvec) == (
                    EnvMinigrid.action_dim, 2))

            # Checking memory type 4 = OAk
            self.assertTrue(
                env_oak_mem.observation_space.shape == (
                    EnvMinigrid.latent_dims +
                    (EnvMinigrid.latent_dims + 1) *
                    learning_setting['memory_length'], ))
            self.assertTrue(
                tuple(env_oak_mem.action_space.nvec) == (
                    EnvMinigrid.action_dim, 2))

            # Checking memory type 5 = None (For LSTM)
            self.assertTrue(
                env_lstm_mem.observation_space.shape == (
                    EnvMinigrid.latent_dims, ))
            self.assertTrue(
                env_lstm_mem.action_space.n == EnvMinigrid.action_dim)

        # Testing memory actions.
        learning_setting['memory_length'] = 4

        # No memory
        # After turning 4 times, obs4 needs to be equal to the obs1.
        learning_setting['memory_type'] = 0
        env_no_mem = EnvMinigrid.MinigridEnv(**learning_setting)
        obs0 = env_no_mem.reset(seed=0)
        obs1, reward, done, _ = env_no_mem.step(0)
        obs2, reward, done, _ = env_no_mem.step(0)
        obs3, reward, done, _ = env_no_mem.step(0)
        obs4, reward, done, _ = env_no_mem.step(0)
        self.assertTrue((obs0 == obs4).all())

        # Lastk Memory
        learning_setting['memory_type'] = 1
        env_lastk_mem = EnvMinigrid.MinigridEnv(**learning_setting)
        obs0 = env_lastk_mem.reset(seed=0)
        obs1, reward, done, _ = env_lastk_mem.step(0)
        obs2, reward, done, _ = env_lastk_mem.step(0)
        obs3, reward, done, _ = env_lastk_mem.step(0)
        obs4, reward, done, _ = env_lastk_mem.step(0)
        self.assertTrue((obs4[128*4:128*5] == obs0[0:128]).all())

        # Bk Memory
        # Checking for storing different binary values in the external memory.
        learning_setting['memory_type'] = 2
        env_bin_mem = EnvMinigrid.MinigridEnv(**learning_setting)
        obs0 = env_bin_mem.reset(seed=0)
        obs1, reward, done, _ = env_bin_mem.step((0, 10))
        valList = list(map(int, obs1[128:]))
        self.assertTrue(int(''.join(str(i) for i in valList), 2) == 10)

        obs2, reward, done, _ = env_bin_mem.step((0, 1))
        valList = list(map(int, obs2[128:]))
        self.assertTrue(int(''.join(str(i) for i in valList), 2) == 1)

        obs3, reward, done, _ = env_bin_mem.step((0, 5))
        valList = list(map(int, obs3[128:]))
        self.assertTrue(int(''.join(str(i) for i in valList), 2) == 5)

        obs4, reward, done, _ = env_bin_mem.step((0, 13))
        valList = list(map(int, obs4[128:]))
        self.assertTrue(int(''.join(str(i) for i in valList), 2) == 13)

        # Ok Memory
        # After turning 4 times, 5th action will save current observation,
        # which is the initial observation.
        learning_setting['memory_type'] = 3
        env_ok_mem = EnvMinigrid.MinigridEnv(**learning_setting)
        obs0 = env_ok_mem.reset(seed=0)
        obs1, reward, done, _ = env_ok_mem.step((0, 0))
        obs2, reward, done, _ = env_ok_mem.step((0, 0))
        obs3, reward, done, _ = env_ok_mem.step((0, 0))
        obs4, reward, done, _ = env_ok_mem.step((0, 0))
        obs5, reward, done, _ = env_ok_mem.step((0, 1))
        self.assertTrue((obs5[128:256] == obs0[0:128]).all())

        # OAk Memory
        # After turning 4 times, 5th action will save current
        # observation+action, which is the initial observation and
        # the movement action 1.
        learning_setting['memory_type'] = 4
        env_oak_mem = EnvMinigrid.MinigridEnv(**learning_setting)
        obs0 = env_oak_mem.reset(seed=0)
        obs1, reward, done, _ = env_oak_mem.step((1, 0))
        obs2, reward, done, _ = env_oak_mem.step((1, 0))
        obs3, reward, done, _ = env_oak_mem.step((1, 0))
        obs4, reward, done, _ = env_oak_mem.step((1, 0))
        obs5, reward, done, _ = env_oak_mem.step((1, 1))
        self.assertTrue((obs5[128:257] == np.append(obs0[0:128], [1])).all())

        # LSTM memory.
        # After turning 4 times, obs4 needs to be equal to the obs1.
        learning_setting['memory_type'] = 5
        env_lstm_mem = EnvMinigrid.MinigridEnv(**learning_setting)
        obs0 = env_lstm_mem.reset(seed=0)
        obs1, reward, done, _ = env_lstm_mem.step(0)
        obs2, reward, done, _ = env_lstm_mem.step(0)
        obs3, reward, done, _ = env_lstm_mem.step(0)
        obs4, reward, done, _ = env_lstm_mem.step(0)
        self.assertTrue((obs0 == obs4).all())

        # Testing observation dimensions after resetting the environment.
        obs_no_mem = env_no_mem.reset(seed=0)
        self.assertTrue(
            obs_no_mem.shape == env_no_mem.observation_space.shape)

        obs_lastk_mem = env_lastk_mem.reset(seed=0)
        self.assertTrue(
            obs_lastk_mem.shape == env_lastk_mem.observation_space.shape)

        obs_bin_mem = env_bin_mem.reset(seed=0)
        self.assertTrue(
            obs_bin_mem.shape == env_bin_mem.observation_space.shape)

        obs_ok_mem = env_ok_mem.reset(seed=0)
        self.assertTrue(
            obs_ok_mem.shape == env_ok_mem.observation_space.shape)

        obs_oak_mem = env_oak_mem.reset(seed=0)
        self.assertTrue(
            obs_oak_mem.shape == env_oak_mem.observation_space.shape)

        obs_lstm_mem = env_lstm_mem.reset(seed=0)
        self.assertTrue(
            obs_lstm_mem.shape == env_lstm_mem.observation_space.shape)

    def test_env_Minigrid(self):

        from pomdp_tmaze_baselines.EnvMinigrid import MinigridEnv
        from pomdp_tmaze_baselines.utils.UtilStableAgents import\
            train_ppo_agent
        from pomdp_tmaze_baselines.utils.UtilPolicies import MlpACPolicy
        from stable_baselines3.common.vec_env import DummyVecEnv

        learning_setting = {}
        learning_setting['envClass'] = MinigridEnv
        learning_setting['learning_rate'] = 1e-3
        learning_setting['discount_rate'] = 0.99
        learning_setting['nn_num_layers'] = 3
        learning_setting['nn_layer_size'] = 8
        learning_setting['n_steps'] = 32
        learning_setting['batch_size'] = 32
        learning_setting['memory_type'] = 3
        learning_setting['memory_length'] = 1
        learning_setting['intrinsic_enabled'] = True
        learning_setting['intrinsic_beta'] = 0.1
        learning_setting['ae_enabled'] = True
        learning_setting['ae_path'] = "models/ae.torch"
        learning_setting['ae_rcons_err_type'] = "MSE"
        learning_setting['eval_enabled'] = False
        learning_setting['eval_freq'] = 1000
        learning_setting['eval_episodes'] = 0
        learning_setting['eval_path'] = None
        learning_setting['env_n_proc'] = 1
        learning_setting['vec_env_cls'] = DummyVecEnv
        learning_setting['tb_log_name'] = "o_k"
        learning_setting['tb_log_dir'] = None
        learning_setting['maze_length'] = 10
        learning_setting['total_timesteps'] = 50
        learning_setting['seed'] = None
        learning_setting['policy'] = MlpACPolicy
        learning_setting['save'] = False
        learning_setting['device'] = 'cpu'
        learning_setting['train_func'] = train_ppo_agent

        # train_ppo_agent(learning_setting=learning_setting)

        env = MinigridEnv(**learning_setting)
        env.reset()
        for i in range(10):
            obs, reward, done, _ = env.step((0, 0))
            pass

    def test_env_Minigrid_check_latent(self):

        from pomdp_tmaze_baselines.EnvMinigrid import MinigridEnv
        from pomdp_tmaze_baselines.utils.UtilStableAgents import\
            train_ppo_agent
        from pomdp_tmaze_baselines.utils.UtilPolicies import CNNACPolicy
        from stable_baselines3.common.vec_env import DummyVecEnv

        learning_setting = {}
        learning_setting['envClass'] = MinigridEnv
        learning_setting['learning_rate'] = 1e-3
        learning_setting['discount_rate'] = 0.99
        learning_setting['nn_num_layers'] = 4
        learning_setting['nn_layer_size'] = 4
        learning_setting['n_steps'] = 256
        learning_setting['batch_size'] = 256
        learning_setting['memory_type'] = 0
        learning_setting['memory_length'] = 1
        learning_setting['intrinsic_enabled'] = False
        learning_setting['intrinsic_beta'] = 0.01
        learning_setting['ae_enabled'] = False
        learning_setting['ae_path'] = "models/ae.torch"
        learning_setting['ae_rcons_err_type'] = "MSE"
        learning_setting['eval_enabled'] = False
        learning_setting['eval_freq'] = 1000
        learning_setting['eval_episodes'] = 0
        learning_setting['eval_path'] = None
        learning_setting['env_n_proc'] = 1
        learning_setting['vec_env_cls'] = DummyVecEnv
        learning_setting['tb_log_name'] = "ae_ppo"
        learning_setting['tb_log_dir'] = "./logs/c_minigrid_tb/"
        learning_setting['maze_length'] = 10
        learning_setting['total_timesteps'] = 100
        learning_setting['seed'] = None
        learning_setting['policy'] = CNNACPolicy
        learning_setting['save'] = False
        learning_setting['device'] = 'cuda:0'
        learning_setting['train_func'] = train_ppo_agent

        # train_ppo_agent(learning_setting=learning_setting)

        env = MinigridEnv(**learning_setting)
        env.reset()
        for i in range(10):
            obs, reward, done, _ = env.step(0)
            pass

    def test_custom_agent(self):
        from utils.UtilStableAgents import train_q_agent
        from pomdp_tmaze_baselines.EnvTMaze import TMazeEnv

        learning_setting = {}
        learning_setting['envClass'] = TMazeEnv
        learning_setting['learning_rate'] = 0.1
        learning_setting['discount_rate'] = 0.99
        learning_setting['epsilon_start'] = 0.33
        learning_setting['epsilon_end'] = 0.33
        learning_setting['eval_enabled'] = False
        learning_setting['eval_freq'] = 1000
        learning_setting['eval_episodes'] = 0
        learning_setting['eval_path'] = None
        learning_setting['tb_log_name'] = "q-tmazev0"
        learning_setting['tb_log_dir'] = None
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
        from pomdp_tmaze_baselines.EnvTMaze import TMazeEnv, TMazeEnvPOMDP,\
            TMazeEnvMemoryWrapped

        env = TMazeEnv(maze_length=6)
        env1 = TMazeEnvPOMDP(maze_length=6)
        env9 = TMazeEnvMemoryWrapped(maze_length=6)

        self.assertIsNotNone(env.reset())
        self.assertIsNotNone(env1.reset())
        self.assertIsNotNone(env9.reset())

    def test_custom_ppo_policy(self):
        from pomdp_tmaze_baselines.EnvTMaze import TMazeEnv
        from pomdp_tmaze_baselines.utils.UtilStableAgents import\
            train_ppo_agent
        from pomdp_tmaze_baselines.utils.UtilPolicies import MlpACPolicy
        from stable_baselines3.common.vec_env import DummyVecEnv

        ppo_learning_setting = {}
        ppo_learning_setting['envClass'] = TMazeEnv
        ppo_learning_setting['learning_rate'] = 1e-3
        ppo_learning_setting['discount_rate'] = 0.99
        ppo_learning_setting['nn_num_layers'] = 4
        ppo_learning_setting['nn_layer_size'] = 512
        ppo_learning_setting['n_steps'] = 64
        ppo_learning_setting['batch_size'] = 64
        ppo_learning_setting['eval_enabled'] = False
        ppo_learning_setting['eval_freq'] = 1000
        ppo_learning_setting['eval_episodes'] = 0
        ppo_learning_setting['eval_path'] = None
        ppo_learning_setting['env_n_proc'] = 1
        ppo_learning_setting['vec_env_cls'] = DummyVecEnv
        ppo_learning_setting['tb_log_name'] = "ppo-tmazev0"
        ppo_learning_setting['tb_log_dir'] = None
        ppo_learning_setting['maze_length'] = 6
        ppo_learning_setting['total_timesteps'] = 50
        ppo_learning_setting['seed'] = None
        ppo_learning_setting['policy'] = MlpACPolicy
        ppo_learning_setting['save'] = False
        ppo_learning_setting['device'] = 'cpu'

        train_ppo_agent(learning_setting=ppo_learning_setting)

    def test_custom_dqn_policy(self):
        from pomdp_tmaze_baselines.EnvTMaze import TMazeEnv
        from pomdp_tmaze_baselines.utils.UtilStableAgents import\
            train_dqn_agent
        from pomdp_tmaze_baselines.utils.UtilPolicies import MlpDQNPolicy
        from stable_baselines3.common.vec_env import DummyVecEnv

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
        dqn_learning_setting['eval_enabled'] = False
        dqn_learning_setting['eval_freq'] = 1000
        dqn_learning_setting['eval_episodes'] = 0
        dqn_learning_setting['eval_path'] = None
        dqn_learning_setting['env_n_proc'] = 1
        dqn_learning_setting['vec_env_cls'] = DummyVecEnv
        dqn_learning_setting['tb_log_name'] = "dqn-tmazev0"
        dqn_learning_setting['tb_log_dir'] = None
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
        from pomdp_tmaze_baselines.EnvTMaze import TMazeEnv
        from pomdp_tmaze_baselines.utils.UtilStableAgents import\
            train_dqn_agent
        from pomdp_tmaze_baselines.utils.UtilPolicies import QLSTMPolicy
        from stable_baselines3.common.vec_env import DummyVecEnv

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
        dqn_learning_setting['eval_enabled'] = False
        dqn_learning_setting['eval_freq'] = 1000
        dqn_learning_setting['eval_episodes'] = 0
        dqn_learning_setting['eval_path'] = None
        dqn_learning_setting['env_n_proc'] = 1
        dqn_learning_setting['vec_env_cls'] = DummyVecEnv
        dqn_learning_setting['tb_log_name'] = "qlstm-tmazev0"
        dqn_learning_setting['tb_log_dir'] = None
        dqn_learning_setting['maze_length'] = 6
        dqn_learning_setting['total_timesteps'] = 100
        dqn_learning_setting['seed'] = None
        dqn_learning_setting['policy'] = QLSTMPolicy
        dqn_learning_setting['save'] = False
        dqn_learning_setting['device'] = 'cpu'

        train_dqn_agent(learning_setting=dqn_learning_setting)

    def test_lstm_ppo_policy(self):
        from pomdp_tmaze_baselines.EnvTMaze import TMazeEnv
        from pomdp_tmaze_baselines.utils.UtilStableAgents import\
            train_ppo_lstm_agent
        from stable_baselines3.common.vec_env import DummyVecEnv

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
        ppoLSTM_learning_setting['intrinsic_enabled'] = False
        ppoLSTM_learning_setting['intrinsic_beta'] = 0.1
        ppoLSTM_learning_setting['eval_enabled'] = False
        ppoLSTM_learning_setting['eval_freq'] = 1000
        ppoLSTM_learning_setting['eval_episodes'] = 0
        ppoLSTM_learning_setting['eval_path'] = None
        ppoLSTM_learning_setting['env_n_proc'] = 1
        ppoLSTM_learning_setting['vec_env_cls'] = DummyVecEnv
        ppoLSTM_learning_setting['tb_log_name'] = "ppolstm-tmazev0"
        ppoLSTM_learning_setting['tb_log_dir'] = None
        ppoLSTM_learning_setting['maze_length'] = 6
        ppoLSTM_learning_setting['total_timesteps'] = 50
        ppoLSTM_learning_setting['seed'] = None
        ppoLSTM_learning_setting['policy'] = "MlpLstmPolicy"
        ppoLSTM_learning_setting['save'] = False
        ppoLSTM_learning_setting['device'] = 'cpu'
        ppoLSTM_learning_setting['train_func'] = train_ppo_lstm_agent

        train_ppo_lstm_agent(learning_setting=ppoLSTM_learning_setting)

    def test_a2c_agent(self):
        from pomdp_tmaze_baselines.EnvTMaze import TMazeEnv
        from pomdp_tmaze_baselines.utils.UtilStableAgents import\
            train_a2c_agent
        from stable_baselines3.common.vec_env import DummyVecEnv

        a2c_learning_setting = {}
        a2c_learning_setting['envClass'] = TMazeEnv
        a2c_learning_setting['learning_rate'] = 1e-3
        a2c_learning_setting['discount_rate'] = 0.99
        a2c_learning_setting['nn_num_layers'] = 4
        a2c_learning_setting['nn_layer_size'] = 512
        a2c_learning_setting['n_steps'] = 256
        a2c_learning_setting['eval_enabled'] = False
        a2c_learning_setting['eval_freq'] = 1000
        a2c_learning_setting['eval_episodes'] = 0
        a2c_learning_setting['eval_path'] = None
        a2c_learning_setting['env_n_proc'] = 1
        a2c_learning_setting['vec_env_cls'] = DummyVecEnv
        a2c_learning_setting['tb_log_name'] = "a2c-tmazev0"
        a2c_learning_setting['tb_log_dir'] = None
        a2c_learning_setting['maze_length'] = 6
        a2c_learning_setting['total_timesteps'] = 50
        a2c_learning_setting['seed'] = None
        a2c_learning_setting['policy'] = "MlpPolicy"
        a2c_learning_setting['save'] = False
        a2c_learning_setting['device'] = 'cpu'

        train_a2c_agent(learning_setting=a2c_learning_setting)

    def test_env_v9(self):

        from pomdp_tmaze_baselines.EnvTMaze import TMazeEnvMemoryWrapped
        from pomdp_tmaze_baselines.utils.UtilStableAgents import\
            train_ppo_agent
        from pomdp_tmaze_baselines.utils.UtilPolicies import MlpACPolicy
        from stable_baselines3.common.vec_env import DummyVecEnv

        env_v9_learning_setting = {}
        env_v9_learning_setting['envClass'] = TMazeEnvMemoryWrapped
        env_v9_learning_setting['learning_rate'] = 1e-3
        env_v9_learning_setting['discount_rate'] = 0.99
        env_v9_learning_setting['nn_num_layers'] = 3
        env_v9_learning_setting['nn_layer_size'] = 8
        env_v9_learning_setting['n_steps'] = 32
        env_v9_learning_setting['batch_size'] = 32
        env_v9_learning_setting['memory_type'] = 3
        env_v9_learning_setting['memory_length'] = 1
        env_v9_learning_setting['intrinsic_enabled'] = False
        env_v9_learning_setting['intrinsic_beta'] = 0.1
        env_v9_learning_setting['eval_enabled'] = False
        env_v9_learning_setting['eval_freq'] = 1000
        env_v9_learning_setting['eval_episodes'] = 0
        env_v9_learning_setting['eval_path'] = None
        env_v9_learning_setting['env_n_proc'] = 1
        env_v9_learning_setting['vec_env_cls'] = DummyVecEnv
        env_v9_learning_setting['tb_log_name'] = "o_k"
        env_v9_learning_setting['tb_log_dir'] = None
        env_v9_learning_setting['maze_length'] = 10
        env_v9_learning_setting['total_timesteps'] = 50
        env_v9_learning_setting['seed'] = None
        env_v9_learning_setting['policy'] = MlpACPolicy
        env_v9_learning_setting['save'] = False
        env_v9_learning_setting['device'] = 'cpu'
        env_v9_learning_setting['train_func'] = train_ppo_agent

        train_ppo_agent(learning_setting=env_v9_learning_setting)


def unittest_main(exit=False):
    print("*** Running unit tests ***")
    unittest.main(__name__, exit=exit)


if __name__ == '__main__':
    unittest_main()
