from stable_baselines3 import PPO, DQN, A2C
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import TensorBoardOutputFormat
# from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from agents.ClassQAgent import QAgent
from agents.ClassSarsaLambdaAgent import SarsaLambdaAgent
import datetime
import numpy as np
from stable_baselines3.common.evaluation import evaluate_policy
import os
import gym
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import compat_gym_seed


def train_q_agent(learning_setting):
    """
    Q-Learning Algorithm, Must-Have Learning Settings And
    Default Parameter Values:
        q_learning_setting = {}
        q_learning_setting['envClass'] = envClass
        q_learning_setting['learning_rate'] = 0.1
        q_learning_setting['discount_rate'] = 0.99
        q_learning_setting['epsilon_start'] = 0.33
        q_learning_setting['epsilon_end'] = 0.33
        q_learning_setting['tb_log_name'] = "q-tmazev0"
        q_learning_setting['tb_log_dir'] = "./logs/t_maze_tensorboard/"
        q_learning_setting['maze_length'] = maze_length
        q_learning_setting['total_timesteps'] = total_timesteps
        q_learning_setting['seed'] = None
        q_learning_setting['save'] = False
    """
    envClass = learning_setting['envClass']
    env = envClass(**learning_setting)

    model = QAgent(env, learning_setting=learning_setting)

    model.learn(total_timesteps=learning_setting['total_timesteps'],
                tb_log_name=learning_setting['tb_log_name'] +
                "-" + str(datetime.datetime.now()))

    # TODO: Implement saving q model
    if learning_setting['save']:
        pass

    return model


def train_sarsa_lambda_agent(learning_setting):
    """
    SARSA(Lambda) Algorithm, Must-Have Learning Settings And
    Default Parameter Values:
        sarsa_learning_setting = {}
        sarsa_learning_setting['envClass'] = envClass
        sarsa_learning_setting['learning_rate'] = 0.1
        sarsa_learning_setting['lambda_value'] = 0.9
        sarsa_learning_setting['discount_rate'] = 0.99
        sarsa_learning_setting['epsilon_start'] = 0.33
        sarsa_learning_setting['epsilon_end'] = 0.33
        sarsa_learning_setting['tb_log_name'] = "sarsa-l-tmazev0"
        sarsa_learning_setting['tb_log_dir'] = "./logs/t_maze_tensorboard/"
        sarsa_learning_setting['maze_length'] = maze_length
        sarsa_learning_setting['total_timesteps'] = total_timesteps
        sarsa_learning_setting['seed'] = None
        sarsa_learning_setting['save'] = False
    """
    envClass = learning_setting['envClass']
    env = envClass(**learning_setting)

    model = SarsaLambdaAgent(env, learning_setting=learning_setting)

    model.learn(total_timesteps=learning_setting['total_timesteps'],
                tb_log_name=learning_setting['tb_log_name'] +
                "-" + str(datetime.datetime.now()))

    # TODO: Implement saving sarsa model
    if learning_setting['save']:
        pass

    return model


def train_dqn_agent(learning_setting):
    """
    Deep Q-Learning Algorithm, Must-Have Learning Settings And
    Default Parameter Values:
        dqn_learning_setting = {}
        dqn_learning_setting['envClass'] = envClass
        dqn_learning_setting['learning_rate'] = 1e-5
        dqn_learning_setting['discount_rate'] = 0.99
        dqn_learning_setting['epsilon_start'] = 0.1
        dqn_learning_setting['epsilon_end'] = 0.1
        dqn_learning_setting['memory_type'] = memory_type
        dqn_learning_setting['memory_length'] = 3
        dqn_learning_setting['intrinsic_enabled'] = False
        dqn_learning_setting['intrinsic_beta'] = 0.01
        dqn_learning_setting['ae_enabled'] = True
        dqn_learning_setting['ae_comm_list'] = comm_list['dqn-tmazev0']
        dqn_learning_setting['ae_path'] = "models/ae.torch"
        dqn_learning_setting['ae_rcons_err_type'] = "MSE"
        dqn_learning_setting['eval_enabled'] = False
        dqn_learning_setting['eval_freq'] = 1000
        dqn_learning_setting['eval_episodes'] = 0
        dqn_learning_setting['eval_path'] = None
        dqn_learning_setting['exploration_fraction'] = 1.0
        dqn_learning_setting['update_interval'] = 128
        dqn_learning_setting['learning_starts'] = 512
        dqn_learning_setting['buffer_size'] = 100000
        dqn_learning_setting['nn_num_layers'] = 4
        dqn_learning_setting['nn_layer_size'] = 512
        dqn_learning_setting['env_n_proc'] = 1
        dqn_learning_setting['vec_env_cls'] = DummyVecEnv
        dqn_learning_setting['tb_log_name'] = "dqn-tmazev0"
        dqn_learning_setting['tb_log_dir'] = "./logs/t_maze_tensorboard_0/"
        dqn_learning_setting['maze_length'] = maze_length
        dqn_learning_setting['total_timesteps'] = total_timesteps
        dqn_learning_setting['seed'] = None
        dqn_learning_setting['policy'] = MlpDQNPolicy
        dqn_learning_setting['save'] = False
        dqn_learning_setting['device'] = 'cuda:0'
    """
    envClass = learning_setting['envClass']
    env_n_proc = learning_setting.get('env_n_proc', 1)
    vec_env_cls = learning_setting.get('vec_env_cls', DummyVecEnv)
    vec_env_kwargs = dict(start_method='forkserver') if isinstance(
        vec_env_cls, DummyVecEnv) else None
    if env_n_proc > 1:
        env = make_vec_env(
            env_id=envClass,
            n_envs=env_n_proc,
            vec_env_cls=vec_env_cls,
            vec_env_kwargs=vec_env_kwargs,
            env_kwargs=dict(**learning_setting))
        eval_env = make_vec_env(
                env_id=envClass,
                n_envs=1,
                vec_env_cls=vec_env_cls,
                vec_env_kwargs=vec_env_kwargs,
                env_kwargs=dict(**learning_setting)
                ) if learning_setting['eval_enabled'] else None
    else:
        env = envClass(**learning_setting)
        eval_env = envClass(
            **learning_setting
            ) if learning_setting['eval_enabled'] else None

    policy_kwargs = dict(net_arch=[learning_setting['nn_layer_size']] *
                         learning_setting['nn_num_layers'])

    model = DQN(learning_setting['policy'], env, verbose=0,
                tensorboard_log=learning_setting['tb_log_dir'],
                seed=learning_setting['seed'],
                policy_kwargs=policy_kwargs,
                learning_rate=learning_setting['learning_rate'],
                gamma=learning_setting['discount_rate'],
                exploration_initial_eps=learning_setting['epsilon_start'],
                exploration_final_eps=learning_setting['epsilon_end'],
                target_update_interval=learning_setting['update_interval'],
                exploration_fraction=learning_setting['exploration_fraction'],
                learning_starts=learning_setting['learning_starts'],
                buffer_size=learning_setting['buffer_size'],
                device=learning_setting['device'])

    model.learn(total_timesteps=learning_setting['total_timesteps'],
                tb_log_name=learning_setting['tb_log_name'] +
                "-" + str(datetime.datetime.now()),
                callback=TensorboardCallback(
                    learning_setting['eval_enabled'],
                    eval_env,
                    learning_setting['eval_freq'],
                    learning_setting['eval_episodes']),
                eval_env=eval_env,
                eval_freq=learning_setting['eval_freq'],
                n_eval_episodes=learning_setting['eval_episodes'],
                eval_log_path=learning_setting['eval_path'],
                )

    if learning_setting['save']:
        model.save("saves/" + learning_setting['tb_log_name'] +
                   "/" + str(datetime.datetime.now()))

    return model


def train_ppo_agent(learning_setting):
    """
    PPO(Proximal Policy Optimization Algorithm, Must-Have Learning Settings And
    Default Parameter Values:
        ppo_learning_setting = {}
        ppo_learning_setting['envClass'] = envClass
        ppo_learning_setting['learning_rate'] = 1e-5
        ppo_learning_setting['discount_rate'] = 0.99
        ppo_learning_setting['nn_num_layers'] = 4
        ppo_learning_setting['nn_layer_size'] = 512
        ppo_learning_setting['n_steps'] = 32
        ppo_learning_setting['batch_size'] = 32
        ppo_learning_setting['memory_type'] = memory_type
        ppo_learning_setting['memory_length'] = 3
        ppo_learning_setting['intrinsic_enabled'] = False
        ppo_learning_setting['intrinsic_beta'] = 0.01
        ppo_learning_setting['ae_enabled'] = True
        ppo_learning_setting['ae_comm_list'] = comm_list['ppo-tmazev0']
        ppo_learning_setting['ae_path'] = "models/ae.torch"
        ppo_learning_setting['ae_rcons_err_type'] = "MSE"
        ppo_learning_setting['eval_enabled'] = False
        ppo_learning_setting['eval_freq'] = 1000
        ppo_learning_setting['eval_episodes'] = 0
        ppo_learning_setting['eval_path'] = None
        ppo_learning_setting['env_n_proc'] = 1
        ppo_learning_setting['vec_env_cls'] = DummyVecEnv
        ppo_learning_setting['tb_log_name'] = "ppo-tmazev0"
        ppo_learning_setting['tb_log_dir'] = "./logs/t_maze_tensorboard_0/"
        ppo_learning_setting['maze_length'] = maze_length
        ppo_learning_setting['total_timesteps'] = total_timesteps
        ppo_learning_setting['seed'] = None
        ppo_learning_setting['policy'] = MlpACPolicy
        ppo_learning_setting['save'] = True
        ppo_learning_setting['device'] = 'cuda:0'
        ppo_learning_setting['train_func'] = train_ppo_agent
    """
    envClass = learning_setting['envClass']
    env_n_proc = learning_setting.get('env_n_proc', 1)
    vec_env_cls = learning_setting.get('vec_env_cls', DummyVecEnv)
    vec_env_kwargs = dict(start_method='forkserver') if isinstance(
        vec_env_cls, DummyVecEnv) else None
    if env_n_proc > 1:
        env = make_vec_env(
            env_id=envClass,
            n_envs=env_n_proc,
            vec_env_cls=vec_env_cls,
            vec_env_kwargs=vec_env_kwargs,
            env_kwargs=dict(**learning_setting))
        eval_env = make_vec_env(
                env_id=envClass,
                n_envs=1,
                vec_env_cls=vec_env_cls,
                vec_env_kwargs=vec_env_kwargs,
                env_kwargs=dict(**learning_setting)
                ) if learning_setting['eval_enabled'] else None
    else:
        env = envClass(**learning_setting)
        eval_env = envClass(
            **learning_setting
            ) if learning_setting['eval_enabled'] else None

    policy_kwargs = dict(net_arch=[
                         dict(pi=[learning_setting['nn_layer_size']] *
                              learning_setting['nn_num_layers'],
                              vf=[learning_setting['nn_layer_size']] *
                              learning_setting['nn_num_layers'])])

    model = PPO(learning_setting['policy'], env, verbose=0,
                tensorboard_log=learning_setting['tb_log_dir'],
                seed=learning_setting['seed'],
                policy_kwargs=policy_kwargs,
                learning_rate=learning_setting['learning_rate'],
                batch_size=learning_setting['batch_size'],
                gamma=learning_setting['discount_rate'],
                n_steps=learning_setting['n_steps'],
                device=learning_setting['device'])

    model.learn(total_timesteps=learning_setting['total_timesteps'],
                tb_log_name=learning_setting['tb_log_name'] +
                "-" + str(datetime.datetime.now()),
                callback=TensorboardCallback(
                    learning_setting['eval_enabled'],
                    eval_env,
                    learning_setting['eval_freq'],
                    learning_setting['eval_episodes']),
                eval_env=eval_env,
                eval_freq=learning_setting['eval_freq'],
                n_eval_episodes=learning_setting['eval_episodes'],
                eval_log_path=learning_setting['eval_path'],
                )

    if learning_setting['save']:
        model.save("saves/" + learning_setting['tb_log_name'] +
                   "/" + str(datetime.datetime.now()))

    return model


def train_a2c_agent(learning_setting):
    """
    A2C(Advantage Actor Critic Algorithm, Must-Have Learning Settings And
    Default Parameter Values:
        a2c_learning_setting = {}
        a2c_learning_setting['envClass'] = envClass
        a2c_learning_setting['learning_rate'] = 1e-5
        a2c_learning_setting['discount_rate'] = 0.99
        a2c_learning_setting['nn_num_layers'] = 4
        a2c_learning_setting['nn_layer_size'] = 512
        a2c_learning_setting['n_steps'] = 128
        a2c_learning_setting['memory_type'] = memory_type
        a2c_learning_setting['memory_length'] = 3
        a2c_learning_setting['intrinsic_enabled'] = False
        a2c_learning_setting['intrinsic_beta'] = 0.01
        a2c_learning_setting['ae_enabled'] = True
        a2c_learning_setting['ae_comm_list'] = comm_list['a2c-tmazev0']
        a2c_learning_setting['ae_path'] = "models/ae.torch"
        a2c_learning_setting['ae_rcons_err_type'] = "MSE"
        a2c_learning_setting['eval_enabled'] = False
        a2c_learning_setting['eval_freq'] = 1000
        a2c_learning_setting['eval_episodes'] = 0
        a2c_learning_setting['eval_path'] = None
        a2c_learning_setting['env_n_proc'] = 1
        a2c_learning_setting['vec_env_cls'] = DummyVecEnv
        a2c_learning_setting['tb_log_name'] = "a2c-tmazev0"
        a2c_learning_setting['tb_log_dir'] = "./logs/t_maze_tensorboard_0/"
        a2c_learning_setting['maze_length'] = maze_length
        a2c_learning_setting['total_timesteps'] = total_timesteps
        a2c_learning_setting['seed'] = None
        a2c_learning_setting['policy'] = "MlpPolicy"
        a2c_learning_setting['save'] = False
        a2c_learning_setting['device'] = 'cuda:0'
    """

    envClass = learning_setting['envClass']
    env_n_proc = learning_setting.get('env_n_proc', 1)
    vec_env_cls = learning_setting.get('vec_env_cls', DummyVecEnv)
    vec_env_kwargs = dict(start_method='forkserver') if isinstance(
        vec_env_cls, DummyVecEnv) else None
    if env_n_proc > 1:
        env = make_vec_env(
            env_id=envClass,
            n_envs=env_n_proc,
            vec_env_cls=vec_env_cls,
            vec_env_kwargs=vec_env_kwargs,
            env_kwargs=dict(**learning_setting))
        eval_env = make_vec_env(
                env_id=envClass,
                n_envs=1,
                vec_env_cls=vec_env_cls,
                vec_env_kwargs=vec_env_kwargs,
                env_kwargs=dict(**learning_setting)
                ) if learning_setting['eval_enabled'] else None
    else:
        env = envClass(**learning_setting)
        eval_env = envClass(
            **learning_setting
            ) if learning_setting['eval_enabled'] else None

    policy_kwargs = dict(net_arch=[
                         dict(pi=[learning_setting['nn_layer_size']] *
                              learning_setting['nn_num_layers'],
                              vf=[learning_setting['nn_layer_size']] *
                              learning_setting['nn_num_layers'])])

    model = A2C(learning_setting['policy'], env, verbose=0,
                tensorboard_log=learning_setting['tb_log_dir'],
                seed=learning_setting['seed'],
                policy_kwargs=policy_kwargs,
                learning_rate=learning_setting['learning_rate'],
                gamma=learning_setting['discount_rate'],
                n_steps=learning_setting['n_steps'],
                device=learning_setting['device'])

    model.learn(total_timesteps=learning_setting['total_timesteps'],
                tb_log_name=learning_setting['tb_log_name'] +
                "-" + str(datetime.datetime.now()),
                callback=TensorboardCallback(
                    learning_setting['eval_enabled'],
                    eval_env,
                    learning_setting['eval_freq'],
                    learning_setting['eval_episodes']),
                eval_env=eval_env,
                eval_freq=learning_setting['eval_freq'],
                n_eval_episodes=learning_setting['eval_episodes'],
                eval_log_path=learning_setting['eval_path'],
                )

    if learning_setting['save']:
        model.save("saves/" + learning_setting['tb_log_name'] +
                   "/" + str(datetime.datetime.now()))

    return model


def train_ppo_lstm_agent(learning_setting):
    """
    PPO LSTM (Proximal Policy Optimization Algorithm,
    Must-Have Learning Settings And
    Default Parameter Values:
        ppolstm_learning_setting = {}
        ppolstm_learning_setting['envClass'] = envClass
        ppolstm_learning_setting['learning_rate'] = 1e-5
        ppolstm_learning_setting['discount_rate'] = 0.99
        ppolstm_learning_setting['nn_num_layers'] = 2
        ppolstm_learning_setting['nn_layer_size'] = 64
        ppolstm_learning_setting['n_steps'] = 32
        ppolstm_learning_setting['batch_size'] = 32
        ppolstm_learning_setting['memory_type'] = memory_type
        ppolstm_learning_setting['memory_length'] = 3
        ppolstm_learning_setting['intrinsic_enabled'] = False
        ppolstm_learning_setting['intrinsic_beta'] = 0.01
        ppolstm_learning_setting['ae_enabled'] = True
        ppolstm_learning_setting['ae_comm_list'] = comm_list['ppo-tmazev0']
        ppolstm_learning_setting['ae_path'] = "models/ae.torch"
        ppolstm_learning_setting['ae_rcons_err_type'] = "MSE"
        ppolstm_learning_setting['eval_enabled'] = False
        ppolstm_learning_setting['eval_freq'] = 1000
        ppolstm_learning_setting['eval_episodes'] = 0
        ppolstm_learning_setting['eval_path'] = None
        ppolstm_learning_setting['env_n_proc'] = 1
        ppolstm_learning_setting['vec_env_cls'] = DummyVecEnv
        ppolstm_learning_setting['tb_log_name'] = "ppo-tmazev0"
        ppolstm_learning_setting['tb_log_dir'] = "./logs/t_maze_tensorboard_0/"
        ppolstm_learning_setting['maze_length'] = maze_length
        ppolstm_learning_setting['total_timesteps'] = total_timesteps
        ppolstm_learning_setting['seed'] = None
        ppolstm_learning_setting['policy'] = MlpACPolicy
        ppolstm_learning_setting['save'] = False
        ppolstm_learning_setting['device'] = 'cuda:0'
    """
    envClass = learning_setting['envClass']
    env_n_proc = learning_setting.get('env_n_proc', 1)
    vec_env_cls = learning_setting.get('vec_env_cls', DummyVecEnv)
    vec_env_kwargs = dict(start_method='forkserver') if isinstance(
        vec_env_cls, DummyVecEnv) else None
    if env_n_proc > 1:
        env = make_vec_env(
            env_id=envClass,
            n_envs=env_n_proc,
            vec_env_cls=vec_env_cls,
            vec_env_kwargs=vec_env_kwargs,
            env_kwargs=dict(**learning_setting))
        eval_env = make_vec_env(
                env_id=envClass,
                n_envs=1,
                vec_env_cls=vec_env_cls,
                vec_env_kwargs=vec_env_kwargs,
                env_kwargs=dict(**learning_setting)
                ) if learning_setting['eval_enabled'] else None
    else:
        env = envClass(**learning_setting)
        eval_env = envClass(
            **learning_setting
            ) if learning_setting['eval_enabled'] else None

    policy_kwargs = dict(net_arch=[dict(vf=[learning_setting['nn_layer_size']]
                                        * learning_setting['nn_num_layers'])],
                         #  shared_lstm=False,
                         #  enable_critic_lstm=True,
                         ortho_init=False)

    model = RecurrentPPO(
        learning_setting['policy'],
        env,
        n_steps=learning_setting['n_steps'],
        learning_rate=learning_setting['learning_rate'],
        batch_size=learning_setting['batch_size'],
        verbose=0,
        tensorboard_log=learning_setting['tb_log_dir'],
        seed=learning_setting['seed'],
        policy_kwargs=policy_kwargs,
        gamma=learning_setting['discount_rate'],
        device=learning_setting['device']
    )

    model.learn(total_timesteps=learning_setting['total_timesteps'],
                tb_log_name=learning_setting['tb_log_name'] +
                "-" + str(datetime.datetime.now()),
                callback=TensorboardCallback(
                    learning_setting['eval_enabled'],
                    eval_env,
                    learning_setting['eval_freq'],
                    learning_setting['eval_episodes']),
                eval_env=eval_env,
                eval_freq=learning_setting['eval_freq'],
                n_eval_episodes=learning_setting['eval_episodes'],
                eval_log_path=learning_setting['eval_path'],
                )

    if learning_setting['save']:
        model.save("saves/" + learning_setting['tb_log_name'] +
                   "/" + str(datetime.datetime.now()))

    return model


class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(
            self, eval_enabled=False,
            eval_env=None, eval_freq=1,
            eval_episodes=1, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)
        self.eval_enabled = eval_enabled
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.eval_episodes = eval_episodes
        # self.n_calls = 0

    def _on_training_start(self):
        output_formats = self.logger.output_formats
        # Save reference to tensorboard formatter object
        # note: the failure case (not formatter found) is
        # not handled here, should be done with try/except.
        self.tb_formatter = None
        if output_formats:
            self.tb_formatter = next(formatter for formatter in output_formats
                                     if isinstance(formatter,
                                                   TensorBoardOutputFormat))

    def _log_success_callback(self, locals_, globals_):
        """
        Callback passed to the  ``evaluate_policy`` function
        in order to log the success rate (when applicable),
        for instance when using HER.
        :param locals_:
        :param globals_:
        """
        info = locals_["info"]

        if locals_["done"]:
            maybe_is_success = info.get("is_success")
            if maybe_is_success is not None:
                self._is_success_buffer.append(maybe_is_success)

    def _on_step(self) -> bool:
        if self.eval_enabled:
            if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:

                self._is_success_buffer = []

                episode_rewards, episode_lengths = evaluate_policy(
                    self.model,
                    self.eval_env,
                    n_eval_episodes=self.eval_episodes,
                    deterministic=True,
                    return_episode_rewards=True,
                    callback=self._log_success_callback)

                mean_reward, _ = np.mean(
                    episode_rewards), np.std(episode_rewards)
                mean_ep_length, _ = np.mean(
                    episode_lengths), np.std(episode_lengths)
                self.last_mean_reward = mean_reward

                self.logger.record("eval/mean_reward", float(mean_reward))
                self.logger.record("eval/mean_ep_length", mean_ep_length)

                if len(self._is_success_buffer) > 0:
                    success_rate = np.mean(self._is_success_buffer)
                    self.logger.record("eval/success_rate", success_rate)

                self.logger.record(
                    "time/total_timesteps",
                    self.num_timesteps, exclude="tensorboard")
                self.logger.dump(self.num_timesteps)

        return True

        # if True:
        #     if (self.locals['self'].env.buf_dones[-1]):
        #         epi_reward = (
        #             self.model.env.unwrapped.envs[0].episode_returns[-1])
        #         epi_number = len(
        #             self.locals['self'].env.unwrapped.envs[0].episode_lengths)
        #         success_ratio = (
        #             self.locals['self'].env.unwrapped.envs[0].
        #             success_count /
        #             self.locals['self'].env.unwrapped.envs[0].
        #             episode_count) * 100
        #         if self.tb_formatter:
        #             self.tb_formatter.writer.add_scalar(
        #                 "_tmaze/Reward per episode",
        #                 epi_reward, epi_number)

        #         if self.tb_formatter:
        #             self.tb_formatter.writer.flush()
        #             self.tb_formatter.writer.\
        #                 add_scalar(
        #                     "_tmaze/Episode length per episode",
        #                     self.locals['self'].env.
        #                     unwrapped.envs[0].
        #                     episode_lengths[-1],
        #                     epi_number)

        #             self.tb_formatter.writer.\
        #                 add_scalar(
        #                     "_tmaze/Success Ratio per episode",
        #                     success_ratio, epi_number)

        #             if self.locals['self'].env.unwrapped.\
        #                     envs[0].intrinsic_enabled == 1:
        #                 if self.locals['self'].env.unwrapped.\
        #                         envs[0].env_type == "TmazeEnv":
        #                     self.tb_formatter.writer.\
        #                         add_text(
        #                             "_tmaze/Frequency Dictionary String " +
        #                             "per episode",
        #                             str(self.locals['self'].env.
        #                                 unwrapped.envs[0].
        #                                 intrinsic_dict), epi_number)

        #             self.tb_formatter.writer.flush()
        # return True


def make_vec_env(
    env_id,
    n_envs=1,
    seed=None,
    start_index=0,
    monitor_dir=None,
    wrapper_class=None,
    env_kwargs=None,
    vec_env_cls=None,
    vec_env_kwargs=None,
    monitor_kwargs=None,
    wrapper_kwargs=None,
):

    env_kwargs = {} if env_kwargs is None else env_kwargs
    vec_env_kwargs = {} if vec_env_kwargs is None else vec_env_kwargs
    monitor_kwargs = {} if monitor_kwargs is None else monitor_kwargs
    wrapper_kwargs = {} if wrapper_kwargs is None else wrapper_kwargs

    def make_env(rank):
        def _init():
            if isinstance(env_id, str):
                env = gym.make(env_id, **env_kwargs)
            else:
                env = env_id(**env_kwargs, env_id=rank)
            if seed is not None:
                compat_gym_seed(env, seed=seed + rank)
                env.action_space.seed(seed + rank)
            # Wrap the env in a Monitor wrapper
            # to have additional training information
            monitor_path = os.path.join(
                monitor_dir, str(rank)
                ) if monitor_dir is not None else None
            # Create the monitor folder if needed
            if monitor_path is not None:
                os.makedirs(monitor_dir, exist_ok=True)
            env = Monitor(env, filename=monitor_path, **monitor_kwargs)
            # Optionally, wrap the environment with the provided wrapper
            if wrapper_class is not None:
                env = wrapper_class(env, **wrapper_kwargs)
            return env

        return _init

    # No custom VecEnv is passed
    if vec_env_cls is None:
        # Default: use a DummyVecEnv
        vec_env_cls = DummyVecEnv

    return vec_env_cls(
        [make_env(i + start_index) for i in range(n_envs)],
        **vec_env_kwargs)
