import gym
from stable_baselines3 import PPO
from PolicyClass import MultiLayerActorCriticPolicy
import torch.multiprocessing as mp


def train1():

    env1 = gym.make('CartPole-v1')
    model1 = PPO(MultiLayerActorCriticPolicy, env1, verbose=0,
                 tensorboard_log="./logs/cartpole_tensorboard/", seed=0)
    model1.learn(total_timesteps=10000, tb_log_name="first_run")


def train2():

    env2 = gym.make('CartPole-v1')
    model2 = PPO('MlpPolicy', env2, verbose=0,
                 tensorboard_log="./logs/cartpole_tensorboard/", seed=0)
    model2.learn(total_timesteps=10000, tb_log_name="second_run")


def test(model, env):
    obs = env.reset()
    for i in range(1000):
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            obs = env.reset()


if __name__ == '__main__':
    mp.set_start_method('spawn')

    processes = []
    for rank in range(1):
        p1 = mp.Process(target=train1)
        p2 = mp.Process(target=train2)
        p1.start()
        p2.start()
        processes.append(p1)
        processes.append(p2)
    for p in processes:
        p.join()
