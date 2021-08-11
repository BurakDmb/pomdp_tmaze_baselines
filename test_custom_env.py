from stable_baselines3 import PPO
from custom_policy import CustomActorCriticPolicy
from t_maze_env import TMazeEnv


env = TMazeEnv(maze_length=6)

model = PPO(CustomActorCriticPolicy, env, verbose=0,
            tensorboard_log="./logs/t_maze_tensorboard/", seed=0)
model.learn(total_timesteps=500000, tb_log_name="first_run")
