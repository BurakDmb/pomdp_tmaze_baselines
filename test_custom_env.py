from stable_baselines3 import PPO
from PolicyClass import MultiLayerActorCriticPolicy
from t_maze_env import TMazeEnv, TMazeEnvV1, TMazeEnvV2, TMazeEnvV3
from t_maze_env import TMazeEnvV4, TMazeEnvV5, TMazeEnvV6


env = TMazeEnv(maze_length=6)
env1 = TMazeEnvV1(maze_length=6)
env2 = TMazeEnvV2(maze_length=6)
env3 = TMazeEnvV3(maze_length=6)
env4 = TMazeEnvV4(maze_length=6)
env5 = TMazeEnvV5(maze_length=6)
env6 = TMazeEnvV6(maze_length=6)

print(env.reset(),
      env1.reset(),
      env2.reset(),
      env3.reset(),
      env4.reset(),
      env5.reset(),
      env6.reset())

model = PPO(MultiLayerActorCriticPolicy, env5, verbose=0,
            tensorboard_log="./logs/t_maze_tensorboard/", seed=0)
model.learn(total_timesteps=500000, tb_log_name="first_run")
