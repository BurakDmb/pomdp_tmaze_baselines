from ClassQAgent import QAgent
from EnvTMaze import TMazeEnv


learning_setting = {}
learning_setting['learning_rate'] = 0.001
learning_setting['discount_rate'] = 0.99
learning_setting['epsilon_start'] = 1.0
learning_setting['epsilon_end'] = 0.1


env = TMazeEnv(maze_length=6)
model = QAgent(env, tensorboard_log="logs/t_maze_tensorboard/",
               learning_setting=learning_setting)
model.learn(total_timesteps=500000, tb_log_name="q_agent-")
