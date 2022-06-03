# PPO Learning Algorithm, Hyperparameter Search
# Number of each hyperparameter is required to be in
# 2's power. (Ex: 1,2,4,8,...)
hyper_parameters = {}
hyper_parameters['learning_rate'] = [1e-7, 1e-5, 1e-4, 1e-3]
hyper_parameters['nn_num_layers'] = [4, 8]
hyper_parameters['nn_layer_size'] = [4, 8, 32, 128]
hyper_parameters['batch_size'] = [32, 128, 256, 512]
hyper_parameters['memory_type'] = [0, 3, 4, 5]
hyper_parameters['memory_length'] = [1, 3, 10, 20]
hyper_parameters['intrinsic_enabled'] = [0, 1]
hyper_parameters['experiment_no'] = [1]
hyper_parameters['total_timesteps'] = [250_000]
hyper_parameters['maze_length'] = [10]

study_name = "hp_parameter_search"
