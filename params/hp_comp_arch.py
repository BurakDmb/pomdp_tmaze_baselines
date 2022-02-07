# Compare Architectures, Experiment Study
# Number of each hyperparameter is required to be in
# 2's power. (Ex: 1,2,4,8,...)
hyper_parameters = {}
hyper_parameters['learning_rate'] = [1e-3]
hyper_parameters['nn_num_layers'] = [4]
hyper_parameters['nn_layer_size'] = [4]
hyper_parameters['batch_size'] = [128]
hyper_parameters['memory_type'] = [0, 3, 4, 5]
hyper_parameters['memory_length'] = [1]
hyper_parameters['intrinsic_enabled'] = [0, 1]
hyper_parameters['experiment_no'] = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
hyper_parameters['total_timesteps'] = [1_000_000]
hyper_parameters['maze_length'] = [10]

study_name = "hp_architecture_search"
