import optuna
# from optuna.visualization import plot_parallel_coordinate


study_name = "hp_parameter_search"
storage_url = "mysql://root:1234@127.0.0.1/pomdp"

# 1- Mysql RDB, used by default.
storage = optuna.storages.RDBStorage(
    url=storage_url)
study = optuna.load_study(study_name=study_name, storage=storage)
# print("Best params: ", study.best_params)
# print("Best value: ", study.best_value)


# hyper_parameters['learning_rate'] = [1e-7, 1e-5, 1e-4, 1e-3]
# hyper_parameters['nn_num_layers'] = [4, 8]
# hyper_parameters['nn_layer_size'] = [4, 8, 32, 128]
# hyper_parameters['batch_size'] = [32, 128, 256, 512]
# hyper_parameters['memory_type'] = [0, 3, 4, 5]
# hyper_parameters['memory_length'] = [1, 3, 10, 20]
# hyper_parameters['intrinsic_enabled'] = [0, 1]
# hyper_parameters['experiment_no'] = [1]
# hyper_parameters['total_timesteps'] = [250_000]
# hyper_parameters['maze_length'] = [10]

# fig = plot_parallel_coordinate(study, params=[
#     "learning_rate",
#     "nn_num_layers",
#     "nn_layer_size",
#     "batch_size"
#     ])

plotContourLogic = True
plotImportanceLogic = True
plotParallelCoordinateLogic = True
plotContourLogic = True
plotSliceLogic = True
plotEDFLogic = True

if plotContourLogic:
    fig = optuna.visualization.plot_contour(study)
    fig.write_html("results/visualization/contour.html")

if plotImportanceLogic:
    fig = optuna.visualization.plot_param_importances(study)
    fig.write_html("results/visualization/importances.html")

if plotParallelCoordinateLogic:
    fig = optuna.visualization.plot_parallel_coordinate(
        study, params=["learning_rate"])
    fig.write_html(
        "results/visualization/parallel_coordinate_learning_rate.html")
    fig = optuna.visualization.plot_parallel_coordinate(
        study, params=["memory_type"])
    fig.write_html(
        "results/visualization/parallel_coordinate_memory_type.html")
    fig = optuna.visualization.plot_parallel_coordinate(
        study, params=["memory_type", "learning_rate"])
    fig.write_html(
        "results/visualization/parallel_coordinate_m_type_lr.html")

if plotContourLogic:
    fig = optuna.visualization.plot_contour(
        study, params=["memory_type", "learning_rate"])
    fig.write_html("results/visualization/contour_memtype_lr.html")
    fig = optuna.visualization.plot_contour(
        study, params=["learning_rate", "memory_type"])
    fig.write_html("results/visualization/contour_lr_memtype.html")
    fig = optuna.visualization.plot_contour(
        study, params=["learning_rate", "batch_size"])
    fig.write_html("results/visualization/contour_lr_batchsize.html")
    fig = optuna.visualization.plot_contour(
        study, params=["learning_rate", "nn_layer_size"])
    fig.write_html("results/visualization/contour_lr_nnlayersize.html")
    fig = optuna.visualization.plot_contour(
        study, params=["learning_rate", "nn_num_layers"])
    fig.write_html("results/visualization/contour_lr_nnnumlayers.html")

if plotSliceLogic:
    fig = optuna.visualization.plot_slice(
        study, params=["learning_rate", "memory_type"])
    fig.write_html("results/visualization/slice.html")

if plotEDFLogic:
    fig = optuna.visualization.plot_edf(study)
    fig.write_html("results/visualization/edf.html")
