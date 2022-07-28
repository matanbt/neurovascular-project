"""
Usage: python scripts/grid_search.py {function_name}
"""
import itertools
import subprocess
import typer


app = typer.Typer()


@app.command()
def lin_regr_wind_grid():
    # Grid on possible window-sizes (lin regression)
    name = 'lin_regr_wind_grid'
    datasets = ['2021_02_01_18_45_51_neurovascular_full_dataset', '2021_02_01_19_19_39_neurovascular_full_dataset']
    wind_backs = [1, 2, 5, 7, 10, 13, 15, 20, 30, 50, 70]
    wind_forwards = [1, 2, 5, 7, 10, 13, 15, 20, 30, 50, 70]
    all_combinations = itertools.product(datasets, wind_backs, wind_forwards)

    for dataset, wind_back, wind_forward in all_combinations:
        print(f">>> running: dataset={dataset}, wind_back={wind_back}, wind_forward={wind_forward}")
        cmd = "python train.py -m experiment=experiment_lin_regr trainer.gpus=1 logger=wandb trainer.min_epochs=50 " \
              "trainer.max_epochs=500 " \
              f"datamodule.dataset_object.dataset_name={dataset} " \
              "datamodule.batch_size=128 " \
              f"datamodule.dataset_object.window_len_neuro_back={wind_back}" \
              f" datamodule.dataset_object.window_len_neuro_forward={wind_forward} " \
              f"name={name}"
        res = subprocess.run(cmd, shell=True)
        if res.returncode != 0:
            print(">>> RUN FAILED!!!")
            print(res.stderr)
            break

    print("DONE")


@app.command()
def ehrf_arch_grid():
    # Grid on possible architectures of ehrf
    name = 'ehrf_arch_grid'
    datasets = ['2021_02_01_18_45_51_neurovascular_full_dataset', '2021_02_01_19_19_39_neurovascular_full_dataset']
    with_distances = [True, False]
    with_vascular_mean = [True, False]
    with_latent_fcnn = [True, False]
    all_combinations = itertools.product(datasets, with_distances, with_vascular_mean, with_latent_fcnn)

    for dataset, with_distances, with_vascular_mean, with_latent_fcnn in all_combinations:
        print(f">>> running: dataset={dataset}, with_distances={with_distances}, "
              f"with_vascular_mean={with_vascular_mean}, with_latent_fcnn={with_latent_fcnn}")
        cmd = f"python train.py -m experiment=experiment_ehrf " \
              f"trainer.gpus=1 logger=wandb trainer.min_epochs=20 trainer.max_epochs=60 " \
              f"model.lr=0.000075 " \
              f"datamodule.dataset_object.dataset_name={dataset} " \
              f"datamodule.batch_size=128" \
              f"datamodule.dataset_object.window_len_neuro_back=15 " \
              f"datamodule.dataset_object.window_len_neuro_forward=10 " \
              f"model.with_vascular_mean={with_vascular_mean} " \
              f"model.with_distances={with_distances} " \
              f"model.conv_1d_hidden_layers=20 " \
              f"model.with_latent_fcnn={with_latent_fcnn} " \
              f"model.latent_hidden_layers=2 model.latent_hidden_layer_dim=500 model.latent_hidden_dropout=0.3 " \
              f"name={name}"
        res = subprocess.run(cmd, shell=True)
        if res.returncode != 0:
            print(">>> RUN FAILED!!!")
            print(res.stderr)
            break

    print("DONE")


if __name__ == '__main__':
    app()