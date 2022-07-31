import itertools
import subprocess

datasets = ['2021_02_01_18_45_51_neurovascular_full_dataset', '2021_02_01_19_19_39_neurovascular_full_dataset']
#wind_backs = [1, 2, 5, 7, 10, 13, 15, 20, 30, 50, 70]
#wind_forwards = [1, 2, 5, 7, 10, 13, 15, 20, 30, 50, 70]
wind_backs = [1, 10, 50]
wind_forwards = [1, 10, 50]
lrs = [0.00001,0.000001]
#decays = [0.00001,0.000001]
dropouts = [0.5,0.6,0.7]
hidden_sizes = ["[800,400,200,100,200,400,800]","[100,200,400,800,400,200,100]","[1000,800,600,400,200,100,50]","[50,100,200,400,600,800,1000]","[150,150,150,150,150,150,150,150]"]

all_combinations = itertools.product(datasets, wind_backs, wind_forwards, lrs, dropouts, hidden_sizes)

for dataset, wind_back, wind_forward, lr, dropout, hidden in all_combinations:
    print(f">>> running: dataset={dataset}, wind_back={wind_back}, wind_forward={wind_forward}, lr={lr}, dropout={dropout}, hidden sizes={hidden}")
    
    cmd = f"python train.py -m experiment=experiment_lin_net.yaml trainer.gpus=1 logger=wandb trainer.min_epochs=50 trainer.max_epochs=200 model.lr={lr} datamodule.dataset_object.dataset_name={dataset} datamodule.batch_size=128 datamodule.dataset_object.window_len_neuro_back={wind_back} datamodule.dataset_object.window_len_neuro_forward={wind_forward} model.dropout={dropout} model.hidden_sizes={hidden} name=lin_net_grid"
    res = subprocess.run(cmd, shell=True)
    if res.returncode != 0:
        print(">>> RUN FAILED!!!")
        print(res.stderr)
        break
    
print("DONE")
