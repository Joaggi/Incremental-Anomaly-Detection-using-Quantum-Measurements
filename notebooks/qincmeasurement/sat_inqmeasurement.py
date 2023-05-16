try:
    from initialization import initialization
except:
    from notebooks.initialization import initialization

parent_path = initialization("Incremental-Anomaly-Detection-using-Quantum-Measurements", "/home/oabustosb/Desktop/")

from run_experiment_hyperparameter_search import run_experiment_hyperparameter_search

import sys 


database = "satimage-2"

settings = {
    "z_dataset": database,
    "z_adaptive": True,
    "z_random_state": 0,
    "z_batch_size": 200,
    "z_threshold": 0.0,
    "z_batch_size_streaming": 1,
    "z_adaptive_end_lr": 1e-7,
    "z_adaptive_decay_steps": 100,
    "z_adaptive_power": 1,
    "z_adaptive_batch_size": 1024,
    "z_adaptive_epochs": 120,
    "z_adaptive_random_state": None,
}


prod_settings_nsl = {
    "z_rff_components": [2000], \
    "z_sigma" :  [0.5,1,1.1,1.2,1.3,1.5,2], \
    "z_memory": [2048], # la puedes aumentar 
    "z_remember_alpha": [0.01, 0.05, 0.1, 0.25, 0.5, 0.85],
    "z_adaptive_base_lr": [1e-2,1e-3,1e-4],
    "z_adaptive": [False, True],
}


prod_settings_kdd= {
    "z_rff_components" :  [2000], \
    "z_sigma" :  [1,1.1,1.2,1.3,1.4], \
    "z_memory": [256],
    "z_remember_alpha": [0.1]
}


prod_settings_unsw= {
    "z_rff_components" :  [2000], \
    "z_sigma" :  [0.5,0.7, 1,1.1,1.2,1.3,1.4], \
    "z_memory": [2048],
    "z_remember_alpha": [0.1]
}



prod_settings = {
    "z_rff_components": [2000], 
    "z_memory": [2048], # la puedes aumentar 
    "z_remember_alpha": [0.01, 0.05, 0.1, 0.25, 0.5, 0.85],
    "z_adaptive_base_lr": [1e-2,1e-3,1e-4],
    "z_adaptive": [False, True],
}


sigmas = []

if database == "cardio":
    sigmas = [2.0, 2.5]
elif database == "ionosphere":
    sigmas = [0.3, 0.9, 1.5]
elif database == "mammography":
    sigmas = [0.4, 1.0]
elif database == "pima":
    sigmas = [0.5, 1.0, 1.25]
elif database == "cover":
    sigmas = [0.5, 1.25]
elif database == "NSL":
    sigmas = [1]
elif database == "satellite":
    sigmas = [1.0, 2.0, 3.0]
elif database == "satimage-2":
    sigmas = [1.5, 2.5, 3]
     

prod_settings["z_sigma"] = sigmas


run_experiment_hyperparameter_search("inqmeasurement", database, parent_path, prod_settings, settings)
