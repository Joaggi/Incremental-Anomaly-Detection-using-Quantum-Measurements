try:
    from initialization import initialization
except:
    from notebooks.initialization import initialization

parent_path = initialization("Incremental-Anomaly-Detection-using-Quantum-Measurements", "/home/jagallegom/")

from run_experiment_hyperparameter_search import run_experiment_hyperparameter_search

import sys 


database = "UNSW"

settings = {
    "z_dataset": database,
    "z_adaptive": True,
    "z_random_state": 0,
    "z_batch_size": 200,
    "z_threshold": 0.0,
    "z_batch_size_streaming": 1,
    "z_adaptive_end_lr": 1e-7,
    "z_adaptive_decay_steps": 50,
    "z_adaptive_power": 1,
    "z_adaptive_batch_size": 1024,
    "z_adaptive_epochs": 50,
    "z_adaptive_random_state": None,
}



prod_settings = {
    "z_rff_components": [200], 
    "z_memory": [2000], # la puedes aumentar 
    "z_remember_alpha": [0.6], #[0.01, 0.05, 0.1, 0.25, 0.5, 0.85],
    "z_adaptive_base_lr": [1e-3], #[1e-2,1e-3,1e-4],
    "z_adaptive": [True],
    "z_thresh_correction_factor": [2**0]
}


sigmas = []

if database == "cardio":
    sigmas = [2.5, 3]
elif database == "ionosphere":
    sigmas = [0.9]
elif database == "satellite":
    sigmas = [0.7]
elif database == "satimage-2":
    sigmas = [0.65, 0.8]
elif database == "mammography":
    sigmas = [0.4, 4]
elif database == "pima":
    sigmas = [0.5]
elif database == "cover":
    sigmas = [0.45, 0.5, 2]
elif database == "NSL":
    sigmas = [1, 1.25]
elif database == "Syn":
    sigmas = [0.1]
elif database == "KDD":
    sigmas = [2]
elif database == "DOS":
    sigmas = [0.1, 0.11, 0.15]
elif database == "UNSW":
    sigmas = [2.5]


prod_settings["z_sigma"] = sigmas


run_experiment_hyperparameter_search("inqmeasurement", database, parent_path, prod_settings, settings)
