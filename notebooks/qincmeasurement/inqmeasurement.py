try:
    from initialization import initialization
except:
    from notebooks.initialization import initialization

parent_path = initialization("Incremental-Anomaly-Detection-using-Quantum-Measurements", "/Doctorado/")

from run_experiment_hyperparameter_search import run_experiment_hyperparameter_search

import sys 

if len(sys.argv) > 1:
    algorithm, database = sys.argv[1], sys.argv[2]
else:
    algorithm, database = "inqmeasurement", "KDD"

settings = {
    "z_algorithm": algorithm,
    "z_database": database,
    "z_max_epochs": int(1),
    "z_adaptive": True,
    "z_random_state": 0,
    "z_batch_size": 200,
    "z_threshold": 0,
    "z_batch_size_streaming": 1,
    "z_adaptive_base_lr": 1e-2,
    "z_adaptive_end_lr": 1e-7,
    "z_adaptive_decay_steps": 100,
    "z_adaptive_power": 1,
    "z_adaptive_batch_size": 1024,
    "z_adaptive_epochs": 120,
    "z_adaptive_random_state": None,
}

prod_settings_nsl= {
    "z_rff_components" :  [2000], \
    "z_sigma" :  [1,1.1,1.2,1.3,1.4], \
    "z_memory": [2048, 4096], # la puedes aumentar 
    "z_remember_alpha": [0.01, 0.05, 0.1, 0.3, 0.5, 0.85],
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


prod_settings = prod_settings_kdd

run_experiment_hyperparameter_search(algorithm, database, parent_path, prod_settings, settings)

