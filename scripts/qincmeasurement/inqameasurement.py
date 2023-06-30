try:
    from initialization import initialization
except:
    from notebooks.initialization import initialization

parent_path = initialization("Incremental-Anomaly-Detection-using-Quantum-Measurements", "/home/jagallegom/")


from run_experiment_hyperparameter_search import run_experiment_hyperparameter_search

import sys 


#if len(sys.argv) > 1:
#    algorithm, database = sys.argv[1], sys.argv[2]
#else:
algorithm, database = "inqameasurement", "ionosphere"

#databases = ["cardio", "ionosphere", "mammography", "pima", "satellite", "satimage-2", "Syn", "KDD", "NSL", "cover", "DOS", "UNSW"]
#databases = [ "pima", "satellite", "satimage-2", "Syn", "KDD", "NSL", "cover", "DOS", "UNSW"]
databases = ["DOS", "UNSW"]


settings = {
    "z_algorithm": algorithm,
    "z_dataset": database,
    "z_max_epochs": int(1),
    "z_adaptive": True,
    "z_random_state": 0,
    "z_batch_size": 300,
    "z_threshold": 0,
    "z_batch_size_streaming": 1,
    "z_adaptive_base_lr": 1e-2,
    "z_adaptive_end_lr": 1e-7,
    "z_adaptive_decay_steps": 100,
    "z_adaptive_power": 1,
    "z_adaptive_batch_size": 1024,
    "z_adaptive_epochs": 120,
    "z_adaptive_random_state": None,
    "z_random_search": True,
    "z_random_search_random_state": 401,
    "z_random_search_iter": 30,
    "z_verbose": 1,
    "z_server": "server",
}

prod_settings_ionosphere= {
    "z_rff_components" :  [256, 512, 1000], \
    "z_sigma" :  [2**i for i in range(-5, 10)], \
    "z_memory": [32, 64, 128, 512, 1024, 2048], # la puedes aumentar 
    "z_window_size": [32, 64, 128, 512, 1024, 2048], # la puedes aumentar 
    "z_adaptive_base_lr": [1e-2,1e-3,1e-4],
    "z_adaptive": [False, True],
    "z_division_threshold": [1/2**5, 1/2**2, 1, 2, 2**4, 2**8, 1/2**8],
}


prod_settings_nsl= {
    "z_rff_components" :  [256, 512], \
    "z_sigma" :  [2**i for i in range(-5, 5)], \
    "z_memory": [32, 64, 128, 512 ], # la puedes aumentar 
    "z_window_size": [32, 64, 128, 512], # la puedes aumentar 
    "z_adaptive_base_lr": [1e-2,1e-3,1e-4],
    "z_adaptive": [False, True],
    "z_division_threshold": [1/2**5, 1/2**2, 1, 2, 2**4, 2**8, 1/2**8],
}


prod_settings_kdd= {
    "z_rff_components" :  [256, 512], \
    "z_sigma" :  [2**i for i in range(-5, 5)], \
    "z_memory": [32, 64, 128, 512],
    "z_window_size": [32, 64, 128, 512], # la puedes aumentar 
    "z_adaptive_base_lr": [1e-2,1e-3,1e-4],
    "z_adaptive": [False, True],
    "z_division_threshold": [1/2**5, 1/2**2, 1, 2, 2**4, 2**8, 1/2**8],
}


prod_settings_unsw= {
    "z_rff_components" :  [256, 512], \
    "z_sigma" :  [2**i for i in range(-5, 5)], \
    "z_memory": [32, 64, 128, 512],
    "z_window_size": [32, 64, 128, 512], # la puedes aumentar
    "z_adaptive_base_lr": [1e-2,1e-3,1e-4],
    "z_adaptive": [False, True],
    "z_division_threshold": [1/2**5, 1/2**2, 1, 2, 2**4, 2**8, 1/2**8],
}




def execution(database):
    print(database)
    sigmas = []

    if database == "cardio":
        prod_settings = prod_settings_ionosphere
    elif database == "ionosphere":
        prod_settings = prod_settings_ionosphere
    elif database == "satellite":
        prod_settings = prod_settings_ionosphere
    elif database == "satimage-2":
        prod_settings = prod_settings_ionosphere
    elif database == "mammography":
        prod_settings = prod_settings_ionosphere
    elif database == "pima":
        prod_settings = prod_settings_ionosphere
    elif database == "cover":
        prod_settings = prod_settings_ionosphere
    elif database == "NSL":
        prod_settings = prod_settings_nsl
    elif database == "Syn":
        prod_settings = prod_settings_ionosphere
    elif database == "KDD":
        prod_settings = prod_settings_kdd
    elif database == "DOS":
        prod_settings = prod_settings_unsw
    elif database == "UNSW":
        prod_settings = prod_settings_unsw


    settings["z_dataset"] = database


    run_experiment_hyperparameter_search(algorithm, database, parent_path, prod_settings, settings)

import sys
import tensorflow as tf
print(sys.argv)

if len(sys.argv) >= 2 and sys.argv[1] != None:
    start = int(sys.argv[1])
    jump = 3

else:
    start = 0
    jump = 1

    


if start == 0:
    process_type = '/device:GPU:0'
elif start == 1:
    process_type = '/device:GPU:1'
elif start == 2:
    process_type = '/device:CPU:0'
else:
    process_type = '/device:GPU:0'

with tf.device(process_type):
    for database in databases:
        execution(database)
