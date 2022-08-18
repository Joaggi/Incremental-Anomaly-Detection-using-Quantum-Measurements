def run_experiment_hyperparameter_search(algorithm, database, parent_path, prod_settings, custom_setting = None):
    from mlflow_create_experiment import mlflow_create_experiment 
    from generate_product_dict import generate_product_dict, add_random_state_to_dict
    from experiment import experiment
    
    #%%
    #import tensorflow as tf 
    #tf.keras.mixed_precision.set_global_policy("mixed_float16")
 
    import logging
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

   
    
    experiments_folder = "/Doctorado/Incremental-Anomaly-Detection-using-Quantum-Measurements/"
    
    print("#--------------------------------#-----------#")
    setting = {
        "z_base_lr": 1e-3,
        "z_end_lr": 1e-7,
        "z_max_epochs": int(50),
        "z_decay_steps": int(50),
        "z_dataset_size": 40000,
        "z_weight_decay": 1e-4, 
        "z_batch_size": 64,
        "z_train_size": 40000,
        "z_test_size": 1000,
        "z_power": 0.5,
        "z_algorithm": f"{algorithm}",
        "z_experiment": f"{database}_{algorithm}",
        "z_dataset": f"{database}",
        "experiments_folder": experiments_folder,
        "logs": experiments_folder + "logs/",
        "models": experiments_folder + "models/",
        "datasets": experiments_folder + "data/",
        "sufix_name_experiment" : f"{database}_{algorithm}",
        "z_dimension": 2,
        "z_run_name": f"{database}_{algorithm}",
        "z_plot_graph_density": True,
        "z_name_of_experiment": 'conditional-density-experiment',
        "z_random_search": True,
        #"z_random_search": False,
        "z_random_search_random_state": 1,
        "z_random_search_iter": 30,
        "z_step": "train", 
        "z_adaptive_epochs": 2000,
        "z_adaptive_learning_rate": 0.0001
    }
    
    if custom_setting is not None:
        setting = dict(setting, **custom_setting)
    
    server = None
    if("z_server" in setting.keys()):
        server = setting["z_server"]

    mlflow = mlflow_create_experiment(f"tracking.db", f"registry.db", setting["z_name_of_experiment"], server)
    
    settings = generate_product_dict(setting, prod_settings)
    settings = add_random_state_to_dict(settings)
    
    
    for i, setting in enumerate(settings):
        if("verbose" in setting and setting["verbose"]): print(i) 
        experiment(setting = setting, mlflow = mlflow)
    


