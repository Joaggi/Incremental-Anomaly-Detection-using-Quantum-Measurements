def run_experiment_hyperparameter_search(algorithm, database, parent_path, prod_settings, custom_setting = None):
    from mlflow_create_experiment import mlflow_create_experiment 
    from generate_product_dict import generate_product_dict, add_random_state_to_dict
    from experiment import experiment
    
    #%%
    #import tensorflow as tf 
    #tf.keras.mixed_precision.set_global_policy("mixed_float16")
 
    import logging
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

   
    
    #experiments_folder = "/Doctorado/Incremental-Anomaly-Detection-using-Quantum-Measurements/"
    
    print("#--------------------------------#-----------#")
    setting = {
        "z_algorithm": f"{algorithm}",
        "z_experiment": f"{database}_{algorithm}",
        "z_run_name": "Incremental_inqameasurement" #f"{database}_{algorithm}",
    }
    
    if custom_setting is not None:
        setting = dict(setting, **custom_setting)
    

    mlflow = mlflow_create_experiment(setting["z_run_name"])
    
    settings = generate_product_dict(setting, prod_settings)
    settings = add_random_state_to_dict(settings)
    print("Settings created!")

    #print(len(settings))
    for i, setting in enumerate(settings):
        #if("verbose" in setting and setting["verbose"]): print(i)
        #print("Experiments!") 
        experiment(setting = setting, mlflow = mlflow)
    


