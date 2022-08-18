import qmc.tf.layers as layers
import qmc.tf.models as models

import numpy as np
from calculate_metrics import calculate_metrics
from find_best_threshold import find_best_threshold

import qincmeasurement  
from tqdm import tqdm


import tensorflow as tf

import timeit 

np.random.seed(42)
tf.random.set_seed(42)


def experiment_qincmeasurement(X_train, y_train, setting, mlflow, best=False):

        
    with mlflow.start_run(run_name=setting["z_run_name"]):

        print("")

        sigma = setting["z_sigma"]
        setting["z_gamma"] = 1/ (2*sigma**2)

        fm_x = layers.QFeatureMapRFF(X_train.shape[1], dim=setting["z_dim_rff"], 
                                     gamma=setting["z_gamma"], random_state=setting["z_random_state"])
        qmd = qincmeasurement.QIncMeasurement(fm_x, setting["z_dim_rff"])
        qmd.compile()

    
        X = X_train[:setting["z_memory"]]
        y = y_train[:setting["z_memory"]]
        X_memory = X[y == 0, :]

        print("Initial fit")
        qmd.fit(X_memory, epochs=1, batch_size=setting["z_batch_size"], verbose=0)
        

        if np.isclose(setting["z_threshold"], 0.0, rtol=0.0):
            thresh = find_best_threshold(y, qmd.predict(X))
            setting["z_threshold"] = thresh

        preds = []
        scores = []

        dataset = tf.data.Dataset.from_tensor_slices(X_train).batch(1)

        print("Streaming...")
        for data in tqdm(dataset):
            #print(data.shape)
            #print("Init predicting...")
            start = timeit.default_timer()
            y_pred = qmd.predict(data)
            stop = timeit.default_timer()
            print('Time inference: ', stop - start)  

            #print("Predicted")
            scores.append(y_pred)
            pred = (y_pred < setting["z_threshold"]).astype(int)
            start = timeit.default_timer()
            qmd.fit(data, epochs=1)
            stop = timeit.default_timer()
            print('Time retraining: ', stop - start)  


            if pred == 0:
                #print("Streaming training")

                start = timeit.default_timer()
                qmd.fit(data, epochs=1)
                stop = timeit.default_timer()
                print('Time retraining: ', stop - start)  

            preds.append(pred)

        metrics = calculate_metrics(y_train, preds, scores, setting["z_run_name"])

        mlflow.log_params(setting)
        mlflow.log_metrics(metrics)

        if best:
            np.savetxt(('artifacts/'+setting["z_name_of_experiment"]+'-preds.csv'), preds, delimiter=',')
            mlflow.log_artifact(('artifacts/'+setting["z_name_of_experiment"]+'-preds.csv'))
            np.savetxt(('artifacts/'+setting["z_name_of_experiment"]+'-scores.csv'), scores, delimiter=',')
            mlflow.log_artifact(('artifacts/'+setting["z_name_of_experiment"]+'-scores.csv'))

        print(f"experiment_dmkde  metrics {metrics}")
        print(f"experiment_dmkde  threshold {setting['z_threshold']}")
