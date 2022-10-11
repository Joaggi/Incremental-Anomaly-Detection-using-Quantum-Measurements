
import qmc.tf.layers as layers
import qmc.tf.models as models

import numpy as np
from calculate_metrics import calculate_metrics
from find_best_threshold import find_best_threshold
import adaptive_rff

import inqmeasurement  
from tqdm import tqdm

import jax
from jax import jit
import jax.numpy as jnp
import numpy as np

import tensorflow as tf

import timeit 

from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances

np.random.seed(42)
tf.random.set_seed(42)


def experiment_inqmeasurement(X, y, setting, mlflow, best=False):

        
    with mlflow.start_run(run_name=setting["z_run_name"]):

        print("")

        sigma = setting["z_sigma"]
        setting["z_gamma"] = 1/ (2*sigma**2)

        from sklearn.preprocessing import MinMaxScaler


        scaler = MinMaxScaler()
        scaler.fit(X[:setting["z_memory"]])
        X_train = scaler.transform(X[:setting["z_memory"], :])
        X_test = scaler.transform(X)

        A = euclidean_distances(X_train, X_train)
        plt.axes(frameon = 0)
        plt.hist(A[np.triu_indices_from(A, k=1)].ravel(), density=True, bins = 40)
        plt.savefig('histogram.png',dpi = 300)
        #plt.show()


        setting["z_adaptive_input_dimension"] = X_train.shape[1]

        y_train = y[:setting["z_memory"]]

        X_memory = X_train[y_train == 0, :]

        datos = np.concatenate([np.random.uniform(-3, 3, \
            size=(X_memory.shape[0],X_memory.shape[1])), X_memory])
        


        model = inqmeasurement.InqMeasurement(X_memory.shape[1], dim_x=setting["z_rff_components"],
                gamma=setting["z_gamma"], random_state=setting["z_random_state"], 
                batch_size=setting["z_batch_size"])

        if setting["z_adaptive"] == True:
            rff_layer = adaptive_rff.fit_transform(setting, datos)
            model.fm_x.update_rff(rff_layer.rff_weights.numpy(), rff_layer.offset.numpy())

        model.initial_train(jnp.array(X_memory), 1)


        if np.isclose(setting["z_threshold"], 0.0, rtol=0.0):
            thresh = find_best_threshold(y_train, model.predict(X_train))
            thresh /= setting["z_division_theshold"]
            setting["z_threshold"] = thresh
            
        
        batch_size = setting["z_batch_size_streaming"]
        num_train = X_test.shape[0]
        num_complete_batches, leftover = divmod(num_train, batch_size)
        num_batches = num_complete_batches + bool(leftover)
        perm = jnp.arange(num_train)
        preds = []
        scores = []
        print(num_batches)
        print("Streaming...")
        for i in tqdm(range(num_batches)):
          batch_idx = perm[i * batch_size: (i + 1)*batch_size]
          data = X_test[batch_idx, :]

          y_pred = model.predict(data)
          scores.append(y_pred)

          pred = (y_pred < thresh).astype(int)
          preds.append(pred)

          # retraining
          if pred == 0: # 
              model.initial_train(data, setting["z_remember_alpha"])

        preds = np.array(jnp.concatenate(preds,axis=0))
        scores = np.array(jnp.concatenate(scores,axis=0))


        metrics = calculate_metrics(y, preds, scores, setting["z_run_name"])

        mlflow.log_params(setting)
        mlflow.log_metrics(metrics)

        if best:
            np.savetxt(('artifacts/'+setting["z_name_of_experiment"]+'-preds.csv'), preds, delimiter=',')
            mlflow.log_artifact(('artifacts/'+setting["z_name_of_experiment"]+'-preds.csv'))
            np.savetxt(('artifacts/'+setting["z_name_of_experiment"]+'-scores.csv'), scores, delimiter=',')
            mlflow.log_artifact(('artifacts/'+setting["z_name_of_experiment"]+'-scores.csv'))

        print(f"experiment_dmkde  metrics {metrics}")
        print(f"experiment_dmkde  threshold {setting['z_threshold']}")
