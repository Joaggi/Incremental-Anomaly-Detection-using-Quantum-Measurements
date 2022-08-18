from experiment_qincmeasurement import experiment_qincmeasurement
from experiment_inqmeasurement import experiment_inqmeasurement

# Uncomment the following line when running a Pyod notebook
# Keep it commented otherwise
#from experiment_pyod import experiment_pyod

def make_experiment(algorithm, X_train, y_train, settings, mlflow, best=False):
    
    if algorithm == "qincmeasurement":
        experiment_qincmeasurement(X_train, y_train, settings, mlflow, best)
    if algorithm == "inqmeasurement":
        experiment_inqmeasurement(X_train, y_train, settings, mlflow, best)
 
