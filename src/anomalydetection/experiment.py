from load_dataset import load_dataset
from min_max_scaler import min_max_scaler
from sklearn.model_selection import train_test_split
from make_experiment import make_experiment


def experiment(setting,  mlflow):

    algorithm = setting["z_algorithm"]
    dataset = setting["z_dataset"]
    # name_of_experiment = setting["z_name_of_experiment"]
    #print("Starting experiment")

    X_train, y_train = load_dataset(dataset, algorithm)
    print("Dataset loaded!")

    X_train = min_max_scaler(X_train)[0]

    print("shape X_train : ", X_train.shape)

    make_experiment(algorithm, X_train, y_train, setting, mlflow)



