import scipy
#import tensorflow as tf
import numpy as np
import scipy.io


def load_dataset(dataset, algorithm):

    nfile = None
    lfile = None
    period = 1
    if dataset == 'NSL':
        nfile = 'data/nsl.txt'
        lfile = 'data/nsllabel.txt'
    elif dataset == 'KDD':
        nfile = 'data/kdd.txt'
        lfile = 'data/kddlabel.txt'
        period = 56
    elif dataset == 'UNSW':
        nfile = 'data/unsw.txt'
        lfile = 'data/unswlabel.txt'
        #period = 164
    elif dataset == 'DOS':
        nfile = 'data/dos.txt'
        lfile = 'data/doslabel.txt'
        #period = 108
    elif dataset == 'Syn':
        nfile = 'data/syn.txt'
        lfile = 'data/synlabel.txt'
    else:
        df = scipy.io.loadmat('data/'+dataset+".mat")
        numeric_total =  df['X']
        labels_total = (df['y']).astype(float).reshape(-1)


    if dataset in ['KDD', 'NSL', 'UNSW', 'DOS', 'Syn']:
        numeric_total = np.loadtxt(nfile, delimiter = ',')
        labels_total = np.loadtxt(lfile, delimiter=',')
        numeric, labels = [], []
        for i in range(len(labels_total)):
            if not i%period: 
                numeric.append(numeric_total[i])
                labels.append(labels_total[i])

    elif dataset == 'cover':
        numeric_total = np.flip(numeric_total)
        labels_total = np.flip(labels_total)
        numeric, labels = [], []
        for i in range(len(labels_total)):
            if not i%50:
                numeric.append(numeric_total[i])
                labels.append(labels_total[i])

    else:
        numeric, labels = numeric_total, labels_total

    if dataset == 'KDD':
        labels = 1 - np.array(labels)


    X = np.array(numeric)
    labels = np.array(labels)

    if(X.ndim == 1): X = X.reshape(-1, 1)

    print(f"Dataset Shape: {X.shape} labels: {labels.shape}")

    return X, labels 
