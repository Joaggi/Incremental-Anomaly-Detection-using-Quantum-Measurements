import scipy
import tensorflow as tf
import numpy as np
import scipy.io

def load_dataset(dataset, algorithm):


    nfile = None
    lfile = None
    if dataset == 'NSL':
        nfile = 'data/nsl.txt'
        lfile = 'data/nsllabel.txt'
    elif dataset == 'KDD':
        nfile = 'data/kdd.txt'
        lfile = 'data/kddlabel.txt'
    elif dataset == 'UNSW':
        nfile = 'data/unsw.txt'
        lfile = 'data/unswlabel.txt'
    elif dataset == 'DOS':
        nfile = 'data/dos.txt'
        lfile = 'data/doslabel.txt'
    else:
        df = scipy.io.loadmat('data/'+dataset+".mat")
        numeric =  df['X']
        labels = (df['y']).astype(float).reshape(-1)


    if dataset in ['KDD', 'NSL', 'UNSW', 'DOS']:
        numeric = np.loadtxt(nfile, delimiter = ',')
        labels = np.loadtxt(lfile, delimiter=',')

    if dataset == 'KDD':
        labels = 1 - labels

    return numeric, labels

