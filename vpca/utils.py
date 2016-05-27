import numpy as np
import os
import pickle
import subprocess
from scipy.io import loadmat


def download():
    """
    Download example file and save matfile to pickle format
    {0: {'data': ..., 't': ...},
     1: {'data': ..., 't': ...},
     ...}
    """
    if not os.path.isdir('jPCA_ForDistribution'):
        subprocess.call(['wget', 'http://churchlandlab.neuroscience.columbia.edu/code/jPCA_ForDistribution.zip'])
        subprocess.call(['unzip', 'jPCA_ForDistribution.zip'])
        subprocess.call(['rm', 'jPCA_ForDistribution.zip'])
        subprocess.call(['rm', '-rf', '__MACOSX'])

    data = loadmat('jPCA_ForDistribution/exampleData.mat')['Data'][0]
    data_dict = {}
    for i in range(data.shape[0]):
        (X, t) = data[i]
        dict_ = {'t': t.ravel(), 'data': X}
        data_dict[i] = dict_

    pickle.dump(data_dict, open('example_data.pickle', 'wb'), protocol=-1)


def pca(X, n_components=2):
    """
    reduce matrix dimensions using
    principal components analysis
    """
    X = X - np.mean(X, axis=0)
    U, S, V = np.linalg.svd(X, full_matrices=False)
    X_pca = U[:,:n_components] * S[:n_components]
    return X_pca
