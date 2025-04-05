import os
import sys
import pathlib

import numpy as np
import matplotlib.pyplot as plt

import matplotlib.cbook as cbook
import matplotlib.mlab as mlab

bw_method = 'scott'
def _kde_method(X, coords):
    # Unpack in case of e.g. Pandas or xarray object
    X = cbook._unpack_to_numpy(X)
    # fallback gracefully if the vector contains only one value
    if np.all(X[0] == X):
        return (X[0] == coords).astype(float)
    kde = mlab.GaussianKDE(X, bw_method)
    return kde.evaluate(coords)

def get_violin_points(data, points=100):
    '''
    data: shape (N, D)
    '''
    return cbook.violin_stats(data, _kde_method, points=points, quantiles=None)