import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import scipy

""" plot means std box """
def plotMeanStdBox(means, std, label=None, savePath=None):
    """
    means and std can be array or value
    label must also be value or array
    """

    means = np.array(means)
    std = np.array(std)
    
    q1 = means + std * scipy.stats.norm.ppf(0.25)
    q3 = means + std * scipy.stats.norm.ppf(0.75)
    whislo = q1 - (q3 - q1)*1.5
    whishi = q3 + (q3 - q1)*1.5
    
    if label is None:
        keys = ['med', 'q1', 'q3', 'whislo', 'whishi']
        if len(means.shape) == 0: stats = [dict(zip(keys, (means, q1, q3, whislo, whishi)))]
        else:                     stats = [dict(zip(keys, vals)) for vals in zip(means, q1, q3, whislo, whishi)]
    else:
        keys = ['med', 'q1', 'q3', 'whislo', 'whishi', 'label']
        if len(means.shape) == 0: stats = [dict(zip(keys, (means, q1, q3, whislo, whishi, label)))]
        else:                     stats = [dict(zip(keys, vals)) for vals in zip(means, q1, q3, whislo, whishi, label)]
        
    plt.subplot().bxp(stats, showfliers=False)
    
    if savePath is not None:
        plt.savefig(savePath)

if __name__ == '__main__':

    # construct some data
    x = np.random.default_rng(0).normal(size=(1000,10))

    means = x.mean(axis=0)
    std = x.std(axis=0)
    

    plotMeanStdBox(means, std, savePath='fig.png')