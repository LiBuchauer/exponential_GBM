# python 3.5
# utf-8

""" Evaluation of single-cell labeling clone size data """

import numpy as np
import matplotlib.pyplot as plt
import pylab
import seaborn

import pyximport
pyximport.install(
    setup_args={"include_dirs": np.get_include()},
    reload_support=True)
import SSA_prime_SPmigration
from importlib import reload
reload(SSA_prime_SPmigration)
from SSA_prime_SPmigration import simstats

pylab.ion()
seaborn.set_context('talk')
plt.rc('text', usetex=True)
cmap = ['lightgreen', 'forestgreen', 'tomato', 'firebrick', 'cornflowerblue',
        'darkblue']


""" Parameters of glioblastoma growth (rates in 1/day)"""

lambS = 0.21  # symmetric division rate of stem cells
lambA = 0.21  # asymmetric division rate of stem cells

beta = 0.1  # migration rate out of clone of active stem cells
lamb1 = 1.1  # division rate of progeny
d1 = 1.1  # probability that a progeny division leads to 2P (and not P+N)
mu1 = 0.1  # death rate of progeny


def tracing_residual(params):
    """ Calculates residuals for clonal data.

    Args:
        params (lmfit Parameters object): Model parameters, partly to be
            fitted.
    Returns:
        List of residuals scaled by assumed errors.
    """

    # extract parameters if Parameters object was passed
    if type(params) == list:
        lambS, lambA, lamb1, d1, mu1, beta = params
    else:
        lambS = params['lambS'].value
        lambA = params['lambA'].value
        beta = params['beta'].value
        lamb1 = params['lamb1'].value
        d1 = params['d1'].value
        mu1 = params['mu1'].value

    # simulation
    meanM, CVM, f1M, f2M = simstats(endtime=20, famnum=100, lambS=lambS,
                                    lambA=lambA, lamb1=lamb1, d1=d1, mu1=mu1,
                                    beta=beta)

    # preparation of data, current procedure:
    # build mean of the last four timepoints (day 20, day 20, day 26 and
    # day 37) assuming that the statistics we are evaluating have reached a
    # steady state. We have only four measurements with considerable bootstrap
    # errors here and thus take a conservative error model, where we propagate
    # the error into the final quantity (the mean of the four values).
    # In the case of the means, this error estimate is about four times as high
    # as the standard error of mean and twice as high as the standard deviation

    # data statistics at the four timepoints
    means = [2.7194, 3.06429, 2.9876, 2.5282]
    CVs = [2.6714, 2.3228, 1.7128, 1.5331]
    f1s = [0.6403, 0.5357, 0.5093, 0.5423]
    f2s = [0.1295, 0.20714, 0.2050, 0.1831]

    meanD = np.mean(means)
    CVD = np.mean(CVs)
    f1D = np.mean(f1s)
    f2D = np.mean(f2s)

    # bootstrapped errors at the four timepoints
    BTmeans = [0.57720, 0.590116, 0.379954, 0.335979]
    BTCVs = [0.85929, 0.5156, 0.2282, 0.2666]
    BTf1s = [0.04065, 0.041996, 0.03902, 0.03826]
    BTf2s = [0.02510, 0.03283, 0.034609, 0.02997]

    EmeanD = np.mean(BTmeans)
    ECVD = np.mean(BTCVs)
    Ef1D = np.mean(BTf1s)
    Ef2D = np.mean(BTf2s)

    # calculate residuals
    residuals = (np.array([meanD, CVD, f1D, f2D]) -
        np.array([meanM, CVM, f1M, f2M])) / np.array([EmeanD, ECVD, Ef1D, Ef2D])
    return residuals
