# python 3.5
# utf-8

""" BLI population growth system to be used in combination with other data
by overhead file bayes_prime.py """

import numpy as np
import matplotlib.pyplot as plt
import pylab
import seaborn
from scipy import stats

import tumour_ODE
try:
    reload
except NameError:
    from importlib import reload
reload(tumour_ODE)
from tumour_ODE import tumour_ODE

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


def BLI_residual(params):
    """ Calculates residuals.

    Args:
        params (lmfit Parameters object): Model parameters, partly to be
            fitted.
    Returns:
        List of residuals scaled by assumed errors.
    """

    lambS = params['lambS'].value
    BLI_res = np.array([(lambS - 0.213) / (0.071/np.sqrt(23))])

    return BLI_res


def lamb_eff_dist():
    """ Plots distribution of individual effective growth rates from
    exponential fits to single mouse BLI curves."""
    def r2(num):
        return np.round(num, 2)
    lambs = [0.21, 0.25, 0.33, 0.13, 0.16, 0.23, 0.31, 0.23, 0.23, 0.23, 0.12,
             0.15, 0.19, 0.21, 0.08, 0.21, 0.3, 0.17, 0.37, 0.21, 0.29, 0.14,
             0.14]
    mean = r2(np.mean(lambs))
    std = r2(np.std(lambs))
    SEM = r2(np.std(lambs))/len(lambs)
    plt.figure()
    plt.hist(lambs, 6, color="#009E73", align='mid', rwidth=0.95)
    plt.plot([np.mean(lambs), np.mean(lambs)], [0, 8], color="#000000")
    plt.plot([np.mean(lambs)+np.std(lambs), np.mean(lambs)+np.std(lambs)], [0, 8], color="#000000", linestyle="--")
    plt.plot([np.mean(lambs)-np.std(lambs), np.mean(lambs)-np.std(lambs)], [0, 8], color="#000000", linestyle="--")
    plt.xlabel('individual growth rate (1/day)')
    plt.ylabel('occurence')
    plt.ylim(0, 6.3)
    plt.title(r'mean growth rate $\mu_{{\lambda_0}}$ = {} /day\\ standard deviation $\sigma_{{\lambda_0}}$ = {} /day '.format(mean, std))
    plt.text(0.33, 5.9, 'N = {}'.format(len(lambs)))

    plt.show()
    pylab.savefig('figures/lambda_dist.pdf', bbox_inches='tight')
    print(stats.kstest(lambs, 'norm'))
