# python 3.5
# utf-8

""" Ki67/dsRed labeling experiment and individual fit for model omega, to be
used in comibnation with other data by overhead file bayes_prime.py """

import numpy as np
import matplotlib.pyplot as plt
import pylab
import seaborn
from scipy.integrate import odeint
import math

import tumour_ODE
try:
    reload
except NameError:
    from importlib import reload
reload(tumour_ODE)
from tumour_ODE import tumour_ODE

pylab.ion()
seaborn.set_context('talk')
seaborn.set_style('ticks')
plt.rc('text', usetex=True)
cmap = ['lightgreen', 'forestgreen', 'tomato', 'firebrick', 'cornflowerblue',
        'darkblue']


""" Parameters of glioblastoma growth (rates in 1/day)"""

lambS = 0.21  # symmetric division rate of stem cells
lambA = 0.36  # asymmetric division rate of stem cells

beta = 0.1  # migration rate out of clone of active stem cells
lamb1 = 1.115  # division rate of progeny
d1 = 1.117  # probability that a progeny division leads to 2P (and not P+N)
mu1 = 0.13  # death rate of progeny


def Ki67_series(t1=30, timevec=np.arange(1, 20, 0.1), lambS=lambS, lambA=lambA,
                beta=beta, lamb1=lamb1, d1=d1, mu1=mu1, control=False):
    """ Simulates a series of dsRed labelings followed by Ki67 staining.

    dsRed is given at t1, all proliferating progeny are labeled;
    at t1+timevec[i] proliferative cells are marked and the ratio of
    Ki67+/(dsRed+,Ki67+) is returned.

    Args:
        t1 (float): Time at which retrovirus with dsRed is given.
        timevec (array): Time span on top of t1 after which proliferating
            cells in the dsRed fraction are stained.
        [lambS ... mu1] (floats): Parameter values for the system.

    Returns:
        [Fraction of labeled cells at each timepoint in the timevec.]
    """
    # run system until t1 and find proliferating cells there
    y0 = [0, 1, 0, 0]
    tvec1 = np.arange(0, t1)
    soln1 = odeint(tumour_ODE, y0, tvec1, args=(lambS, lambA, lamb1, d1, mu1))
    solution1 = soln1.T
    # take the last coloumn of the solution to form starting values for the
    # labeled run
    y1 = solution1[:, -1]
    # the SCs and the last compartment are not marked (quiescent/exh.)
    y1[0] = 0
    y1[1] = 0
    y1[3] = 0
    # new time vector goes from t1 to t1+timevec (equivalent to 0 to timevec)
    tvec2 = np.concatenate((np.array([0]), timevec))
    soln2 = odeint(tumour_ODE, y1, tvec2, args=(lambS, lambA, lamb1, d1, mu1))
    solution2 = soln2.T
    # for every timepoint in timevec, calculate ratio of proliferative cells
    # over all cells (first timepoint is t1, disregard)
    total = np.sum(solution2, axis=0)[1:]
    prolif = solution2[2][1:]
    Ki67frac = prolif/total

    if control:
        plt.figure()
        plt.plot(tvec2[1:], prolif, label='prolif')
        plt.plot(tvec2[1:], total, label='total')
        plt.yscale('log')
        plt.legend(loc=0)

        plt.figure()
        plt.plot(tvec2[1:], solution2[1, 1:]/total, label='aSC')
        plt.plot(tvec2[1:], solution2[2, 1:]/total, label='Progeny')
        plt.plot(tvec2[1:], solution2[3, 1:]/total, label='Terminal')

        plt.legend()

    return Ki67frac


def Ki67_residual(params):
    """ Calculates residuals.

    Args:
        params (lmfit Parameters object): Model parameters, partly to be
            fitted.
    Returns:
        List of residuals scaled by assumed errors.
    """
    # # data values
    # data_times = [1, 1, 3, 6, 6, 6, 7, 9, 9, 12, 13, 13, 13]
    # data_vals = [0.5917, 0.3973, 0.2644, 0.1027, 0.2381, 0.1888, 0.0991,
    #              0.1516, 0.1311, 0.1615, 0.2015, 0.068, 0.094]
    # data_errors = [0.1 for x in data_vals]

    # data values (means)
    data_times = [1, 3, 6, 7, 9, 12, 13]
    multiplicity = [2, 1, 3, 1, 2, 1, 3]  # animals used per tp
    data_vals = [0.4945, 0.2644, 0.1765, 0.0991, 0.1414, 0.1615, 0.1211]
    data_errors = [0.06/math.sqrt(multiplicity[i]) for i in range(len(data_vals))]

    # extract parameters if Parameters object was passed
    if type(params) == list:
        lambS, lambA, lamb1, d1, mu1 = params
    else:
        lambS = params['lambS'].value
        lambA = params['lambA'].value
        lamb1 = params['lamb1'].value
        d1 = params['d1'].value
        mu1 = params['mu1'].value

    # calculate model values, problem: time series is not monotonous,
    # get results for unique version and then reassamble
    time_mono = list(set(data_times))
    modvals = Ki67_series(t1=30, timevec=time_mono, lambS=lambS, lambA=lambA,
                          lamb1=lamb1, d1=d1, mu1=mu1,)
    # make dict of these unique values
    mod_dict = dict(zip(time_mono, modvals))
    # create value list coherent to multiple appearance of days in data
    model_vals = [mod_dict[day] for day in data_times]
    # calculate residuals
    residuals = (np.array(data_vals) - model_vals) / np.array(data_errors)
    return residuals
