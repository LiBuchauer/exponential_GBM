# python 3.5
# utf-8

""" Data collected from Zhe Zhu's doctoral thesis
(https://katalog.ub.uni-heidelberg.de/titel/67447115) and Cell Stem Cell
publication (10.1016/j.stem.2014.04.007) for fitting model prime
as well as a new FACS experiment on dsred labelled tumors. """

import numpy as np
import matplotlib.pyplot as plt
import pylab
import seaborn
from scipy.integrate import odeint
from matplotlib.patches import Rectangle

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


def rest_residual(params):
    """ Calculates residuals.

    Args:
        params (lmfit Parameters object): Model parameters, partly to be
            fitted.
    Returns:
        List of residuals scaled by assumed errors.
    """
    # data ratios for 0) Ki67GFP+/Ki67+, 1) Ki67GFP+/GFP+, 2) G0/GFP+,
    # 3) GFP-/all 4) Ki67/bulk
    # data_ratios = [0.06, 0.1, 0.83, 0.30]
    # data_errors = [0.02, 0.05, 0.02, 0.05]

    # extract parameters if Parameters object was passed
    if type(params) == list:
        lambS, lambA, lamb1, d1, mu1 = params
    else:
        lambS = params['lambS'].value
        lambA = params['lambA'].value
        lamb1 = params['lamb1'].value
        d1 = params['d1'].value
        mu1 = params['mu1'].value
    # simulation
    yinit = [0, 1, 0, 0]  # start from one active stem cell on day 0
    timevec = np.arange(0, 50)
    solution = odeint(tumour_ODE, yinit, timevec, args=(lambS, lambA, lamb1,
                                                        d1, mu1))
    sol = solution.T
    SC = np.sum(sol[0:2], axis=0)[-1]
    qSC = sol[0][-1]  # quiescent stem cells, not relevant in this model verion
    aSC = sol[1][-1]
    progeny = np.sum(sol[2:], axis=0)[-1]
    terminal = sol[3][-1]
    prolif = (0.17*sol[1]+sol[2])[-1]  # actively prolfierating cells
    total = np.sum(sol, axis=0)[-1]

    # experimentally determiend ratios from Zhu's thesis and paper, error
    # estimates likewise taken from there:
    # 1) fraction of actively proliferating GFP+ cells = 0.17
    # 2) fraction of GFP+ cells within all proliferating cells = 0.09
    # 3) fraction of overall proliferatively active cells in tumor = 0.22
    # 4) fraction of stem cells in the tumour, taken from FACS eperiment on
    #    dsRed labelled tumour = 0.23
    # GFP/prolif value from PCNA value in paper
    data_ratios = [0.09, 0.22, 0.23]
    data_errors = [0.02, 0.04, 0.03]

    # calculate relevant ratios from the model (2), 3) and 4))
    model_ratios = [0.17*aSC/prolif, prolif/total, aSC/total]

    # model_ratios = [aSC/total]
    # calculate residuals
    residuals = (np.array(data_ratios) - np.array(model_ratios)) / \
        np.array(data_errors)
    return residuals
