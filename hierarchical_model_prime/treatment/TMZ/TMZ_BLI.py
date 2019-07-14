# python 3.5
# utf-8
""" Population growth and decline with chemotherapy and relapse, model
predictions are geenrated based on the assumption that cell divisions during
chemotherapy lead to death and using MCMC-generated parameter sets from
bayes_prime or baymes_prime_tracing. The position of these -h5 files with
parameter combinations has to be indicated in the code. """

import numpy as np
import matplotlib.pyplot as plt
import pylab
import seaborn
from scipy.integrate import odeint
import pandas as pd
try:
    reload
except NameError:
    from importlib import reload

import TMZ_ODE
reload(TMZ_ODE)
from TMZ_ODE import tumour_ODE, treatment_ODE
from TMZ_data import time_analysis

plt.rc('text', usetex=True)
seaborn.set_context('talk')
seaborn.set_style('ticks')
pylab.ion()
cmap = ['white', 'seagreen', 'indianred', 'lightgrey']
#
# ['lightgreen', 'forestgreen', 'yellow', 'tomato', 'firebrick',
#         'cornflowerblue', 'darkblue']


""" Parameters of glioblastoma growth (rates in 1/day),
fixed in this analysis and derived from undisturbed growth data."""

lambS = 0.21  # symmetric division rate of stem cells
lambA = 0.36  # asymmetric division rate of stem cells

beta = 0.1  # migration rate out of clone of active stem cells
lamb1 = 1.10  # division rate of progeny
d1 = 1.10  # differentiation rate of progeny
mu1 = 0.13  # death rate of exhausted cells

""" Parameters for reactivation and chemotherapy."""

xi_TMZ = 1  # TMZ induced death rate proportionality constant
mu_TMZ = 0  # TMZ induced death rate in exhausted compartment
t_TMZ = 10  # length of time window in which TMZ has effects


def BLI_treatment_plot(lambS=lambS, lambA=lambA,
                       lamb1=lamb1, d1=d1, mu1=mu1,
                       xi_TMZ=xi_TMZ, t_TMZ=t_TMZ, mu_TMZ=mu_TMZ):
    """ Plots the simulated growth as absolute and subpopulation fractional
    plots. """
    # rounding wrapper
    def r3(num):
        return np.round(num, 3)
    def r1(num):
        return np.round(num, 1)

    # treatment timepoint and length of observation after end of treatment
    t_treat = 55
    t_recover = 50

    # import dataframes with parameters
    df = pd.read_hdf('figures/chain_200_50000.h5', key='MCMC')
    dfMAP = pd.read_hdf('figures/chain_200_50000.h5', key='MLE')
    # best Log-likelihood
    best = dfMAP['lnPosterior'][0]
    # result collection for 5%maxlikelihood
    results5 = []
    for i in range(len(df)):
        # check if this set is in the 5% range, else pass
        if df['lnPosterior'][i] >= best+np.log(0.05):
            # get parameters from dataframe
            lambS = df['lambS'][i]
            lambA = df['lambA'][i]
            lamb1 = df['lamb1'][i]
            d1 = df['d1'][i]
            mu1 = df['mu1'][i]

            # simulation, undisturbed until treatment timepoint
            yinit = [0, 1, 0, 0]  # start from one active stem cell on day 0
            timevec = np.arange(0, t_treat, 0.1)
            solution0 = odeint(tumour_ODE, yinit, timevec, args=(lambS, lambA,
                                                                 lamb1, d1, mu1)).T
            # chemo-death during time window t_TMZ
            yinit = solution0[:, -1]
            timevec = np.arange(0, t_TMZ, 0.1)
            solution1 = odeint(treatment_ODE, yinit, timevec, args=(lambS, lambA,
                                                                    lamb1, d1, mu1,
                                                                    xi_TMZ, mu_TMZ)).T
            # end of chemo-presence, cells proliferate again and reactivation continues
            yinit = solution1[:, -1]
            timevec = np.arange(0, t_recover, 0.1)
            solution2 = odeint(tumour_ODE, yinit, timevec, args=(lambS, lambA,
                                                                 lamb1, d1, mu1)).T
            timevec = np.arange(0, t_treat+t_TMZ+t_recover, 0.1)
            solution = np.hstack((solution0[:, :], solution1[:, :], solution2[:, :]))

            # normalise by value at treatment time point
            normval = np.sum(solution0[:, -1])
            solution = solution/normval

            results5.append(np.sum(solution, axis=0))

    # for all timepoints, find upper and lower bounds of 68 and 95 % confidence
    # regions
    L_5 = np.min(np.array(results5), axis=0)
    U_5 = np.max(np.array(results5), axis=0)

    # get simulation from overall best found parameter set
    lambS = dfMAP['lambS'][0]
    lambA = dfMAP['lambA'][0]
    lamb1 = dfMAP['lamb1'][0]
    d1 = dfMAP['d1'][0]
    mu1 = dfMAP['mu1'][0]

    # simulation, undisturbed until treatment timepoint
    yinit = [0, 1, 0, 0]  # start from one active stem cell on day 0
    timevec = np.arange(0, t_treat, 0.1)
    solution0 = odeint(tumour_ODE, yinit, timevec, args=(lambS, lambA,
                                                         lamb1, d1, mu1)).T
    # chemo-death during time window t_TMZ
    yinit = solution0[:, -1]
    timevec = np.arange(0, t_TMZ, 0.1)
    solution1 = odeint(treatment_ODE, yinit, timevec, args=(lambS, lambA,
                                                            lamb1, d1, mu1,
                                                            xi_TMZ, mu_TMZ)).T
    # end of chemo-presence, cells proliferate again and reactivation continues
    yinit = solution1[:, -1]
    timevec = np.arange(0, t_recover, 0.1)
    solution2 = odeint(tumour_ODE, yinit, timevec, args=(lambS, lambA,
                                                         lamb1, d1, mu1)).T
    timevec = np.arange(0, t_treat+t_TMZ+t_recover, 0.1)
    MAPsolution = np.hstack((solution0[:, :], solution1[:, :], solution2[:, :]))
    MAPsolsum = np.sum(MAPsolution, axis=0)

    # normalise by value at treatment time point
    normval = np.sum(solution0[:, -1])
    MAPsolsum = MAPsolsum/normval

    # plot
    fig, axes = plt.subplots(1, 1, sharex=True, figsize=(4.5, 4))

    axes.plot(timevec, MAPsolsum, color='black', label=r'model, MAP')
    axes.fill_between(timevec, L_5, U_5, facecolor='slategrey',
                     label=r'model, $\alpha$=0.05 bounds', zorder=0,
                     interpolate=True)
    # treatment time window
    axes.axvline(t_treat, ls='--', color='black', label='TMZ treatment window')
    axes.axvline(t_treat+t_TMZ, ls='--', color='black')

    axes.set_yscale('log')
    axes.set_ylim([5*1e-3, 30])
    axes.set_ylabel('normalised tumour size')
    axes.set_xlim([32, 110])

    h, l = axes.get_legend_handles_labels()
    h2 = [h[0], h[2], h[1]]
    l2 = [l[0], l[2], l[1]]
    legend = axes.legend(h2, l2, loc='upper left', frameon=True)
    legend.get_frame().set_facecolor('white')
    axes.set_xlabel('time (days)')

    plt.tight_layout()
    pylab.savefig('figures/TMZ_treatment_model.pdf', bbox_inches='tight')

    # normalised solution
    fig, axes = plt.subplots(1, 1, sharex=True, figsize=(4.5, 4))
    solN = MAPsolution/MAPsolution.sum(axis=0)
    stack = axes.stackplot(timevec, solN, colors=cmap, labels = ['', 'Stem cells', 'Proliferating progeny', 'Terminally differentiated'])
    axes.set_ylabel('fractional tumour composition')
    axes.axvline(t_treat, ls='--', color='black', label='TMZ treatment window')
    axes.axvline(t_treat+t_TMZ, ls='--', color='black')
    h, l = axes.get_legend_handles_labels()
    h2 = [h[1], h[2], h[3], h[0]]
    l2 = [l[1], l[2], l[3], l[0]]
    legend = axes.legend(h2, l2, loc='upper left', frameon=True)
    legend.get_frame().set_facecolor('white')
    axes.set_ylim([0, 1])
    axes.set_xlabel('time (days)')
    axes.set_xlim([32, 110])
    plt.tight_layout()
    pylab.savefig('figures/TMZ_treatment_stack.pdf', bbox_inches='tight')
