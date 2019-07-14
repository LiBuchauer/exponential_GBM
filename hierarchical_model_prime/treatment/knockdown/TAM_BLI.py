# python 3.5
# utf-8
""" Population growth under knock-down treatment, model
predictions are generated based on the assumption that TAM induces a given
fraction of stem cells to turn into progenitor cells with a given time delay
and using MCMC-generated parameter sets from bayes_prime or
bayes_prime_tracing. The position of these -h5 files with
parameter combinations has to be indicated in the code. """

import numpy as np
import matplotlib.pyplot as plt
import pylab
import seaborn
from scipy.integrate import odeint
import pandas as pd
from lmfit import *
try:
    reload
except NameError:
    from importlib import reload

import TAM_ODE
reload(TAM_ODE)
import TAM_data
reload(TAM_data)
from TAM_ODE import tumour_ODE
from TAM_data import time_analysis

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
lambA = 0.43  # asymmetric division rate of stem cells

beta = 0.36  # migration rate out of clone of active stem cells
lamb1 = 0.97  # division rate of progeny
d1 = 1.1  # differentiation rate of progeny
mu1 = 0.24  # death rate of exhausted cells

""" Parameters for Tlx-knockdown."""

eff = 1  # efficiency of TAM-recombination per single shot
t_delay = 5  # number of days needed for the gene switch to show full effect


def TAM_growth(t_treat, lambS, lambA, lamb1, d1, mu1,
               eff, t_delay, endtime=100, step=0.01):
    """ Given a timevector and a set of parameters, simulates the ODE system
    of undisturbed tumour growth until a specific day, then applies tamoxifen
    inducing knock-down in a fraction of stem cells given by efficiency
    on each day the treatment is given. The cells are not affected
    instantaneously, but after the given delay t_delay."""
    t_TAM = int(10)
    t_diff = endtime - t_treat - t_TAM - t_delay
    # simulation, undisturbed until treatment timepoint
    yinit = [0, 1, 0, 0]  # start from one active stem cell on day 0
    # the delay in the gene switch taking effect is added to time before
    # each administration as a means of delaying the administration's effect.
    timevec = np.arange(0, t_treat+t_delay+step, step)
    solution0 = odeint(tumour_ODE, yinit, timevec, args=(lambS, lambA,
                                                         lamb1, d1, mu1)).T
    # we want to set the cell number at the time of the first administration
    # to 1, thus we extract it here
    indx = len(np.arange(0, t_treat+step, step))
    normval = np.sum(solution0, axis=0)[indx]
    # on each day the knock-out is applied, take fraction efficiency of the
    # stem cells (hiding from TMZ and normal) and turn them into progeny cells
    for i in range(t_TAM):
        # construct new initial conditions from last solution
        end = solution0[:, -1]
        yinit = [end[0]*(1-eff), end[1]*(1-eff),
                 end[2]+(end[0]+end[1])*eff, end[3]]
        timevec = np.arange(0, 1+step, step)
        new_sol = odeint(tumour_ODE, yinit, timevec, args=(lambS, lambA,
                                                           lamb1, d1, mu1)).T
        solution0 = np.hstack((solution0[:, :-1], new_sol))

    # after this, run normal system until endtime without further changes
    yinit = solution0[:, -1]
    timevec = np.arange(0, t_diff+step, step)
    new_sol = odeint(tumour_ODE, yinit, timevec, args=(lambS, lambA,
                                                       lamb1, d1, mu1)).T

    solution = np.hstack((solution0[:, :-1], new_sol))
    # normalise solution to first treatment timepoint
    solution = solution/normval

    return solution


def BLI_treatment_plot():
    """ Plots the simulated growth as absolute and subpopulation fractional
    plots.. """
    # rounding wrapper
    def r3(num):
        return np.round(num, 3)
    def r1(num):
        return np.round(num, 1)

    # treatment timepoint, endtime, simulation timestep
    t_treat = 40
    endtime = 100
    step = 0.01

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
            # eff = df['eff'][i]
            # run simulation for this parameter set
            solution = TAM_growth(t_treat, lambS, lambA,
                                  lamb1, d1, mu1, eff, t_delay, endtime, step)

            solution_sum = np.sum(solution, axis=0)
            # store the result into the prepared array
            results5.append(solution_sum)

    # for all timepoints, find upper and lower bounds
    L_5 = np.min(np.array(results5), axis=0)
    U_5 = np.max(np.array(results5), axis=0)

    # get simulation from overall best found parameter set
    lambS = dfMAP['lambS'][0]
    lambA = dfMAP['lambA'][0]
    lamb1 = dfMAP['lamb1'][0]
    d1 = dfMAP['d1'][0]
    mu1 = dfMAP['mu1'][0]
    # eff = dfMAP['eff'][0]

    MAPsolution = TAM_growth(t_treat, lambS, lambA,
                             lamb1, d1, mu1, eff, t_delay, endtime, step)

    timevec = np.arange(0, endtime+step, step)
    MAPsolsum = np.sum(MAPsolution, axis=0)


    # plot together
    fig, axes = plt.subplots(1, 1, sharex=True, figsize=(4.5, 4))

    axes.plot(timevec, MAPsolsum, color='black', label=r'model, MAP')
    axes.fill_between(timevec, L_5, U_5, facecolor='slategrey',
                     label=r'model, $\alpha$=0.05 bounds', zorder=0,
                     interpolate=True)
    axes.set_yscale('log')
    axes.set_ylim([5*1e-3, 40])
    axes.set_ylabel('normalised tumour size')
    axes.set_xlim([32, 75])
    axes.axvline(t_treat, ls='--', color='black', label='TMZ treatment window')
    axes.axvline(t_treat+10, ls='--', color='black')
    h, l = axes.get_legend_handles_labels()
    h2 = [h[0], h[2], h[1]]
    l2 = [l[0], l[2], l[1]]
    legend = axes.legend(h2, l2, loc='lower left', frameon=True)
    legend.get_frame().set_facecolor('white')
    axes.set_xlabel('time (days)')

    plt.tight_layout()
    pylab.savefig('figures/TAM_treatment_model.pdf', bbox_inches='tight')

    # normalised solution
    fig, axes = plt.subplots(1, 1, sharex=True, figsize=(4.5, 4))
    solN = MAPsolution/MAPsolution.sum(axis=0)
    stack = axes.stackplot(timevec, solN, colors=cmap, labels = ['', 'Stem cells', 'Proliferating progeny', 'Terminally differentiated'])
    axes.set_ylabel('fraction per subpopulation')
    axes.axvline(t_treat, ls='--', color='black', label='TAM treatment window')
    axes.axvline(t_treat+10, ls='--', color='black')
    h, l = axes.get_legend_handles_labels()
    h2 = [h[1], h[2], h[3], h[0]]
    l2 = [l[1], l[2], l[3], l[0]]
    legend = axes.legend(h2, l2, frameon=True, loc='upper left')
    legend.get_frame().set_facecolor('white')
    axes.set_ylim([0, 1])
    axes.set_xlabel('time (days)')
    axes.set_xlim([32, 75])
    plt.tight_layout()
    pylab.savefig('figures/TAM_treatment_stack.pdf', bbox_inches='tight')


def r3(num):
    return np.round(num, 3)
