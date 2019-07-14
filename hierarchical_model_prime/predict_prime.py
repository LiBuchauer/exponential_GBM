# python 3.5
# utf-8

""" Imports a presaved set of MCMC sampled parameters, uses these to simulate
all required values and experiments and produces plots for them."""

import numpy as np
import matplotlib.pyplot as plt
import pylab
import seaborn
from scipy.integrate import odeint
import math
import pandas as pd
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


def bounds_Ki67(
        filepath,
        plottype='individual'):
    """ Given a filepath, imports the MCMC results from the .h5 file found
    there as well as the MAP parameter set contained separately. Simulates
    the Ki67 series for all parameter values found and identifies 0.68 and
    0.95 confidence regions which are then plotted together with the data and
    the MAP prediction.
    The additional keyword plottype decides whether the mean positive data
    fraction per day is plotted ('mean') togetehr with an error bar derived
    from the error model, or data points should be plotted individually."""
    # import dataframes
    df = pd.read_hdf(filepath, key='MCMC')
    dfMAP = pd.read_hdf(filepath, key='MLE')
    # best Log-likelihood
    best = dfMAP['lnPosterior'][0]
    # simulate all series and store results
    timevec = np.arange(0, 21, 0.1)
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
            # run simulation with these paramters
            this_result = Ki67_series(timevec=timevec, lambS=lambS, lambA=lambA,
                                      lamb1=lamb1, d1=d1, mu1=mu1)
            # store the result into the prepared array
            results5.append(this_result)

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

    MAP_model = Ki67_series(timevec=timevec, lambS=lambS, lambA=lambA,
                            lamb1=lamb1, d1=d1, mu1=mu1)
    # data values (means)
    data_times = [1, 3, 6, 7, 9, 12, 13]
    multiplicity = [2, 1, 3, 1, 2, 1, 3]  # animals used per tp
    data_vals = [0.4945, 0.2644, 0.1765, 0.0991, 0.1414, 0.1615, 0.1211]
    # error model: estimate of std from largest std on days with 3 animals,
    # division by sqrt(N) to get SEM estimate
    data_errors = [0.06/(math.sqrt(multiplicity[i])) for i in
                   range(len(data_vals))]

    # individual data points instead
    data_times_ind = [1, 1, 3, 6, 6, 6, 7, 9, 9, 12, 13, 13, 13]
    data_vals_ind = [0.5917, 0.3973, 0.2644, 0.1027, 0.2381, 0.1888, 0.0991,
                     0.1516, 0.1311, 0.1615, 0.2015, 0.068, 0.094]


    # plot
    plt.figure(figsize=(8, 5))
    plt.xlabel('time between dsRed and Ki67 (days)')
    plt.ylabel(r'(Ki67$^{+}$ dsRed$^{+}$ cells)/(dsRed$^{+}$ cells)')

    if plottype == 'mean':
        plt.errorbar(data_times, data_vals, yerr=data_errors, fmt='o',
                     label='experiment with SEM estimate', color='firebrick', zorder=10)
    elif plottype == 'individual':
        plt.plot(data_times_ind, data_vals_ind, 'o',
                 label='experiment, individual mice', color='firebrick', zorder=10)
    plt.plot(timevec, MAP_model, '-', label='model, MAP', color='black',
             zorder=2)
    plt.fill_between(timevec, L_5, U_5, facecolor='slategrey',
                     label=r'model, $\alpha$=0.05 bounds', zorder=0,
                     interpolate=True)

    plt.xlim([0, 20])
    plt.legend(loc=0)
    seaborn.despine()
    if plottype == 'mean':
        pylab.savefig('figures/Ki67_bounds.pdf', bbox_inches='tight')
    elif plottype == 'individual':
        pylab.savefig('figures/Ki67_bounds_individual.pdf', bbox_inches='tight')


def Ki67_series(timevec, lambS, lambA, lamb1, d1, mu1):
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
    # start of experiment at t1
    t1 = 30
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

    return Ki67frac


def bounds_medley(filepath='figures/chain_200_50000.h5'):
    """ Given a filepath, imports the MCMC results from the .h5 file found
    there as well as the MAP parameter set contained separately. Simulates
    tumour growth for all parameter values found and identifies
    0.95 confidence regions which are then plotted together with the data and
    the MAP prediction for a number of ratios of tumour composition as well
    as the effective overall growth rate. """
    # import dataframes
    df = pd.read_hdf(filepath, key='MCMC')
    dfMAP = pd.read_hdf(filepath, key='MLE')
    # simulate all fractions and store results, define order of things here
    items = ['stem cells',
             'proliferating\n progeny',
             'terminally\n differentiated\n cells',
             'actively\n proliferating\n cells',
             'proliferating SCs\n within all\n proliferating cells',
             r'$\lambda_{{\mathrm{{eff}}}}$']

    # best Log-likelihood
    best = dfMAP['lnPosterior'][0]
    # result collection for 5%maxlikelihood
    results = [[] for i in range(6)]
    for i in range(len(df)):
        # check if this set is in the 5% range, else pass
        if df['lnPosterior'][i] >= best+np.log(0.05):
            # get parameters from dataframe
            lambS = df['lambS'][i]
            lambA = df['lambA'][i]
            lamb1 = df['lamb1'][i]
            d1 = df['d1'][i]
            mu1 = df['mu1'][i]
            # run simulation with these paramters
            sol = odeint(tumour_ODE, [0, 1, 0, 0], np.arange(30),
                         args=(lambS, lambA, lamb1, d1, mu1)).T
            # store the result into the prepared array
            cellsum = np.sum(sol, axis=0)[-1]
            results[0].append((sol[0] + sol[1])[-1]/cellsum) # stem cells
            results[1].append(sol[2][-1]/cellsum)  # prolif. progeny
            results[2].append(sol[3][-1]/cellsum)  # exhausted cells
            results[3].append((0.17*sol[1] + sol[2])[-1]/cellsum)  # proliferating
            results[4].append(0.17*sol[1][-1]/((0.17*sol[1] + sol[2])[-1]))  # pro. SC
            results[5].append(lambS)
    # for all information, find upper and lower bounds of 95 %
    # confidence regions
    L_5 = np.min(np.array(results), axis=1)
    U_5 = np.max(np.array(results), axis=1)

    # get simulation from overall best found parameter set
    lambS = dfMAP['lambS'][0]
    lambA = dfMAP['lambA'][0]
    lamb1 = dfMAP['lamb1'][0]
    d1 = dfMAP['d1'][0]
    mu1 = dfMAP['mu1'][0]
    # run simulation with these paramters
    sol = odeint(tumour_ODE, [0, 1, 0, 0], np.arange(30),
                 args=(lambS, lambA, lamb1, d1, mu1)).T
    # store the results for plotting
    MAP_result = np.empty(len(items))
    cellsum = np.sum(sol, axis=0)[-1]
    MAP_result[0] = (sol[0] + sol[1])[-1]/cellsum  # stem cells
    MAP_result[1] = sol[2][-1]/cellsum  # prolif. progeny
    MAP_result[2] = sol[3][-1]/cellsum  # exhausted cells
    MAP_result[3] = (0.17*sol[1] + sol[2])[-1]/cellsum  # proliferating
    MAP_result[4] = 0.17*sol[1][-1]/((0.17*sol[1] + sol[2])[-1])  # pro. SC
    MAP_result[5] = lambS
    print(MAP_result)
    # data values (means)
    data_vals = [0.23, 0, 0, 0.22, 0.09, 0.21]
    data_errors = [0.04, 0, 0, 0.04, 0.02, 0.02]

    # plot all this as bar charts
    N = len(items)
    ind = np.arange(N)  # x loactions for the bars
    ind[-1] += 1
    width = 0.4  # width of the bars

    fig, ax = plt.subplots(figsize=(11, 5))
    # different axis for lambda_eff, cut x axis into two
    ax2 = ax.twinx()
    modbars = ax.bar(ind[:-1], MAP_result[:-1], width, color='slategrey',
                     yerr=[MAP_result[:-1]-L_5[:-1], U_5[:-1]-MAP_result[:-1]])
    databars = ax.bar(ind[:-1]+width, data_vals[:-1], width, color='firebrick',
                      yerr=data_errors[:-1])

    # add n.a. label to missinh experimental values
    for rect in databars[1:3]:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width()/2.0, height, 'n.a.',
            ha='center', va='bottom')

    modbars = ax2.bar(ind[-1], MAP_result[-1], width, color='slategrey',
                      yerr=[[MAP_result[-1]-L_5[-1]], [U_5[-1]-MAP_result[-1]]])
    databars = ax2.bar(ind[-1]+width, data_vals[-1], width, color='firebrick',
                       yerr=data_errors[-1])

    ax.set_ylabel('fraction of tumour')
    ax.set_xticks(ind+width/2)
    ax.set_xticklabels(items)
    ax2.legend((modbars[0], databars[0]), (r'model, MAP with $\alpha$=0.05 bounds',
                                           'experiment with SEM estimate'),
               bbox_to_anchor=(0.85, 1))

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax2.set_ylabel('rate (1/day)')
    ax2.spines['top'].set_visible(False)
    ax2.spines['left'].set_visible(False)

    # add n.a. label to missinh experimental values
    for rect in databars[1:3]:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width()/2.0, height, 'n.a.',
            ha='center', va='bottom')




    # optical divide between the groups of bars
    ax2.plot(ind[-1]-1+width/2, 0, 'ro', markersize=20, clip_on=False,
             zorder=500, color='white')

    pylab.savefig('figures/rest_bounds.pdf', bbox_inches='tight')
