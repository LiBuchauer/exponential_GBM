# python 3.5
# -*- coding: utf-8 -*-
""" Bioluminescence data from TMZ-treated mice, fitting the growth rate of the
post treatment slope in order to determine whether growth there is faster than
during the primaray. """

import numpy as np
import matplotlib.pyplot as plt
import pylab
import seaborn
from lmfit import *

plt.rc('text', usetex=True)
seaborn.set_context('talk')
seaborn.set_style('ticks')
pylab.ion()

""" starting values for fitting """
# exponential model
l0s = 0.3  # growth rate

# start value
N0s = 3*10000

# relative error
error_s = 0.33

""" data
structure: (animal identifier, [timepoints], [raw BLI measurement data])
"""

# Showing here only data after the lowest point has been passed as the goal is
# to fit the growth rates of the relapse
data = []

data.append(('A1965', np.array([63, 67, 70, 74, 77, 81]),
                [4.94E+09, 6.89E+09, 1.14E+10,
                1.68E+10, 2.89E+10, 1.79E+11]))

data.append(('A1969', np.array([98, 102, 105, 109, 112]),
                [9.89E+08, 3.05E+09, 5.89E+09, 1.83E+10, 8.53E+10]))

data.append(('A1970', np.array([81, 84, 88, 91, 95]),
                 [1.87E+09, 4.15E+09, 1.51E+10, 3.54E+10, 6.34E+10]))

data.append(('A1972', np.array([77, 81, 84, 88, 91]),
                [1.82E+09, 5.67E+09, 2.26E+10, 1.71E+11, 2.40E+11]))

data.append(('A8944', np.array([81, 84, 88, 91, 102, 105]),
                [2.52E+08, 3.16E+08, 3.77E+08, 1.37E+09,
                 4.23E+10, 5.24E+10]))


""" Exponential growth with rate l0. """


def exp_residual(params, data_time, data_vals, errors):
    """ Calculates residuals for exponential growth model given the input of
    data time series and parameter object. """
    # extract parameters
    l0 = params['l0'].value
    N0 = params['N0'].value

    # calculate model values at data timepoints
    model_vals = N0 * np.exp(l0 * data_time)

    # calculate residuals, normalise by errors
    residuals = (data_vals - model_vals) / errors

    return residuals


def fit_exp(data_time, data_vals, error=error_s, l0=l0s, N0=N0s):
    """ Fits exponential growth curve to data, returns best fit parameters
    and chi2, red chi2, AIC. """
    # initialise parameter class and add relevant parameters
    params = Parameters()
    params.add('l0', value=l0, vary=1, min=0, max=5)
    params.add('N0', value=N0, vary=1, min=0)

    # make sure data and data_times are np.arrays
    data_time = np.array(data_time)
    data_vals = np.array(data_vals)
    errors = data_vals * error
    # run minimisation
    fit = minimize(exp_residual, params, args=(data_time, data_vals, errors),
                   method='leastsq')

    # return fit results and goodness of fit info
    l0 = fit.params['l0'].value
    l0e = fit.params['l0'].stderr
    N0 = fit.params['N0'].value
    N0e = fit.params['N0'].stderr
    chi2 = fit.chisqr
    chi2red = fit.redchi
    AIC = fit.aic

    return l0, l0e, N0, N0e, chi2, chi2red, AIC


def grid_plot_data():
    """ Grid with individual plot of every curve, data only. """
    fig, axes = plt.subplots(2, 3, sharex=True, figsize=(15, 10))

    for i in range(len(data)):
        ax = axes.reshape(-1)[i]
        ax.set_yscale('log')
        ax.set_xlim([20, 70])
        ax.plot(data[i][1], data[i][2], 'o', color='firebrick')
        ax.set_title(data[i][0])
    fig.tight_layout()
    pylab.savefig('figures/BLI_individually.pdf', bbox_inches='tight')


def grid_plot_fit():
    """ Plots grid with all animals and a fit according to the chosen type
    onto each of them. Fit results are also supplied. """

    fig, axes = plt.subplots(1, 5, sharex=False, figsize=(12.5, 3.3))
    for i in range(len(data)):
        # plot the data
        ax = axes.reshape(-1)[i]
        ax.set_yscale('log')
        # ax.set_xlim([0, 70])
        ax.errorbar(data[i][1], data[i][2], yerr=error_s*np.array(data[i][2]),
                    fmt='o', color='firebrick')
        mod_times = np.arange(data[i][1][0]-5, data[i][1][-1]+5)

        # run the fit for this dataset
        l0, l0e, N0, N0e, chi2, chi2red, AIC = \
            fit_exp(data[i][1], data[i][2], error=error_s, l0=l0s, N0=N0s)
        model_vals = N0 * np.exp(l0 * mod_times)
        ax.plot(mod_times, model_vals, '-', color='black')
        ax.set_title(r'$\lambda_0$ = {} $\pm$ {}'.format(
            np.round(l0, 2), np.round(l0e, 2)))
        ax.annotate(r'$\chi^2_r$ = {}'.format(np.round(chi2red, 1)),
                    xy=(0.45, 0.1), xycoords='axes fraction')
        ax.set_xlabel('time (days)')
    axes[0].set_ylabel('BLI (a.u.)')
    fig.tight_layout()
    pylab.savefig('figures/TMZ_relapse_fits.pdf',
                  bbox_inches='tight')
