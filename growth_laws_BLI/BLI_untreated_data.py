# python 3.5
# -*- coding: utf-8 -*-
""" Bioluminescence data from untreated glioblastoma - fitting of exponential,
Gompertzian and linear radial models and plotting of results with fit
statistics. """

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
N0s = 3*10000  # cell number

# Gompertz model
g0s = 0.05  # growth rate
K0s = 10**14  # capacity
N0s = 3*1000  # cell number

# power law model
r0s = 10  # linear expansion rate
R0s = 10000  # starting radius

# estimated relative error of data (Gaussian noise with std given relative
# to abs size)
error_s = 0.33

""" data
structure: (animal identifier, [timepoints], [raw BLI measurement data])
"""

# No treatment data
data = []

data.append(('A5449', [38, 41, 44, 47, 51, 54, 58, 61], [2.69E+08,	2.38E+09,
             3.89E+09, 7.22E+09,	1.28E+10,	1.03E+10,	3.06E+10,
             3.95E+10]))

data.append(('A5451', [38, 41, 44, 47, 51, 54, 58, 61], [1.45E+08,	1.80E+09,
             4.97E+09,	1.12E+10, 1.80E+10,	2.64E+10, 2.84E+10,	5.18E+10]))

data.append(('A5452', [38, 41, 44, 47, 51, 54], [1.83E+08,	7.42E+08,
             1.87E+09, 6.22E+09,	2.12E+10,	3.74E+10]))

data.append(('A5469', [34, 37, 40, 43, 46, 49, 52, 56, 59, 63], [3.39E+08,
             6.47E+08,	1.29E+09,	2.55E+09,	2.86E+09,	4.48E+09, 1.27E+10,
             1.66E+10,	2.04E+10,	1.50E+10]))

data.append(('A5471', [34, 37, 40, 43, 46, 49, 52, 56, 59, 63, 66], [5.23E+08,
             6.69E+08,	1.38E+09,	2.20E+09,	3.72E+09,	2.95E+09,
             6.85E+09,	1.48E+10,	2.78E+10,	4.39E+10,	1.03E+11]))

data.append(('A5472', [34, 37, 40, 43, 46, 49], [4.76E+08,	1.38E+09,
             8.11E+09,	6.87E+09,	2.75E+10,	1.48E+10]))

data.append(('A5483', [34, 37, 40, 43], [4.10E+09,	1.05E+10,	3.44E+10,
             6.40E+10]))

data.append(('A15231', [37, 40, 43, 46, 49], [1.16E+09, 4.46E+09, 7.09E+09,
             8.64E+09, 1.73E+11]))

data.append(('A15232', [37, 40, 43, 46, 49, 52, 56], [3.18E+09,	5.88E+09,
             6.57E+09,	1.52E+10,	3.52E+10,	1.86E+11,	1.94E+11]))

data.append(('A15233', [37, 40, 43, 46, 49, 52, 56], [1.82E+08, 2.46E+08,
             3.35E+08,	1.30E+09,	1.67E+09,	3.42E+09,	1.24E+10]))

data.append(('A33362', [22, 27,	33,	37,	41,	44,	48,	51,	55],
             [1.68E+09,	3.21E+09,	4.74E+09,	1.24E+10,	2.00E+10,
             5.02E+10,	7.04E+10,	8.95E+10,	6.93E+10]))

data.append(('A33364', [22, 27,	33,	37,	41,	44,	48,	51,	55,	58,	62,	65,	69],
             [7.05E+08,	5.82E+08,	6.79E+08,	6.99E+08,	9.38E+08,
             2.08E+09,	3.60E+09,	8.09E+09,	1.55E+10,	2.66E+10,
             6.73E+10,	7.89E+10,	7.11E+10]))

data.append(('A33365', [22, 27,	33,	37,	41,	44,	48,	51,	55,	58,	62], [3.38E+08,
             3.46E+08,	4.87E+08,	7.64E+08,	1.11E+09,	2.97E+09,
             2.59E+10,	2.21E+10,	4.37E+10,	5.25E+10,	8.20E+10]))

data.append(('A33367', [22, 27,	33,	37,	41,	44,	48], [3.71E+08,	4.02E+08,
             5.83E+09,	1.66E+10,	3.54E+10,	5.92E+10,	3.35E+10]))

data.append(('A33369', [22, 27,	33,	37,	41], [4.84E+08,	4.53E+08,	6.58E+08,
             1.38E+09,	1.30E+10]))

data.append(('A33370', [22, 27,	33,	37,	41,	44,	48,	51,	55], [3.40E+08,
             6.81E+08,	1.42E+09,	6.16E+09,	1.30E+10,	3.47E+10,
             3.60E+10,	1.13E+11,	1.99E+11]))

# Cre- TAM control data, also considered as untreated data
data.append(('A8367', [28, 31, 35, 38, 42, 45], [9.97E+08,	4.97E+09,
             2.18E+10,	4.64E+10,	9.37E+10,	1.62E+11]))

data.append(('A8954', [33, 36, 40, 43, 47, 50, 54, 57], [1.64E+10,	1.56E+10,
             4.48E+10,	9.50E+10,	2.58E+11,	2.15E+11,	3.37E+11,
             7.36E+11]))

data.append(('A8960', [30, 33, 37, 40, 44], [9.09E+08,	7.44E+08,	9.46E+08,
             7.51E+09,	2.26E+10]))

data.append(('A8969', [33, 36, 40, 43, 47, 50, 54, 57, 60], [3.91E+09,
             9.88E+09,	1.89E+10,	1.52E+11,	2.01E+11,	4.22E+11,
             7.32E+11,	5.56E+11,	1.07E+12]))

data.append(('A8970', [33, 36, 40, 43, 47, 50], [1.79E+09,	8.86E+09,
             7.33E+10,	1.18E+11,	1.87E+11,	2.44E+11]))

data.append(('A8976', [36, 40, 43, 47, 50, 54], [7.27E+10,	1.70E+11,
             2.01E+11,	8.59E+11,	4.87E+11,	9.14E+11]))

data.append(('A8980', [36, 40, 43, 47, 50, 54], [4.40E+09,	5.34E+09,
             5.53E+09,	2.03E+10,	2.85E+10,	3.45E+10]))


""" Model1: Exponential growth with rate l0. """


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

    # for logarithmic error model, not in use
    # data_time = np.array(data_time)
    # data_vals = np.log(np.array(data_vals))

    # errors
    errors = data_vals * error

    # run minimisation
    fit = minimize(exp_residual, params, args=(data_time, data_vals, errors),
                   method='nelder')

    # return fit results and goodness of fit info
    l0 = fit.params['l0'].value
    l0e = fit.params['l0'].stderr
    N0 = fit.params['N0'].value
    N0e = fit.params['N0'].stderr
    chi2 = fit.chisqr
    chi2red = fit.redchi
    AIC = fit.aic

    return l0, l0e, N0, N0e, chi2, chi2red, AIC


""" Model 2: Gompertzian growth with growth parameter g0 and capacity K0."""


def gomp_residual(params, data_time, data_vals, errors):
    """ Calculates residuals for Gompertz growth model given the input of
    data time series and parameter object. """
    # extract parameters
    g0 = params['g0'].value
    K0 = params['K0'].value
    N0 = params['N0'].value

    # calculate model values at data timepoints
    model_vals = K0 * np.exp(np.exp(-g0 * data_time) * np.log(N0 / K0))

    # calculate residuals, normalise by errors
    residuals = (data_vals - model_vals) / errors

    return residuals


def fit_gomp(data_time, data_vals, error=error_s, l0=l0s, K0=K0s, N0=N0s):
    """ Fits exponential growth curve to data, returns best fit parameters
    and chi2, red chi2, AIC. """
    # initialise parameter class and add relevant parameters
    params = Parameters()
    params.add('g0', value=l0, vary=1, min=0, max=0.1)
    params.add('K0', value=K0, vary=1, min=1, max=10**15)
    params.add('N0', value=N0, vary=1, min=1)

    # make sure data and data_times are np.arrays
    data_time = np.array(data_time)
    data_vals = np.array(data_vals)
    errors = data_vals * error
    # run minimisation
    fit = minimize(gomp_residual, params, args=(data_time, data_vals, errors),
                   method='nelder')

    # return fit results and goodness of fit info
    g0 = fit.params['g0'].value
    g0e = fit.params['g0'].stderr
    N0 = fit.params['N0'].value
    N0e = fit.params['N0'].stderr
    K0 = fit.params['K0'].value
    K0e = fit.params['K0'].stderr
    chi2 = fit.chisqr
    chi2red = fit.redchi
    AIC = fit.aic

    return g0, g0e, K0, K0e, N0, N0e, chi2, chi2red, AIC


""" Model 3: Power law growth with radial growth rate l0."""


def power_residual(params, data_times, data_vals, errors):
    """ Calculates residuals for linear radial growth model given the input of
    data time series and parameter object. """
    # extract parameters
    r0 = params['r0'].value
    R0 = params['R0'].value

    # calculate model values at data timepoints
    model_vals = 4*np.pi/3*np.power(r0 * data_times + R0, 3)

    # calculate residuals, normalise by errors
    residuals = (data_vals - model_vals) / errors

    return residuals


def fit_power(data_time, data_vals, error=error_s, r0=r0s, R0=R0s):
    """ Fits exponential growth curve to data, returns best fit parameters
    and chi2, red chi2, AIC. """
    # initialise parameter class and add relevant parameters
    params = Parameters()
    params.add('r0', value=r0, vary=1, min=0)
    params.add('R0', value=R0, vary=1, min=0)

    # make sure data and data_times are np.arrays
    data_time = np.array(data_time)
    data_vals = np.array(data_vals)
    errors = data_vals * error
    # run minimisation
    fit = minimize(power_residual, params, args=(data_time, data_vals, errors),
                   method='nelder')

    # return fit results and goodness of fit info
    r0 = fit.params['r0'].value
    r0e = fit.params['r0'].stderr
    R0 = fit.params['R0'].value
    R0e = fit.params['R0'].stderr
    chi2 = fit.chisqr
    chi2red = fit.redchi
    AIC = fit.aic

    return r0, r0e, R0, R0e, chi2, chi2red, AIC


""" Fit all data sets and plot results """


def grid_plot_data():
    """ Large grid with individual plot of every curve, data only. """
    fig, axes = plt.subplots(5, 5, sharex=False, figsize=(28, 20))

    for i in range(len(data)):
        ax = axes.reshape(-1)[i]
        ax.set_yscale('log')
        ax.set_xlim([20, 70])
        ax.plot(data[i][1], data[i][2], 'o')
        ax.set_title(data[i][0])
    fig.tight_layout()
    pylab.savefig('figures/BLI_individually.pdf', bbox_inches='tight')


def grid_plot_fit(growth='exponential'):
    """ Plots grid with all animals and a fit according to the chosen type
    onto each of them. Fit results are also supplied. Possible types are
    'exponential', 'gompertz', 'power'."""

    fig, axes = plt.subplots(5, 5, sharex=False, figsize=(15, 16))
    for i in range(len(data)):
        # plot the data
        ax = axes.reshape(-1)[i]
        ax.set_yscale('log')
        ax.errorbar(data[i][1], data[i][2], yerr=error_s*np.array(data[i][2]),
                    fmt='o', color='firebrick')
        mod_times = np.arange(data[i][1][0]-5, data[i][1][-1]+5)
        # run the fit for this dataset according to the key
        if growth == 'exponential':
            l0, l0e, N0, N0e, chi2, chi2red, AIC = \
                fit_exp(data[i][1], data[i][2], error=error_s, l0=l0s, N0=N0s)
            model_vals = N0 * np.exp(l0 * mod_times)
            ax.plot(mod_times, model_vals, '-', color='black')
            ax.set_title(r'$\lambda_0$ = {}'.format(
                np.round(l0, 2), np.round(N0/10**4, 2)))
            ax.annotate(r'$\chi^2_r$ = {}'.format(np.round(chi2red, 1)),
                        xy=(0.45, 0.2), xycoords='axes fraction')
            ax.annotate('AIC = {}'.format(np.round(AIC, 1)),
                        xy=(0.45, 0.1), xycoords='axes fraction')

        elif growth == 'gompertz':
            g0, g0e, K0, K0e, N0, N0e, chi2, chi2red, AIC = \
                fit_gomp(data[i][1], data[i][2], error=error_s, l0=l0s, K0=K0s,
                        N0=N0s)
            model_vals = K0 * np.exp(np.exp(-g0 * mod_times) * np.log(N0 / K0))
            ax.plot(mod_times, model_vals, '-', color='black')
            ax.set_title(r'$g_0$ = {} \newline $K_0$ = {} $\cdot 10^{{12}}$'.format(
                np.round(g0, 2), np.round(K0/10**12, 2), np.round(N0/10**3, 2)))
            ax.annotate(r'$\chi^2_r$ = {}'.format(np.round(chi2red, 1)),
                        xy=(0.45, 0.2), xycoords='axes fraction')
            ax.annotate('AIC = {}'.format(np.round(AIC, 1)),
                        xy=(0.45, 0.1), xycoords='axes fraction')

        if growth == 'power':
            r0, r0e, R0, R0e, chi2, chi2red, AIC = \
                fit_power(data[i][1], data[i][2], error=error_s, r0=r0s, R0=R0s)
            model_vals = 4*np.pi/3*np.power(r0 * mod_times + R0, 3)
            ax.plot(mod_times, model_vals, '-', color='black')
            ax.set_title(r'$r_0$ = {}'.format(
                np.round(r0, 2), np.round(R0, 2)))
            ax.annotate(r'$\chi^2_r$ = {}'.format(np.round(chi2red, 1)),
                        xy=(0.45, 0.2), xycoords='axes fraction')
            ax.annotate('AIC = {}'.format(np.round(AIC, 1)),
                        xy=(0.45, 0.1), xycoords='axes fraction')
        if i in (18, 19, 20, 21, 22):
            ax.set_xlabel('time (days)')
        if i in (0, 5, 10, 15, 20):
            ax.set_ylabel('BLI (a.u.)')

    axes[4][3].axis('off')
    axes[4][4].axis('off')
    fig.tight_layout()
    pylab.savefig('figures/BLI_untreated_fits_{}.pdf'.format(growth),
                  bbox_inches='tight')
