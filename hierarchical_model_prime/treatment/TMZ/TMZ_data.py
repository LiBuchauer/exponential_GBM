# python 3.5
# -*- coding: utf-8 -*-
""" Bioluminescence data from TMZ treated glioblastoma  """

import numpy as np
import matplotlib.pyplot as plt
import pylab
import seaborn
from scipy.interpolate import interp1d

plt.rc('text', usetex=True)
seaborn.set_context('talk')
seaborn.set_style('ticks')
pylab.ion()


""" data
structure: (animal identifier, [timepoints], [raw BLI measurement data],
treatment day of first treatment only)
"""

# TMZ-treatment data
TMZ_data = []

TMZ_data.append(('A1965', [32, 36, 39, 42, 46, 50, 53, 57, 60, 63, 67, 70, 74,
                           77, 81],
                [1.04E+10, 2.57E+10, 2.79E+10, 3.15E+10, 7.00E+10, 3.39E+10,
                2.00E+10, 9.11E+09, 7.90E+09, 4.94E+09, 6.89E+09, 1.14E+10,
                1.68E+10, 2.89E+10, 1.79E+11], 50))

TMZ_data.append(('A1969', [32, 36, 39, 42, 46, 57, 60, 63, 66, 68, 70, 74, 77,
                           81, 84, 88, 91, 95, 98, 102, 105, 109, 112],
                [1.18E+08, 9.20E+08, 2.27E+09, 1.72E+10, 1.36E+10, 4.21E+10,
                 5.71E+10, 1.43E+10, 3.66E+09, 1.75E+09, 1.55E+09, 1.27E+09,
                 2.28E+09, 5.97E+08, 7.83E+08, 7.97E+08, 5.40E+08, 1.05E+09,
                 9.89E+08, 3.05E+09, 5.89E+09, 1.83E+10, 8.53E+10], 57))

TMZ_data.append(('A1970', [32, 36, 39, 42, 46, 53, 57, 60, 63, 67, 70, 74, 77,
                           81, 84, 88, 91, 95],
                 [4.81E+08, 1.79E+09, 1.28E+09, 2.34E+09, 1.79E+10, 6.49E+10,
                 3.00E+10, 2.19E+10, 2.40E+10, 1.09E+10, 6.91E+09, 7.71E+09,
                 3.32E+09, 1.87E+09, 4.15E+09, 1.51E+10, 3.54E+10, 6.34E+10],
                 53))

TMZ_data.append(('A1972', [32, 36, 39, 42, 46, 50, 53, 57, 60, 63, 67, 70, 74,
                           77, 81, 84, 88, 91],
                [1.83E+09, 2.75E+09, 2.91E+09, 6.13E+09, 4.70E+10, 5.54E+10,
                 5.95E+10, 3.83E+10, 1.02E+10, 5.82E+09, 3.41E+09, 3.96E+09,
                 2.30E+09, 1.82E+09, 5.67E+09, 2.26E+10, 1.71E+11, 2.40E+11],
                 50))

TMZ_data.append(('A8944', [32, 35, 39, 42, 46, 49, 53, 56, 60, 64, 68, 72, 75,
                           78, 81, 84, 88, 91, 102, 105],
                [9.37E+07, 6.25E+07, 1.39E+08, 3.29E+08, 4.33E+08, 2.89E+09,
                 7.36E+09, 1.27E+10, 2.83E+10, 3.50E+09, 2.22E+09, 1.05E+09,
                 4.28E+08, 5.47E+08, 2.52E+08, 3.16E+08, 3.77E+08, 1.37E+09,
                 4.23E+10, 5.24E+10], 60))


def interpol(data_time, data_vals):
    """ Interpolates the BLI data in the log space, retransforms into
    BLI unit space and returns the time grid as well as the interpolated
    values. """
    # form fine time vector in data range
    intertime = np.arange(data_time[0], data_time[-1], 0.1)
    # interpolation function
    f_interp = interp1d(data_time, np.log(data_vals), kind='quadratic',
                        bounds_error=False)
    # get interpolated data and transform back
    interdata = np.exp(f_interp(intertime))
    return intertime, interdata


def time_analysis(timevec, datavec, treatment):
    """ Given a time vector and a data vector for these times together with the
    treatment timepoint, finds the time point of lowest BLI value and the time
    point at which BLI relapsed to exactly treatment value. Returns these
    time points together with their BLI values."""
    def nearest_indx(array, value):
        return (np.abs(array-value)).argmin()
    def r3(stuff):
        return np.round(stuff, 3)
    # make input into lists
    timevec = list(r3(timevec))
    datavec = list(datavec)
    # find treatment timepoint' index
    tr_indx = timevec.index(treatment)
    # find lowest BLI value AFTER TREATMENT and its index
    BLI_low = np.min(datavec[tr_indx:])
    low_indx = datavec.index(BLI_low)
    # get time point
    t_low = timevec[low_indx]
    # find index of BLI value closest to treatment BLI value AFTER LOWEST POINT
    re_indx = nearest_indx(datavec[low_indx:], datavec[tr_indx])
    # get timepoint
    t_relapse = timevec[low_indx+re_indx]
    BLI_relapse = datavec[low_indx+re_indx]
    return (t_low, BLI_low), (t_relapse, BLI_relapse)


def TMZ_plot():
    """ Plots raw data as well as interpolation and positions of minimum and
    relapse to pre-treatment-value as identified,

    1st version with big ugly colored spheres."""
    def r1(stuff):
        return np.round(stuff, 1)

    fig, axes = plt.subplots(2, 3, figsize=(9, 6), sharex=True)
    # collect decay and relapse times
    t_min = []
    t_relapse = []
    reduction = []

    # plot raw data
    for mouse, ax in zip(TMZ_data, [axes[0][0], axes[0][1], axes[0][2],
                                    axes[1][0], axes[1][1], axes[1][2]]):
        ax.plot(mouse[1], mouse[2], '-o', label='data', color='firebrick')
        intertime, interdata = interpol(mouse[1], mouse[2])
        ax.plot(intertime, interdata, '--', color='black',
                label='interpolation')
        ax.set_yscale('log')
        if ax in [axes[1][0], axes[1][1], axes[0][2]]:
            ax.set_xlabel('time (days)')
        if ax in [axes[0][0], axes[1][0]]:
            ax.set_ylabel('BLI (a.u.)')
        # plot special timepoints
        ax.plot(mouse[3], mouse[2][mouse[1].index(mouse[3])], 'o',
                color='slategrey', ms=20, label='start of TMZ administration')
        low, relapse = time_analysis(intertime, interdata, mouse[3])
        ax.plot(low[0], low[1], 'o', color='cornflowerblue', ms=20,
                label='smallest size')
        ax.plot(relapse[0], relapse[1], 'o', color='lightcoral', ms=20,
                label='relapse to pre-TMZ size')
        ax.set_title('$t_m$ = {}, $t_r$ = {},\n $F$ = {}'.format(
            r1(low[0]-mouse[3]), r1(relapse[0]-mouse[3]), r1(relapse[1]/low[1])))
        t_min.append(low[0]-mouse[3])
        t_relapse.append(relapse[0]-mouse[3])
        reduction.append(relapse[1]/low[1])
    axes[0][0].legend(bbox_to_anchor=(3.9, -0.5))
    axes[1][2].axis('off')
    plt.tight_layout()
    pylab.savefig('figures/TMZ_data.pdf', bbox_inches='tight')


def TMZ_plot2():
    """ Plots raw data as well as interpolation and indicates during which
    timeframe treatment was given,

    2nd version of plot without spheres."""
    def r1(stuff):
        return np.round(stuff, 1)

    fig, axes = plt.subplots(2, 3, figsize=(9, 6), sharex=False)

    # plot raw data
    for mouse, ax in zip(TMZ_data, [axes[0][0], axes[0][1], axes[0][2],
                                    axes[1][0], axes[1][1], axes[1][2]]):
        ax.plot(mouse[1], mouse[2], '-o', label='data', color='firebrick')
        intertime, interdata = interpol(mouse[1], mouse[2])
        ax.plot(intertime, interdata, '--', color='black',
                label='interpolation')
        ax.set_yscale('log')
        if ax in [axes[1][0], axes[1][1], axes[0][2]]:
            ax.set_xlabel('time (days)')
        if ax in [axes[0][0], axes[1][0]]:
            ax.set_ylabel('BLI (a.u.)')

        # treatment time window
        ax.axvline(mouse[3], ls='-', color='slategrey',
            label='daily TMZ administrations', zorder=0)
        for i in range(1, 10):
            ax.axvline(mouse[3]+i, ls='-', color='slategrey', zorder=0)

    axes[0][0].legend(bbox_to_anchor=(3.9, -0.5))
    axes[1][2].axis('off')

    plt.tight_layout()
    pylab.savefig('figures/TMZ_data_tw.pdf', bbox_inches='tight')
