# python 3.5
# -*- coding: utf-8 -*-
""" Bioluminescence data from Tlx-knockdown treated glioblastoma  """

import numpy as np
import matplotlib.pyplot as plt
import pylab
import seaborn
from scipy.interpolate import interp1d
from scipy.signal import argrelextrema

plt.rc('text', usetex=True)
seaborn.set_context('talk')
seaborn.set_style('ticks')
pylab.ion()


""" data
structure: (animal identifier, [timepoints], [raw BLI measurement data],
treatment day of first treatment only)
"""

# TAM-treatment data
TAM_data = []

TAM_data.append(('A8343',
                 [28, 31, 35, 42, 45, 49, 51, 56, 59, 63, 66, 70, 73, 77, 80,
                  84, 87, 91],
                 [3.80E+08, 9.57E+08, 2.91E+09, 1.72E+10, 1.40E+10, 9.52E+09,
                  9.74E+09, 2.34E+10, 3.43E+10, 4.06E+10, 4.14E+10, 2.89E+10,
                  7.53E+09, 4.35E+09, 2.29E+10, 2.55E+10, 3.54E+10,	7.36E+10],
                 42))

TAM_data.append(('A8366',
                 [28, 31, 35, 38, 42, 45, 49, 51, 56, 59, 63],
                 [6.23E+08, 4.98E+08, 9.80E+08, 5.05E+09, 1.44E+10, 1.09E+10,
                  9.09E+09, 1.08E+10, 1.99E+10, 1.91E+10, 1.35E+10],
                 42))

TAM_data.append(('A8646',
                 [39, 43, 46, 50, 53, 57, 60, 64, 67, 71, 74, 78, 82, 86, 90,
                  93, 96, 99, 102, 106, 109, 120, 123, 127, 130, 134, 141,
                  148],
                 [1.43E+09, 4.86E+09, 7.29E+09, 3.92E+09, 3.17E+09, 1.30E+10,
                  2.93E+10, 2.21E+11, 1.24E+11, 1.51E+11, 8.46E+10, 2.93E+10,
                  1.58E+10, 1.22E+10, 9.14E+09, 2.06E+10, 1.05E+10, 9.92E+09,
                  9.57E+09, 8.90E+09, 7.89E+09, 7.24E+09, 8.07E+09, 7.62E+09,
                  8.76E+09, 9.78E+09, 1.01E+10, 2.11E+10],
                 43))

TAM_data.append(('A8639',
                 [31, 36, 39, 43, 46, 50, 53, 57, 60, 64],
                 [1.79E+08, 5.61E+08, 9.23E+08, 2.69E+10, 3.73E+10, 1.43E+10,
                  2.72E+10, 2.20E+10, 1.08E+11, 8.80E+11],
                 43))

TAM_data.append(('A8959',
                 [30, 33, 37, 40, 44, 47, 51, 54, 57, 62, 66],
                 [2.26E+09, 2.14E+09, 8.90E+09, 1.86E+10, 2.78E+10, 2.48E+10,
                  2.92E+10, 2.45E+10, 1.81E+10, 1.73E+10, 1.10E+10],
                 37))

TAM_data.append(('A8978',
                 [31, 35, 38, 42, 45, 49, 52, 56, 60, 64],
                 [6.33E+09, 2.96E+09, 9.76E+09, 2.77E+10, 1.96E+10, 1.62E+10,
                  1.15E+10, 5.24E+09, 7.41E+09, 6.75E+09],
                  31))

TAM_data.append(('A8981',
                 [31, 35, 38, 42, 45, 49, 52, 56, 60, 64, 68, 71, 74, 77, 80,
                  84, 87, 98, 101, 105, 108, 112, 119, 126, 129, 133],
                 [5.00E+08, 5.13E+08, 1.03E+09, 1.51E+09, 4.63E+09, 2.65E+09,
                  2.14E+09, 2.27E+09, 2.47E+09, 2.75E+09, 5.40E+09, 1.24E+10,
                  6.21E+09, 9.15E+09, 7.07E+09, 3.24E+09, 3.12E+09, 1.43E+09,
                  1.88E+09, 1.77E+09, 1.33E+09, 1.53E+09, 1.35E+09, 2.63E+09,
                  2.01E+09, 8.07E+09],
                  49))


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


def time_analysis(timevec, datavec, treatment, data=True):
    """ Given a time vector and a data vector for these times together with the
    treatment timepoint, finds the time point of highes overall BLI value.
    Returns this timepoint and its value."""
    def nearest_indx(array, value):
        return (np.abs(array-value)).argmin()
    def r3(stuff):
        return np.round(stuff, 3)
    # make input into lists
    timevec = list(r3(timevec))
    datavec = list(datavec)
    # find treatment timepoint' index
    tr_indx = timevec.index(treatment)
    # find highest BLI value AFTER TREATMENT and its index
    if data:  # proceed differently for data and simulated results
        # also: as one mouse paks twice, fudge this code into finding the first
        # peak by forcing it to return a result before day 70
        try:
            cut_indx = timevec.index(70)
        except ValueError:
            cut_indx = len(timevec)-1
        BLI_high = np.max(datavec[tr_indx:cut_indx])
        high_indx = datavec.index(BLI_high)
        t_high = timevec[high_indx]
    else:  # for simulated data use local max function

        high_indx = argrelextrema(np.array(datavec)[tr_indx:], np.greater)[0][0]
        BLI_high = datavec[high_indx+tr_indx]
        t_high = timevec[high_indx+tr_indx]

    return (t_high, BLI_high)


def TAM_plot():
    """ Plots raw data as well as interpolation and position of maximum
    reached BLI value as identified,

    1st version with big ugly spheres."""
    def r1(stuff):
        return np.round(stuff, 1)

    fig, axes = plt.subplots(3, 3, figsize=(9, 9), sharex=False)
    # collect peak times and relative increase
    t_max = []
    overshoot = []

    # plot raw data
    for mouse, ax in zip(TAM_data, [axes[0][0], axes[0][1], axes[0][2],
                                    axes[1][0], axes[1][1], axes[1][2],
                                    axes[2][0]]):
        ax.plot(mouse[1], mouse[2], '-o', label='data', color='firebrick')
        intertime, interdata = interpol(mouse[1], mouse[2])
        ax.plot(intertime, interdata, '--', color='black',
                label='interpolation')
        ax.set_yscale('log')

        if ax in [axes[2][0], axes[1][1], axes[1][2]]:
            ax.set_xlabel('time (days)')
        if ax in [axes[0][0], axes[1][0], axes[2][0]]:
            ax.set_ylabel('BLI (a.u.)')

        # plot special timepoints
        ax.plot(mouse[3], mouse[2][mouse[1].index(mouse[3])], 'o',
                color='slategrey', ms=20, label='TAM administration')

        high = time_analysis(intertime, interdata, mouse[3])
        os = high[1]/(mouse[2][mouse[1].index(mouse[3])])
        ax.plot(high[0], high[1], 'o', color='cornflowerblue', ms=20,
                label='maximum size after treatment')
        ax.set_title(r'$t_M$ = {}, F = {}'.format(
            r1(high[0]-mouse[3]), r1(os)))
        t_max.append(high[0]-mouse[3])
        overshoot.append(os)

    axes[2][1].axis('off')
    axes[2][2].axis('off')
    axes[0][0].legend(bbox_to_anchor=(2.8, -1.9))
    # fig.suptitle(r'Tlx-knockdown: $t_{{max}}$ = {} $\pm$ {} days, F = {} $\pm$ {}'.format(
    #     r1(np.mean(t_max)), r1(np.std(t_max)/np.sqrt(len(t_max))),
    #     r1(np.mean(overshoot)), r1(np.std(overshoot)/np.sqrt(len(overshoot)))),
    #     y=0.93)
    plt.tight_layout()
    pylab.savefig('figures/TAM_data.pdf', bbox_inches='tight')


def TAM_plot2():
    """ Plots raw data as well as interpolation and indication of when
    treatment was applied,

    2nd version without ugly spheres."""
    def r1(stuff):
        return np.round(stuff, 1)

    fig, axes = plt.subplots(3, 3, figsize=(9, 9), sharex=False)

    # plot raw data
    for mouse, ax in zip(TAM_data, [axes[0][0], axes[0][1], axes[0][2],
                                    axes[1][0], axes[1][1], axes[1][2],
                                    axes[2][0]]):
        ax.plot(mouse[1], mouse[2], '-o', label='data', color='firebrick')
        intertime, interdata = interpol(mouse[1], mouse[2])
        ax.plot(intertime, interdata, '--', color='black',
                label='interpolation')
        ax.set_yscale('log')

        if ax in [axes[2][0], axes[1][1], axes[1][2]]:
            ax.set_xlabel('time (days)')
        if ax in [axes[0][0], axes[1][0], axes[2][0]]:
            ax.set_ylabel('BLI (a.u.)')

        # treatment time window
        ax.axvline(mouse[3], ls='-', color='slategrey',
            label='daily TAM administrations', zorder=0)
        for i in range(1, 10):
            ax.axvline(mouse[3]+i, ls='-', color='slategrey', zorder=0)

    axes[2][1].axis('off')
    axes[2][2].axis('off')
    axes[0][0].legend(bbox_to_anchor=(2.8, -1.9))
    # fig.suptitle(r'Tlx-knockdown: $t_{{max}}$ = {} $\pm$ {} days, F = {} $\pm$ {}'.format(
    #     r1(np.mean(t_max)), r1(np.std(t_max)/np.sqrt(len(t_max))),
    #     r1(np.mean(overshoot)), r1(np.std(overshoot)/np.sqrt(len(overshoot)))),
    #     y=0.93)
    plt.tight_layout()
    pylab.savefig('figures/TAM_data_tw.pdf', bbox_inches='tight')
