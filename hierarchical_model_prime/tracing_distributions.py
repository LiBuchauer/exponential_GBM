from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import pylab
from scipy.stats import skew, ks_2samp
from numpy.random import choice as randchoice
import seaborn
import pandas as pd
import pyximport
pyximport.install(
    setup_args={"include_dirs": np.get_include()},
    reload_support=True)
import SSA_prime_SPmigration
from importlib import reload
reload(SSA_prime_SPmigration)
from SSA_prime_SPmigration import family_of_clones_OUTSIDE as family_of_clones
seaborn.set_context('talk')
plt.rc('text', usetex=True)
pylab.ion()

""" Parameters of glioblastoma growth """
lambS = 0.22  # symmetric division rate of stem cells
lambA = 0.39  # asymmetric division rate of stem cells
lamb1 = 1.07  # division rate of progeny
d1 = 1.08  # differentiation rate of progeny to terminal state
mu1 = 0.11  # death rate of progeny
beta = 0.36  # migration rate out of clone of active stem cells


""" Experimental data """

d2016 = []
# day 5-1 (from "Graph with 4 animals")
d2016.append((5, np.concatenate((np.ones(157), 2*np.ones(31), 3*np.ones(8),
                                 4*np.ones(2), 5*np.ones(1)))))

# day 5-2 (from "10108-5 days graph")
d2016.append((5, np.concatenate((np.ones(57), 2*np.ones(14), 3*np.ones(1)))))

# day 10-1 (from "whole clones with percentage")
d2016.append((10, np.concatenate((np.ones(47), 2*np.ones(9), 3*np.ones(2)))))

# day 10-2 (from "whole clones with percentage")
d2016.append((10, np.concatenate((np.ones(86), 2*np.ones(12), 3*np.ones(1),
                                 6*np.ones(1)))))

# day 20-1 (from "Graph with 4 animals")
d2016.append((20, np.concatenate((np.ones(89), 2*np.ones(18), 3*np.ones(8),
                                 4*np.ones(7), 5*np.ones(8), 6*np.ones(2),
                                 7*np.ones(2), 8*np.ones(2), 11*np.ones(1),
                                 26*np.ones(1), 82*np.ones(1)))))

# day 20-2 (from "Graph with 4 animals")
d2016.append((20, np.concatenate((np.ones(75), 2*np.ones(29), 3*np.ones(14),
                                 4*np.ones(7), 5*np.ones(2), 6*np.ones(5),
                                 7*np.ones(1), 8*np.ones(2), 12*np.ones(1),
                                 17*np.ones(1), 21*np.ones(1), 44*np.ones(1),
                                 69*np.ones(1)))))

# day 26 (from "Graph with 4 animals")
d2016.append((26, np.concatenate((np.ones(82), 2*np.ones(33), 3*np.ones(14),
                                 4*np.ones(11), 5*np.ones(7), 6*np.ones(3),
                                 7*np.ones(3), 8*np.ones(1), 9*np.ones(1),
                                 18*np.ones(1), 21*np.ones(1), 23*np.ones(1),
                                 24*np.ones(2), 46*np.ones(1)))))

# day 26 (from "Graph with 4 animals")
d2016.append((37, np.concatenate((np.ones(77), 2*np.ones(26), 3*np.ones(15),
                                 4*np.ones(13), 5*np.ones(2), 6*np.ones(1),
                                 7*np.ones(1), 8*np.ones(3), 10*np.ones(1),
                                 15*np.ones(1), 29*np.ones(1),
                                 32*np.ones(1)))))

""" 2016 data with mice pooled by timepoint """
d2016merged = []

d2016merged.append((5, np.concatenate((np.ones(157), 2*np.ones(31), 3*np.ones(8),
                                 4*np.ones(2), 5*np.ones(1), np.ones(57),
                                 2*np.ones(14), 3*np.ones(1)))))

d2016merged.append((10, np.concatenate((np.ones(47), 2*np.ones(9), 3*np.ones(2),
                                  np.ones(86), 2*np.ones(12), 3*np.ones(1),
                                  6*np.ones(1)))))

d2016merged.append((20, np.concatenate((np.ones(89), 2*np.ones(18), 3*np.ones(8),
                                 4*np.ones(7), 5*np.ones(8), 6*np.ones(2),
                                 7*np.ones(2), 8*np.ones(2), 11*np.ones(1),
                                 26*np.ones(1), 82*np.ones(1), np.ones(75),
                                 2*np.ones(29), 3*np.ones(14),4*np.ones(7),
                                 5*np.ones(2), 6*np.ones(5), 7*np.ones(1),
                                 8*np.ones(2), 12*np.ones(1), 17*np.ones(1),
                                 21*np.ones(1), 44*np.ones(1),69*np.ones(1)))))

d2016merged.append((26, np.concatenate((np.ones(82), 2*np.ones(33), 3*np.ones(14),
                                 4*np.ones(11), 5*np.ones(7), 6*np.ones(3),
                                 7*np.ones(3), 8*np.ones(1), 9*np.ones(1),
                                 18*np.ones(1), 21*np.ones(1), 23*np.ones(1),
                                 24*np.ones(2), 46*np.ones(1)))))

d2016merged.append((37, np.concatenate((np.ones(77), 2*np.ones(26), 3*np.ones(15),
                                 4*np.ones(13), 5*np.ones(2), 6*np.ones(1),
                                 7*np.ones(1), 8*np.ones(3), 10*np.ones(1),
                                 15*np.ones(1), 29*np.ones(1),
                                 32*np.ones(1)))))


""" simulation calls and processing """
def experiment(
        endtime,
        famnum=100,
        timepoints=[5, 10, 20, 26, 37],
        plot=False,
        subsample=False,
        sss=100,
        lambS=lambS,
        lambA=lambA,
        lamb1=lamb1,
        d1=d1,
        mu1=mu1,
        beta=beta):
    """ Labels famnum stem cells initially. The  SCs are then passed to the
    family-evolver, finally results are merged and a histogram of clone size
    distributions is plotted. Mean, CV and skewness of the distribution are
    returned.
    If subsample==True, the clone distribution is subsampled to size sss and
    plots and statistic calculation is then performed on this."""
    # find the quiescent/SC ratio in steady distribution state for the given
    # parameters and uses this to determine the fraction of initially labeled
    # cells that will go on to do action
    collection = []
    # for each active clone, evolve its family tree and store the results
    for i in range(famnum):
        complete = family_of_clones(endtime, lambS, lambA, lamb1, d1, mu1, beta)
        collection += complete

    # transform data to information per timepoint rather than per clone
    time_coll = np.array(collection).T
    # for each day, keep just the non-zero things
    per_day = []
    for day in range(int(endtime)+1):
        per_day.append([cl for cl in time_coll[day] if cl != 0])

    # if subsampling is required, reduce the size of the lists at each tp

    if subsample:
        sub_per_day = []
        for day in range(int(endtime)+1):
            sub_per_day.append(randchoice(per_day[day], sss, replace=False))
        per_day = sub_per_day

    if plot:
        for tp in timepoints:
            if tp <= endtime:
                histdata = per_day[tp]
                fig, ax = plt.subplots(figsize=(4, 2.5))
                ax.hist(histdata, bins=np.arange(0, int(max(histdata)))+1.5,
                    color='slategrey')
                ax.set_yscale('log')
                ax.set_xlabel('clone size (cells)')
                ax.set_ylabel('appearances')
                ax.set_title('simulation day {}'.format(tp))
                pylab.savefig('figures/clone_histogram_SP_day{}.pdf'.format(tp),
                              bbox_inches='tight')

    # calculate statistics on this experiment and return
    means = [np.mean(dat) for dat in per_day]
    CVs = [np.std(dat)/np.mean(dat) for dat in per_day]
    skews = [skew(dat) for dat in per_day]

    sizecounts = [np.bincount(dat) for dat in per_day]
    lengths = [len(dat) for dat in per_day]

    frac1 = [sizecounts[i][1]/lengths[i] for i in range(int(endtime)+1)]
    frac2 = []
    for i in range(int(endtime)+1):
        try:
            frac2.append(sizecounts[i][2]/lengths[i])
        except IndexError:
            frac2.append(0)
    fracL = np.ones(int(endtime)+1)-np.array(frac1)-np.array(frac2)

    return (means, CVs, skews, frac1, frac2, fracL)


def noisy(
        endtime,
        famnum,
        repetitions,
        subsample=False,
        sss=100):
    """ Runs repetions repetitions of an experiment with famnum initially
    labeled stem cells until endtime and plots 6 summary statistics for each
    of them as a seperate line in combined plots. """
    results = []
    timevec = np.arange(endtime+1)
    for R in range(repetitions):
        thisresult = experiment(endtime=endtime, famnum=famnum, plot=False,
                                subsample=subsample, sss=sss)
        results.append(thisresult)

    fig, axes = plt.subplots(2, 3, sharex=True, figsize=(15, 10))
    for R in range(repetitions):

        axes[0][0].set_title('Mean clonesize')
        axes[0][0].plot(timevec, results[R][0], '-o')
        #axes[0][0].set_ylim([1, 3.5])

        axes[0][1].set_title('CV clonesize')
        axes[0][1].plot(timevec, results[R][1], '-o')
        #axes[0][1].set_ylim([0, 3])

        axes[0][2].set_title('Skew clonesize')
        axes[0][2].plot(timevec, results[R][2], '-o')

        axes[1][0].set_title('Fraction of size 1 clones')
        axes[1][0].plot(timevec, results[R][3], '-o')
        axes[1][0].set_ylim([-0.05, 1.05])
        axes[1][0].set_xlabel('time (days)')

        axes[1][1].set_title('Fraction of size 2 clones')
        axes[1][1].plot(timevec, results[R][4], '-o')
        axes[1][1].set_ylim([-0.05, 1.05])
        axes[1][1].set_xlabel('time (days)')

        axes[1][2].set_title('Fraction of larger clones')
        axes[1][2].plot(timevec, results[R][5], '-o')
        axes[1][2].set_ylim([-0.05, 1.05])
        axes[1][2].set_xlabel('time (days)')

        fig.suptitle("""Showing {} realisations with {} initially labeled SCs each
                       Parameters: lambS = {}, lambA = {}, lamb1 = {},
                       d1 = {}, mu1 = {}, beta = {}""".format(
                       lambS, lambA, lamb1, d1, mu1, beta))
        pylab.savefig('figures/sim_stats_{}fams.pdf'.format(famnum),
                      bbox_inches='tight')


def hist_figs():
    """ Plot experimentally observed clone size distributions as histograms. """
    i=0
    for mouse in d2016merged:
        i+=1
        # count appearances of clone sizes
        fig, ax = plt.subplots(figsize=(4, 2.5))
        num = ax.hist(mouse[1], bins=np.arange(0, int(max(mouse[1])))+1.5,
            color='firebrick')
        ax.set_xlabel('clone size (cells)')
        ax.set_ylabel('appearances')
        ax.set_yscale('log')
        ax.set_title('data day {}'.format(mouse[0]))
        pylab.savefig("figures/mouse_day{}_{}.pdf".format(mouse[0],i),bbox_inches='tight')
        plt.close()


"""" Supporting functions producing simple statistics from exprimental and
simulated data. """


def CV(data):
    return np.std(data)/np.mean(data)


def frac1(data):
    ones = Counter(data)[1]
    clones = len(data)
    return ones/clones


def frac2(data):
    twos = Counter(data)[2]
    clones = len(data)
    return twos/clones


def problem_fig(
        endtime,
        famnum,
        repetitions,
        subsample=False,
        sss=100,
        BTrep=100):
    """ Illustrates the apparent contradiction of exponetial population growth
    and saturating clonal growth. """
    # data part
    tps, (Lmean, Emean), (LCV, ECV), (Lskew, Eskew), (Lf1, Ef1), (Lf2, Ef2), (LfL, EfL) = \
        all_bootstrap(BTrep)

    fig, ax = plt.subplots(1, 1, figsize=(5, 3.3))

    ax.set_ylabel('mean clone size (cells)')
    ax.errorbar(tps, Lmean, yerr=Emean, fmt='o', color='firebrick',
        label='data with bootstrap SE estimate')
    ax.set_ylim([0.8, 100])
    ax.set_xlabel('time (days)')
    ax.set_xticks([0,10, 20, 30])
    ax.set_yscale('log')

    # bulk simulation part
    y0 = [0, 1, 0, 0]
    tvec1 = np.arange(0, endtime)
    soln1 = odeint(tumour_ODE, y0, tvec1, args=(lambS, lambA, lamb1, d1, mu1))
    solution1 = soln1.T
    ax.plot(tvec1, np.sum(solution1, axis=0), '-', color='black',
        label='standard SPD-model')

    # migration simulation part
    timevec = np.arange(endtime+1)
    simresult = experiment(endtime=endtime, famnum=famnum, plot=False,
                                subsample=subsample, sss=sss)

    ax.plot(timevec, simresult[0], '--', color='black',
        label='SPD-model with migration')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.45))


    fig.tight_layout()
    pylab.savefig('clone_vs_bulk.pdf', bbox_inches='tight')


def bounds_tracing(filepath='figures/chain_50_10000_sim.h5'):
    """ Given a filepath, imports the MCMC results from the .h5 file found
    there. Simulates the forward path of a subset of parameter sets with the
    required posterior probability and plots the resulting band together with
    the data."""
    # import dataframes
    df = pd.read_hdf(filepath, key='MCMC')
    dfMAP = pd.read_hdf(filepath, key='MLE')
    # best Log-likelihood
    best = dfMAP['lnPosterior'][0]
    # time until which simulation is run
    endtime = 37
    tps, (Lmean, Emean), (LCV, ECV), (Lskew, Eskew), (Lf1, Ef1), (Lf2, Ef2), \
        (LfL, EfL) = all_bootstrap(BTrep=10000)

    # number of simulations that should be considered (do not do all because
    # of high cost)
    simnum = 1000
    results = []
    simcount = 0
    for i in range(len(df)):
        if (df['lnPosterior'][i] >= best+np.log(0.05)) and (simcount <= simnum):
            # get parameters from dataframe
            lambS = df['lambS'][i]
            lambA = df['lambA'][i]
            lamb1 = df['lamb1'][i]
            d1 = df['d1'][i]
            mu1 = df['mu1'][i]
            beta = df['beta'][i]

            # run simulation with these paramters
            this_result = experiment(endtime, famnum=200, plot=False,
                                     lambS=lambS, lambA=lambA, lamb1=lamb1,
                                     d1=d1, mu1=mu1, beta=beta)
            # store the result into the prepared array
            results.append(this_result)
            simcount += 1
    # for all timepoints, find upper and lower bounds of 95 % confidence
    # regions
    L_5 = np.min(np.array(results), axis=0)
    U_5 = np.max(np.array(results), axis=0)
    timevec = np.arange(endtime+1)
    # plot
    fig, axes = plt.subplots(2, 3, sharex=True, figsize=(9, 6))

    # simulation
    axes[0][0].fill_between(timevec, L_5[0], U_5[0], facecolor='slategrey',
                            zorder=0, interpolate=True)
    axes[0][1].fill_between(timevec, L_5[1], U_5[1], facecolor='slategrey',
                            zorder=0, interpolate=True,
                            label=r'model, $\alpha=0.05$ bounds')
    axes[0][2].fill_between(timevec, L_5[2], U_5[2], facecolor='slategrey',
                            zorder=0, interpolate=True)
    axes[1][0].fill_between(timevec, L_5[3], U_5[3], facecolor='slategrey',
                            zorder=0, interpolate=True)
    axes[1][1].fill_between(timevec, L_5[4], U_5[4], facecolor='slategrey',
                            zorder=0, interpolate=True)
    axes[1][2].fill_between(timevec, L_5[5], U_5[5], facecolor='slategrey',
                            zorder=0, interpolate=True)
    # data
    axes[0][0].set_ylabel('mean clone size')
    axes[0][0].errorbar(tps, Lmean, yerr=Emean, fmt='o', color='firebrick')
    axes[0][0].set_ylim([1, 3.5])

    axes[0][1].set_ylabel('CV clone size')
    axes[0][1].errorbar(tps, LCV, yerr=ECV, fmt='o', color='firebrick',
                        label='data with bootstrap SE estimate')
    axes[0][1].set_ylim([0, 3])
    axes[0][1].legend(bbox_to_anchor=(2.5, 1.35))

    axes[0][2].set_ylabel('skew clone size')
    axes[0][2].errorbar(tps, Lskew, yerr=Eskew, fmt='o', color='firebrick')
    axes[0][2].set_ylim([0, 13])

    axes[1][0].set_ylabel('fraction of size 1-cell clones')
    axes[1][0].errorbar(tps, Lf1, yerr=Ef1, fmt='o', color='firebrick')
    axes[1][0].set_ylim([0, 1])

    axes[1][1].set_ylabel('fraction of 2-cell clones')
    axes[1][1].errorbar(tps, Lf2, yerr=Ef2, fmt='o', color='firebrick')
    axes[1][1].set_ylim([0, 1])
    axes[1][1].set_xlabel('time (days)')

    axes[1][2].set_ylabel(r'fraction of $>$2-cell clones')
    axes[1][2].errorbar(tps, LfL, yerr=EfL, fmt='o', color='firebrick')
    axes[1][2].set_ylim([0, 1])
    axes[1][2].set_xlim([-1, 38])
    axes[1][2].set_xticks([0, 10, 20, 30])

    fig.tight_layout()
    pylab.savefig('figures/tracing_bounds.pdf', bbox_inches='tight')


def bootstrap_clonesizes(data, BTrep=1000):
    """ Bootstrap standard errors of mean and CV of given clonesize dataset.

    The datavectors should be in such form that that every entry is the number
    of cells in a single clone. What is calculated here are the standard
    errors of mean clonesize and CV of the clonesize.

    For reference: The SE of any sample statistic is the standard deviation
    of the sampling distribution for that statistic.

    Args:
        data (list, ndarray, or list thereof) - The data to be resampled.
        BTrep (int) - Number of bootstrap repetitions to be performed.
                      Defaults to 1000.

    Returns:
        Standard error of mean, standard error of coefficient of variation
            (CV), SE of skewness, SE of fraction of size 1 clones, SE of
            fraction of size 2 clones

    """
    # check if input is list of lists, if so: flatten
    # turn in to numpy array to allow multiple indexing later
    if isinstance(data[0], (list, np.ndarray)):
        original = np.array([dp for sublist in data for dp in sublist])
    else:
        original = np.array(data)

    # get number of integer vectors for resampling
    size = len(original)  # size of original sample
    resample_inds = [np.random.randint(size, size=size) for i in range(BTrep)]
    # get resamples according to these indices
    resamples = [original[indx] for indx in resample_inds]

    # get mean, CV, fracs for each new sample
    means = np.mean(resamples, axis=1)
    stds = np.std(resamples, axis=1)
    CVs = stds/means
    skews = skew(resamples, axis=1)
    f1 = np.array([frac1(samp) for samp in resamples])
    f2 = np.array([frac2(samp) for samp in resamples])
    fL = np.ones(BTrep) - f1 - f2

    # calculate standard deviation of these values --> bootstrap se estimate
    se_mean = np.nanstd(means, ddof=1)
    se_CVs = np.nanstd(CVs, ddof=1)
    se_skew = np.nanstd(skews, ddof=1)
    se_f1 = np.nanstd(f1, ddof=1)
    se_f2 = np.nanstd(f2, ddof=1)
    se_fL = np.nanstd(fL, ddof=1)

    return se_mean, se_CVs, se_skew, se_f1, se_f2, se_fL


def all_bootstrap(BTrep=1000):
    tps = []
    Lmean = []
    LCV = []
    Lskew = []
    Lf1 = []
    Lf2 = []
    LfL = []
    Emean = []
    ECV = []
    Eskew = []
    Ef1 = []
    Ef2 = []
    EfL = []

    for mouse in d2016:
        print(mouse[0])
        tps.append(mouse[0])

        Lmean.append(np.nanmean(mouse[1]))
        LCV.append(CV(mouse[1]))
        Lskew.append(skew(mouse[1]))
        Lf1.append(frac1(mouse[1]))
        Lf2.append(frac2(mouse[1]))
        LfL.append(1-frac1(mouse[1])-frac2(mouse[1]))

        SEres = bootstrap_clonesizes(mouse[1], BTrep=BTrep)
        Emean.append(SEres[0])
        ECV.append(SEres[1])
        Eskew.append(SEres[2])
        Ef1.append(SEres[3])
        Ef2.append(SEres[4])
        EfL.append(SEres[5])

        print(np.nanmean(mouse[1]), CV(mouse[1]), frac1(mouse[1]),
              frac2(mouse[1]), 1-frac1(mouse[1])-frac2(mouse[1]))
        print(SEres)

    return tps, (Lmean, Emean), (LCV, ECV), (Lskew, Eskew), (Lf1, Ef1), (Lf2, Ef2), (LfL, EfL)


def KS_test(endtime=37, famnum=1000):
    """ Performs a Kolmogorov-Smirnov-test between the experimental clone
    size distribution and a simulated distribution starting from a given
    number of labelled families evolving it until endtime.
    Returns the test result for each timepoint for which experimental
    information is available."""
    # extract timepoints
    timepoints = [x[0] for x in d2016merged]

    # calculate and store simulation result for each of these timepoints
    collection = []
    # for each clone, evolve its family tree and store the results
    for i in range(famnum):
        complete = family_of_clones(endtime, lambS, lambA, lamb1, d1, mu1, beta)
        collection += complete

    # transform data to information per timepoint rather than per clone
    time_coll = np.array(collection).T
    # for each day, keep just the non-zero things
    per_day = []
    for day in range(int(endtime)+1):
        per_day.append([cl for cl in time_coll[day] if cl != 0])

    # for each timepoint, KS-test and print the result
    for i in range(len(timepoints)):
        DD, pp = ks_2samp(d2016merged[i][1], per_day[timepoints[i]])
        print(DD, pp)
