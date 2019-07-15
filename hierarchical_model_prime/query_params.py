# python 3.5
# utf-8
""" Functionality for analysing samples of the posterior, e.g. with respect
to the relationship between two parameters within paramter sets or
for analysing probability masses within certain bounds. """

import numpy as np
import matplotlib.pyplot as plt
import pylab
import seaborn
import pandas as pd

plt.rc('text', usetex=True)
seaborn.set_context('talk')
seaborn.set_style('ticks')
pylab.ion()

def parameter_relations(filepath):
    """ Given the path to a .h5 file storing posterior samples, imports these
    and analyses them according to a number of questions:
    - how likely is it that differentiated cells divide faster than stem cells?
    - (for posterior samples from unconstrained model runs:) how likely is it
        that differentiated cells divide slower than they move on into the
        terminal state?
    - how likely is it that stem cells divide asymetrically more often than
        they do symetrically?
    - how far apart are (d1-lamb1) and mu1 on average (is there are a
        correlation here?)

    Results are printed to screen and stored into a text file.
    """
    # import parameter file
    df = pd.read_hdf(filepath, key='MCMC')
    chainname = filepath.split('/')[1]

    # 1) how likely is it that differentiated cells divide faster than SCs?
    df['q1'] = df['lamb1'] - (df['lambA'] + df['lambS'])
    df['q1_sign'] = np.sign(df.q1)
    q1_pos = df['q1_sign'].value_counts()[1]
    q1_neg = df['q1_sign'].value_counts()[-1]
    q1_posfrac = q1_pos/(q1_pos+q1_neg)
    state1 = "Progeny divides faster than SCs: p = {}\n".format(q1_posfrac)

    # 2) how likely is it that differentiated cells divide slower than they
    # move on into the terminal state?
    df['q2'] = df['d1'] - df['lamb1']
    df['q2_sign'] = np.sign(df.q2)
    try:
        q2_pos = df['q2_sign'].value_counts()[1]
        q2_neg = df['q2_sign'].value_counts()[-1]
        q2_posfrac = q2_pos/(q2_pos+q2_neg)
        state2 = "Progeny differentiates faster than it divides: p = {}\n".format(
            q2_posfrac)
    except KeyError:
        state2 = "[Probably the constraint lamb1 < d1 was used in the model]\n"

    # 3) how likely is it that stem cells divide asymetrically more often than
    # they do symetrically?
    df['q3'] = df['lambA'] - df['lambS']
    df['q3_sign'] = np.sign(df.q3)
    q3_pos = df['q3_sign'].value_counts()[1]
    q3_neg = df['q3_sign'].value_counts()[-1]
    q3_posfrac = q3_pos/(q3_pos+q3_neg)
    state3 = "SCs divide asym. more often than sym.: p = {}\n".format(q3_posfrac)

    # 4) Is there a correlation between the death rate of terminal cells and
    # the effective rate of moving through the proliferating progeny
    # compartment?
    q4_corr = np.corrcoef(df['q2'], df['mu1'])[0][1]
    state4 = "Correlation between (d1-lamb1) and mu1: R = {}\n".format(q4_corr)
    plt.figure()
    plt.scatter(df['q2'], df['mu1'], s=10, alpha=0.1, color='firebrick')
    plt.xlabel(r'$d_1 - \lambda_1$')
    plt.ylabel(r'$\mu_1$')
    plt.title('Pearson\'s R = {}'.format(r2(q4_corr)))
    plt.savefig('figures/corr_d1_lamb1_mu1.pdf')

    # write to text file and print to screen
    print(state1 + state2 + state3 + state4)
    datafile = open('figures/parameter_relations_{}.txt'.format(chainname), 'w')
    datafile.write(state1 + state2 + state3 + state4)


def r2(num):
    return np.round(num, 2)


def probability_masses(filepath):
    """ Given the path to a .h5 file storing posterior samples, imports these
    and analyses what fraction of the probability mass is contained within
    the parameter bounds defined by the alpha=0.05 criterion (goal is to
    facilitate intuitive comparison with frequentist bounds and/or the
    gaussian distribution). """
    # import dataframes with parameters
    df = pd.read_hdf(filepath, key='MCMC')
    dfMAP = pd.read_hdf(filepath, key='MLE')
    dfUPPER = pd.read_hdf(filepath, key='UPPER')
    dfLOWER = pd.read_hdf(filepath, key='LOWER')
    # best Log-likelihood
    best = dfMAP['lnPosterior'][0]
    chainname = filepath.split('/')[1]
    # counters for all probability masses of intereset
    c_hyper = 0
    c_cuboid = 0
    c_lambS = 0
    c_lambA = 0
    c_lamb1 = 0
    c_d1 = 0
    c_mu1 = 0
    # conditions for being within the 0.05 bounds
    cond_lambS = ()
    for i in range(len(df)):
        # conditions for being within the 0.05 bounds
        cond_hyper = (df['lnPosterior'][i] >= best+np.log(0.05))
        cond_lambS = (dfLOWER['lambS'][0] <= df['lambS'][i] <= dfUPPER['lambS'][0])
        cond_lambA = (dfLOWER['lambA'][0] <= df['lambA'][i] <= dfUPPER['lambA'][0])
        cond_lamb1 = (dfLOWER['lamb1'][0] <= df['lamb1'][i] <= dfUPPER['lamb1'][0])
        cond_d1 = (dfLOWER['d1'][0] <= df['d1'][i] <= dfUPPER['d1'][0])
        cond_mu1 = (dfLOWER['mu1'][0] <= df['mu1'][i] <= dfUPPER['mu1'][0])
        cond_cuboid = (cond_lambS and cond_lambA and cond_lamb1 and cond_d1 and
                       cond_mu1)

        if cond_hyper:
            c_hyper += 1
        if cond_cuboid:
            c_cuboid += 1
        if cond_lambS:
            c_lambS += 1
        if cond_lambA:
            c_lambA += 1
        if cond_lamb1:
            c_lamb1 += 1
        if cond_d1:
            c_d1 += 1
        if cond_mu1:
            c_mu1 += 1

    # evaluate fractions within the bounds / regions, print and save
    total = len(df)
    state1 = "Probability mass in alpha=0.05-hypersurface = {}\n".format(c_hyper/total)
    state2 = "Probability mass in 5D hypercuboid = {}\n".format(c_cuboid/total)
    state3 = "Probability mass within alpha=0.05-lambS bounds = {}\n".format(c_lambS/total)
    state4 = "Probability mass within alpha=0.05-lambA bounds = {}\n".format(c_lambA/total)
    state5 = "Probability mass within alpha=0.05-lamb1 bounds = {}\n".format(c_lamb1/total)
    state6 = "Probability mass within alpha=0.05-d1 bounds = {}\n".format(c_d1/total)
    state7 = "Probability mass within alpha=0.05-mu1 bounds = {}\n".format(c_mu1/total)

    print(state1 + state2 + state3 + state4 + state5 + state6 + state7)
    datafile = open('figures/probability_masses_{}.txt'.format(chainname), 'w')
    datafile.write(state1 + state2 + state3 + state4 + state5 + state6 + state7)
