# python 3.5
# utf-8

""" Script for Bayesian parameter estimation with model prime, imports tumour
ODEs and residual evaluation functions for BLI growth law data, Ki67 index
data and miscellaneous "rest" data. This version incorporates the
single-cell tracing data and is extended by a stochastic simulation module in
order to allow comparison. """

import numpy as np
import matplotlib.pyplot as plt
import pylab
import seaborn
import emcee
import corner
import pandas as pd

import BLI_prime
import Ki67_prime
import rest_prime
import tracing_prime

try:
    reload
except NameError:
    from importlib import reload
reload(BLI_prime)
reload(Ki67_prime)
reload(rest_prime)
reload(tracing_prime)

from BLI_prime import BLI_residual, combined_BLI_plot
from Ki67_prime import Ki67_residual, combined_Ki67_plot
from rest_prime import rest_residual, combined_rest_plot
from tracing_prime import tracing_residual

pylab.ion()
seaborn.set_context('talk')
plt.rc('text', usetex=True)
cmap = ['lightgreen', 'forestgreen', 'tomato', 'firebrick', 'cornflowerblue',
        'darkblue']


""" Parameters of glioblastoma growth (rates in 1/day)"""

lambS = 0.21  # symmetric division rate of stem cells
lambA = 0.2  # asymmetric division rate of stem cells
lamb1 = 0.9  # division rate of progeny
d1 = 1.1  # probability that a progeny division leads to 2P (and not P+N)
mu1 = 0.1  # death rate of progeny
beta = 0.5  # migration rate out of clone of active stem cells

""" Log-likelihood functions of individual datasets, omitting constant arising
from prefactor 1/(sqrt(2*pi*std**2)), thus not allowing to fit parts of a
flexibel error model here. """


def lnlike_BLI(lambS=lambS):
    """ Returns log-likelihood of the average BLI growth rate assuming model
    omega (for death in all progeny compartments as well as only in the
    terminal one, as events in the progeny compartment have no effect on the
    long-term growth rate)."""
    model_rate = lambS
    data_rate = 0.213
    data_error = 0.071/np.sqrt(23)
    lnlike = (data_rate - model_rate)**2 / (2*data_error**2)
    return -lnlike


def lnlike_rest(lambS=lambS, lambA=lambA, lamb1=lamb1, d1=d1, mu1=mu1):
    """ Returns log-likelihood of the ratio data points assuming model omega
    and the curently active ODE system. """
    residuals = rest_residual([lambS, lambA, lamb1, d1, mu1])
    lnlike = residuals**2/2
    return -np.sum(lnlike)


def lnlike_Ki67(lambS=lambS, lambA=lambA, lamb1=lamb1, d1=d1, mu1=mu1):
    """ Returns log-likelihood of the ratio data points assuming model omega
    and the curently active ODE system. """
    residuals = Ki67_residual([lambS, lambA, lamb1, d1, mu1])
    lnlike = residuals**2/2
    return -np.sum(lnlike)


def lnlike_tracing(lambS=lambS, lambA=lambA, lamb1=lamb1, d1=d1, mu1=mu1,
                   beta=beta):
    """ Returns log-likelihood of the statistics derived from the tracing
    data. """
    residuals = tracing_residual([lambS, lambA, lamb1, d1, mu1, beta])
    lnlike = residuals**2/2
    return -np.sum(lnlike)


""" Flat prior function incorporating bounds on the parameters """


def lnprior(lambS=lambS, lambA=lambA, lamb1=lamb1, d1=d1, mu1=mu1, beta=beta):
    """ Returns 0 (\equiv finite prior, defined up to a constant) if all
    parameters are in the allowd range and -Infinity (\equiv prior of 0) if
    one or more of them are out of range """
    C1 = (0 <= lambS <= 4)  # cells do not divide more than 4 times a day
    pr1 = 1/4.

    C2 = (0 <= lambA <= 4)  # cells do not divide more than 4 times a day
    # pr2 = 1/4.*np.exp(lnlambA)
    pr2 = 1/4.

    C3 = (lamb1 <= d1)  # progeny compartment by itself unable to grow

    C4 = (0 <= lamb1 <= 4)  # cells do not divide more than 4 times a day
    pr4 = 1/4.

    C5 = (0 <= d1 <= 4)  # differentiation speed
    pr5 = 1/4.

    C6 = (0 <= mu1 <= 4)  # death rate of exhausted cells
    # pr6 = 1/4.*np.exp(lnmu1)
    pr6 = 1/4.

    C7 = (0 <= beta <= 4)  # migration speed
    pr7 = 1/4.

    # if one condition is not satisfied, this prior is invalid, else return
    # the log sum
    if (C1 and C2 and C3 and C4 and C5 and C6 and C7):
        return np.sum(np.log([pr1, pr2, pr4, pr5, pr6, pr7]))
    return -np.inf


""" Combine all the above into log-posterior from which to MCMC sample """


def lnpost(theta):
    """ Using the input parameters, gets prior and the likelihoods of all
    datasets and returns the probability of the input parameters using model
    omega."""
    # unpack theta into our parameters
    lambS, lambA, lamb1, d1, mu1, beta = theta
    prior = lnprior(lambS=lambS, lambA=lambA, lamb1=lamb1, d1=d1, mu1=mu1,
                    beta=beta)
    # check directly if queried parameters are out of bounds
    if not np.isfinite(prior):
        return -np.inf
    # else, calculate the log likelihood for each dataset
    like_BLI = lnlike_BLI(lambS=lambS)
    like_rest = lnlike_rest(lambS=lambS, lambA=lambA, lamb1=lamb1, d1=d1,
                            mu1=mu1)
    like_Ki67 = lnlike_Ki67(lambS=lambS, lambA=lambA, lamb1=lamb1, d1=d1,
                            mu1=mu1)
    like_tracing = lnlike_tracing(lambS=lambS, lambA=lambA, lamb1=lamb1, d1=d1,
                                  mu1=mu1, beta=beta)
    # add up logs of prior and likelihoods of datasets
    lnprob = prior + like_BLI + like_rest + like_Ki67 + like_tracing
    return lnprob


""" emcee ensemble sampler from small ball around initial parameter vals """


def MCMC_sample(walkers=100, steps=10000, lambS=lambS, lambA=lambA,
                lamb1=lamb1, d1=d1, mu1=mu1, beta=beta, diagnostics=False,
                storechain=True, burnin=2000):
    """ Initiates MCMC sampling from the posterior defined above using given
    number of walkers for given number of steps. Returns sampler object which
    carries the resulting parameters for all walkers at all steps. """
    # get starting positions for each walker
    theta = [lambS, lambA, lamb1, d1, mu1, beta]
    ndim = len(theta)
    startpos = [theta + 1e-1*np.random.rand(ndim) for i in range(walkers)]
    # set up sampler and run from the given starting positions
    sampler = emcee.EnsembleSampler(walkers, ndim, lnpost, threads=4)
    sampler.run_mcmc(startpos, steps)


    if diagnostics:
        try:
            autocorrest = np.round(sampler.acor, 2)
        except Exception:
            autocorrest = [np.nan, np.nan, np.nan, np.nan, np.nan]
        # plot simple diagnostics of a subset of walkers - time series and
        # autocorrelation for all parameters
        diagfig, axes = plt.subplots(len(theta), 2, figsize=(9, 18))
        titles = ['lambS', 'lambA', 'lamb1', 'd1', 'mu1', 'beta']
        # get a random subset of walkers for plotting, say 15
        plotinds = np.random.choice(range(walkers), size=15, replace=False)
        diagfig.suptitle("""walkers = {}, steps = {}, autocorrelation times = {},
                            mean acceptance fraction = {}""".format(walkers, steps,
                                autocorrest, np.mean(sampler.acceptance_fraction)))
        for f in range(len(theta)):
            for p in range(len(plotinds)):
                timeseries = sampler.chain[plotinds[p], :, f]
                axes[f][0].set_title('time series {}'.format(titles[f]))
                axes[f][0].plot(timeseries)
                axes[len(theta)-1][0].set_xlabel('index')

                axes[f][1].set_title('autocorrelation {}'.format(titles[f]))
                maxlag = min(steps, 200)
                axes[f][1].plot(range(maxlag), autocorr(timeseries, maxlag))
                axes[len(theta)-1][1].set_xlabel('lag')
        pylab.savefig('figures/diagnostics_{}_{}_C.pdf'.format(walkers,
                                                             (steps)))

    # find n-dim MAP and 1D credible regions at 5% level
    lambS_MAP, lambA_MAP, lamb1_MAP, d1_MAP, mu1_MAP, beta_MAP = \
        find_MAP(sampler.flatchain, sampler.flatlnprobability)
    lambS_5min, lambS_5max = find_CR(sampler.flatchain[:, 0],
                                     sampler.flatlnprobability)
    lambA_5min, lambA_5max = find_CR(sampler.flatchain[:, 1],
                                     sampler.flatlnprobability)
    lamb1_5min, lamb1_5max = find_CR(sampler.flatchain[:, 2],
                                     sampler.flatlnprobability)
    d1_5min, d1_5max = find_CR(sampler.flatchain[:, 3],
                               sampler.flatlnprobability)
    mu1_5min, mu1_5max = find_CR(sampler.flatchain[:, 4],
                                 sampler.flatlnprobability)
    beta_5min, beta_5max = find_CR(sampler.flatchain[:, 5],
                                   sampler.flatlnprobability)

    # plot
    samples = sampler.chain[:, burnin:, :].reshape((-1, ndim))
    fig = corner.corner(samples, labels=[r'$\lambda_S$ (1/day)',
                                         r'$\lambda_A$ (1/day)', r'$\lambda_1$ (1/day)',
                                         r'$d_1$ (1/day)',
                                         r'$\mu_1$ (1/day)',
                                         r'$\beta$ (1/day)'],
                        truths=[[lambS_MAP], [lambA_MAP], [lamb1_MAP], [d1_MAP], [mu1_MAP], [beta_MAP]],
                        truth_color=['firebrick'],
                        hist_kwargs={'color': 'darkgrey', 'histtype':'stepfilled'})
    fig.savefig("figures/MCMC_{}_{}_sim.pdf".format(walkers, steps))
    plt.close()

    # print stuff to file
    datafile = open('figures/results_{}_{}_sim.txt'.format(walkers, steps), 'w')

    datafile.write('1) n-dimensional MAP values \n \n')
    datafile.write('{} \n{} \n{} \n{} \n{} \n{} \n \n'.format(lambS_MAP,
                                                         lambA_MAP,
                                                         lamb1_MAP,
                                                         d1_MAP,
                                                         mu1_MAP,
                                                         beta_MAP))

    datafile.write('2) 0.05 credibility boundaries in 1-dim \n \n')
    datafile.write('{}, {} \n'.format(lambS_5min, lambS_5max))
    datafile.write('{}, {} \n'.format(lambA_5min, lambA_5max))
    datafile.write('{}, {} \n'.format(lamb1_5min, lamb1_5max))
    datafile.write('{}, {} \n'.format(d1_5min, d1_5max))
    datafile.write('{}, {} \n \n'.format(mu1_5min, mu1_5max))
    datafile.write('{}, {} \n \n'.format(beta_5min, beta_5max))

    datafile.write('3) quantiles: (2.5, 16, 50, 84, 97.5)\n')
    datafile.write('{}\n'.format(corner.quantile(samples[:, 0],
                                            [0.025, 0.16, 0.5, 0.84, 0.975])))
    datafile.write('{}\n'.format(corner.quantile(samples[:, 1],
                                            [0.025, 0.16, 0.5, 0.84, 0.975])))
    datafile.write('{}\n'.format(corner.quantile(samples[:, 2],
                                            [0.025, 0.16, 0.5, 0.84, 0.975])))
    datafile.write('{}\n'.format(corner.quantile(samples[:, 3],
                                            [0.025, 0.16, 0.5, 0.84, 0.975])))
    datafile.write('{}\n'.format(corner.quantile(samples[:, 4],
                                            [0.025, 0.16, 0.5, 0.84, 0.975])))
    datafile.write('{}\n'.format(corner.quantile(samples[:, 5],
                                            [0.025, 0.16, 0.5, 0.84, 0.975])))
    datafile.close()

    # plot profile posteriors
    lnprobs = sampler.flatlnprobability
    sampled_params = sampler.flatchain
    best_post = np.max(lnprobs)

    fig, axes = plt.subplots(2, 3, sharey=True, figsize=(12, 7))

    axes[0][0].plot(sampled_params[:, 0], -(lnprobs-best_post), 'o', color='darkgrey', alpha=0.1, rasterized=True)
    axes[0][0].set_title(r'$\lambda_S = {}_{{\ {}}}^{{\ {}}}$ 1/day'.format(r2(lambS_MAP),
        r2(lambS_5min), r2(lambS_5max)))
    axes[0][0].axhline(y=3, xmin=0., xmax=1, color='firebrick')
    axes[0][0].set_ylim([-0.2, 5])
    axes[0][0].set_ylabel(r'log $p(\theta | \theta_{{MAP}})$')

    axes[0][1].plot(sampled_params[:, 1], -(lnprobs-best_post), 'o', color='darkgrey', alpha=0.1, rasterized=True)
    axes[0][1].set_title(r'$\lambda_A = {}_{{\ {}}}^{{\ {}}}$ 1/day'.format(r2(lambA_MAP),
        r2(lambA_5min), r2(lambA_5max)))
    axes[0][1].set_xlim([0, 2])
    axes[0][1].axhline(y=3, xmin=0., xmax=1, color='firebrick')

    axes[0][2].plot(sampled_params[:, 2], -(lnprobs-best_post), 'o', color='darkgrey', alpha=0.1, rasterized=True)
    axes[0][2].set_title(r'$\lambda_1 = {}_{{\ {}}}^{{\ {}}}$ 1/day'.format(r2(lamb1_MAP),
        r2(lamb1_5min), r2(lamb1_5max)))
    axes[0][2].axhline(y=3, xmin=0., xmax=1, color='firebrick')

    axes[1][0].plot(sampled_params[:, 3], -(lnprobs-best_post), 'o', color='darkgrey', alpha=0.1, rasterized=True)
    axes[1][0].set_title(r'$d_1 = {}_{{\ {}}}^{{\ {}}}$ 1/day'.format(r2(d1_MAP),
        r2(d1_5min), r2(d1_5max)))
    axes[1][0].axhline(y=3, xmin=0., xmax=1, color='firebrick')
    axes[1][0].set_ylim([-0.2, 5])
    axes[1][0].set_ylabel(r'log $p(\theta | \theta_{{MAP}})$')

    axes[1][1].plot(sampled_params[:, 4], -(lnprobs-best_post), 'o', color='darkgrey', alpha=0.1, rasterized=True)
    axes[1][1].set_title(r'$\mu_1 = {}_{{\ {}}}^{{\ {}}}$ 1/day'.format(r2(mu1_MAP),
        r2(mu1_5min), r2(mu1_5max)))
    axes[1][1].set_xlim([-0.05, 0.7])
    axes[1][1].axhline(y=3, xmin=0., xmax=1, color='firebrick')
    axes[1][1].set_xlabel('rates (1/day)')

    axes[1][2].plot(sampled_params[:, 5], -(lnprobs-best_post), 'o', color='darkgrey', alpha=0.1, rasterized=True)
    axes[1][2].set_title(r'$\beta = {}_{{\ {}}}^{{\ {}}}$ 1/day'.format(r2(beta_MAP),
        r2(beta_5min), r2(beta_5max)))
    axes[1][2].axhline(y=3, xmin=0., xmax=1, color='firebrick')

    fig.subplots_adjust(hspace=0.35)

    pylab.savefig('figures/post_profiles_{}_{}_sim.pdf'.format(walkers, steps),
                  bbox_inches='tight')
    plt.close()

    if storechain:  # writes a the whole chain (if chain is short) or a
        # randomly subsampled set of size Nchain to a dataframe and then to a
        # .h5 file. Seperately saves the best found values (MLE).
        Nchain = 20000
        burned_lnP = sampler.flatlnprobability[burnin:]
        burned_params = sampler.flatchain[burnin:]
        chainlength = len(burned_lnP)
        if chainlength > Nchain:
            ch_indxs = np.random.choice(np.arange(chainlength), Nchain,
                                        replace=False)
            ch_probs = burned_lnP[ch_indxs]
            ch_params = burned_params[ch_indxs]
            print('long chain subsampled')
        else:
            ch_probs = burned_lnP
            ch_params = burned_params
            print('whole (short) chain stored')

        # create dataframe from this
        df = pd.DataFrame(
            {'lnPosterior': ch_probs,
             'lambS': [x[0] for x in ch_params],
             'lambA': [x[1] for x in ch_params],
             'lamb1': [x[2] for x in ch_params],
             'd1': [x[3] for x in ch_params],
             'mu1': [x[4] for x in ch_params],
             'beta': [x[5] for x in ch_params]})

        # mini dataframe for the MLE values
        dfMLE = pd.DataFrame(
            {'lnPosterior': [best_post],
             'lambS': [lambS_MAP],
             'lambA': [lambA_MAP],
             'lamb1': [lamb1_MAP],
             'd1': [d1_MAP],
             'mu1': [mu1_MAP],
             'beta': [beta_MAP]})

        df.to_hdf('figures/chain_{}_{}_sim.h5'.format(walkers, steps), key='MCMC',
                  mode='w')
        dfMLE.to_hdf('figures/chain_{}_{}_sim.h5'.format(walkers, steps), key='MLE',
                     mode='r+')
    return sampler


def HDI(posterior_samples, credible_mass):
    """ Computes highest density interval from a sample of representative
    values, estimated as the shortest credible interval Takes Arguments
    posterior_samples (samples from posterior) and credible mass."""
    sorted_points = sorted(posterior_samples)
    ciIdxInc = np.ceil(credible_mass * len(sorted_points)).astype('int')
    nCIs = len(sorted_points) - ciIdxInc
    ciWidth = [0]*nCIs
    for i in range(0, nCIs):
        ciWidth[i] = sorted_points[i + ciIdxInc] - sorted_points[i]
    HDImin = sorted_points[ciWidth.index(min(ciWidth))]
    HDImax = sorted_points[ciWidth.index(min(ciWidth))+ciIdxInc]
    return(HDImin, HDImax)


def autocorr(vec, maxlag):
    """ Given a vector vec, calculates the statistical autocorrelation for all
    lags between 0 and maxlag and returns them as array."""
    corrs = [np.corrcoef(np.array([vec[0:len(vec)-t], vec[t:len(vec)]]))[0, 1]
             for t in range(maxlag)]
    return np.array(corrs)


def find_MAP(samples, lnprobs):
    """ Given a list of parameter values and another corresponding list of
    their log posteriors finds the most likely set in the list and returns
    it."""
    best_index = np.argmax(lnprobs)
    best_params = samples[best_index]
    return best_params


def find_CR(samples1D, lnprobs, alpha=0.05):
    """ Given a list of single dimension parameter values and a corresponding
    list of log posteriors from the original n-dim parameter set, finds those
    log posteriors that are within alpha credibility and return the resulting
    boundaries on the parameter. """
    # get best available posterior probability
    best = np.max(lnprobs)
    # get allowed log differencefrom required credibility level
    logalpha = np.log(alpha)
    # find all params where the lnprob is within the allowed range
    good_params = [samples1D[i] for i in range(len(samples1D)) if lnprobs[i] >= best+logalpha]
    return np.min(good_params), np.max(good_params)


def r2(num):
    return np.round(num, 2)
