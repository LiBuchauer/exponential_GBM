# python3
# utf-8

""" Functionality for importing vis3D simulation results from several
independent runs, calculating statistics and plotting them. """

from os import listdir
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn
import itertools
import pylab

""" Plotting preliminaries """
pylab.ion()
plt.rc('text', usetex=True)
seaborn.set_context('talk')
seaborn.set_style('ticks')


def data_import(folder='map_data'):
    """ Imports all simulation results found in the given folder.
    Prints all used parameter settings and the number of simulation results
    available for each of them.
    Args:
        folder (string): Name of the folder to parse for data
    Returns:
        pandas datafile with the concatenated data.
    """

    # get all filenames in the data folder
    filenames = listdir(folder)
    # import and merge into one dataframe
    frames = []
    for name in filenames:
        if name.startswith('data'):
            frames.append(pd.read_hdf(folder+'/'+name, key='data'))
    df = pd.concat(frames)
    # print number of available simulations per setting combination
    gf = df.groupby(['settings', 'maxtime']).count()
    print(gf)

    return df


def plot_timecourse(folder='map_data'):
    """ Using all data available in the given folder, extracts the time courses
    per parameter setting in there and plots means and standard deviations of
    them as a function of time. """
    # import data and group it
    df = data_import(folder)
    grouped_df = df.groupby(['settings', 'maxtime'])

    # prepare to extract all (interpolated) timecourses from df
    time_lists = []
    stem_lists = []
    progeny_lists = []
    sum_lists = []
    labels = []
    i = 0
    for key, group in grouped_df:
        labels.append(key)
        time_lists.append([])
        stem_lists.append([])
        progeny_lists.append([])
        sum_lists.append([])
        for index, row in group.iterrows():
            time_lists[i].append(row['time_interp'])
            stem_lists[i].append(row['stem_interp'])
            progeny_lists[i].append(row['progeny_interp'])
            sum_lists[i].append(row['sum_interp'])
        i += 1

    # for each of these rows, get the mean and std
    time = [np.nanmean(np.array(setX), axis=0) for setX in time_lists]
    stem_means = [np.nanmean(np.array(setX), axis=0) for setX in stem_lists]
    stem_stds = [np.nanstd(np.array(setX), axis=0) for setX in stem_lists]
    prog_means = [np.nanmean(np.array(setX), axis=0) for setX in progeny_lists]
    prog_stds = [np.nanstd(np.array(setX), axis=0) for setX in progeny_lists]
    sum_means = [np.nanmean(np.array(setX), axis=0) for setX in sum_lists]
    sum_stds = [np.nanstd(np.array(setX), axis=0) for setX in sum_lists]

    # generate plot
    fig, axes = plt.subplots(1, 3, sharex=True, figsize=(14, 5))
    axes[0].set_title('Stem Cells')
    axes[1].set_title('Progeny')
    axes[2].set_title('All Cells')
    axes[0].set_yscale('log')
    axes[1].set_yscale('log')
    axes[2].set_yscale('log')

    # make labels readable
    new_labels = []
    for lab in labels:
        new_labels.append(lab[0].replace('_', ' ') + ' ' + str(lab[1]))

    print(new_labels)

    palette = itertools.cycle(seaborn.color_palette())
    for i in range(len(grouped_df)):
        col = next(palette)
        axes[0].fill_between(time[i], stem_means[i]-stem_stds[i],
            stem_means[i]+stem_stds[i], color=col, alpha=0.5)
        axes[0].plot(time[i], stem_means[i], label=new_labels[i])

        axes[1].fill_between(time[i], prog_means[i]-prog_stds[i],
            prog_means[i]+prog_stds[i], color=col, alpha=0.5)
        axes[1].plot(time[i], prog_means[i], label=new_labels[i])
        axes[1].set_ylim(ymin=1)

        axes[2].fill_between(time[i], sum_means[i]-sum_stds[i],
            sum_means[i]+sum_stds[i], color=col, alpha=0.5)
        axes[2].plot(time[i], sum_means[i], label=new_labels[i])
    axes[0].legend(loc=2)
    axes[0].set_ylabel('cell number')
    axes[1].set_xlabel('time (days)')
    seaborn.despine()
    fig.tight_layout()
    tag = folder.split('/')[-1]
    pylab.savefig('figures/timecourse_{}.pdf'.format(tag),
                  bbox_inches='tight')

    # generate plot
    fig, axes = plt.subplots(1, 1, sharex=True, figsize=(5, 5))
    axes.set_title('Tumour Growth')
    axes.set_yscale('log')

    palette = itertools.cycle(seaborn.color_palette())
    for i in range(len(grouped_df)):
        col = next(palette)
        axes.fill_between(time[i], sum_means[i]-sum_stds[i],
            sum_means[i]+sum_stds[i], color=col, alpha=0.5)
        axes.plot(time[i], sum_means[i], label=new_labels[i], color=col)
    axes.legend(loc=2)
    axes.set_ylabel('cell number')
    axes.set_xlabel('time (days)')
    seaborn.despine()
    fig.tight_layout()
    pylab.savefig('figures/timecourse_sum_{}.pdf'.format(tag),
                  bbox_inches='tight')
