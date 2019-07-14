# python3
# utf-8

"""
vis3D: stochastic simulation of 3D tumour growth allowing to explore the
effects of migration and dispersal on the overall growth rate. Entails stem
cells and progeny cells. Stem cells divide symmetrically or asymetrically,
progeny can only die in this simplified version. Both cell types may migrate
according to specific rates. Migration may happen in a random direction or
away from the center of mass of the tumour according to the user settings.
"""

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import time
import sys


""" Parameters for agent-based simulations. """
N = 200  # grid size

lambS = 0.2  # rate of symmetric division of active stem cells
lambA = 0.4  # rate of asymmetric division of active stem cells
d1 = 0.0  # death rate of progeny cells
rhoS = 0.3  # migration rate of stem cells
rhoP = 0.0  # migration rate of progeny cells

dispersal = 3  # magnification factor of migration decisions, allows cells to
# stray further faster
noise = 2  # magnitude of noise on dispersal path

""" Stochastic simulation.
Description of event types and their order:
0) stem cells divide symmetrically with rate lambS
    - restrictions: can happen only if the chosen cell is alone at its
    current position (double occupation can occur due to migration)
    and there is an empty neighbour where the daughter cell can be put
1) stem cells divide asymmetrically with rate lambA
    - restrictions: can happen only if the chosen cell is alone at its
    current position (double occupation can occur due to migration)
    and there is an empty neighbour where the daughter cell can be put
2) stem cells migrate to another grid point with rate rhoS
    - can always happen, for crowded and uncrowded cells likewise
3) progeny cells die with rate d1
    - can always happen independently of crowding
4) progeny cells migrate to another grid point with rate rhoP
    - can always happen, for crowded and uncrowded cells likewise

Crowding of grid points comes about due to migration processes in this simple
scenario. When a migration decision is taken, there is no check whether the
target grid point is empty - migration is always allowed. However, cells on
grid points occupied by more than one cell are no longer allowed to divide
because of contact inhibition.

"""

""" Functions for generating randomness and other auxiliary functions"""

elements = np.array([  # 6 first order neighbours
                     (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0),
                     (0, 0, 1), (0, 0, -1),
                     # 12 second order neighbours
                     (1, 1, 0), (-1, 1, 0), (1, -1, 0), (-1, -1, 0),
                     (1, 0, 1), (-1, 0, 1), (1, 0, -1), (-1, 0, -1),
                     (0, 1, 1), (0, -1, 1), (0, 1, -1), (0, -1, -1),
                     # 8 third order neighbours
                     (1, 1, 1), (-1, 1, 1), (1, -1, 1), (1, 1, -1),
                     (-1, -1, 1), (-1, 1, -1), (1, -1, -1), (-1, -1, -1)])


def get_direction():
    """ Returns random direction to one of the 26 neighbours of a lot at
    random. """
    indx = np.random.choice(np.arange(26), 1)
    return indx


def get_next_event(propvec):
    """ Given a propensity vector, returns the waiting time until the next
    event and event type. The input vector must be an np array! """
    # get waiting time until next event from sum of propensities
    dt = -np.log(np.random.rand())/np.sum(propvec)

    # get index of event that will happen (~ eventtype)
    norm_cum = np.cumsum(propvec)/np.sum(propvec)
    e_type = np.argmax(norm_cum > np.random.rand())

    return dt, e_type


def get_cell_ID(cellnumber):
    """ Given a number of cells returns a random pick among them. """
    cell_ID = np.random.randint(cellnumber)
    return cell_ID


def empty_neighbour(cell_position, occupation, N):
    """ Given a gridpoint and a grid indicating current occupation with all
    cells combined finds all empty neighbours of the cell and returns one of
    them at random."""
    # first, check if all neighbours of the given cell are within bounds,
    # else print a warning
    if ((cell_position[0] > N-2) or (cell_position[0] < 1) or
            (cell_position[1] > N-2) or (cell_position[1] < 1) or
            (cell_position[2] > N-2) or (cell_position[2] < 1)):
        print('Warning: neighbour situation could not be assessed properly because of boundary issues. \nUse larger grid.')
        return None

    # get indices of neighbours of the cell in question
    neighbours = elements + cell_position
    # get the cellcount at each neighbour position
    neighbour_occupation = occupation[tuple(zip(*neighbours))]
    # find the empty ones
    empties = np.where(neighbour_occupation == 0)[0]
    # if there is no space, return None
    if len(empties) == 0:
        return None
    # else choose one of the empty ones at random
    else:
        indx = np.random.randint(len(empties))
        return neighbours[empties[indx]]


def random_move(dispersal, noise):
    """ Given a gridpoint and a grid incicating curent occupation with all
    cells combined returns a random migration vector. To accomodate dispersal,
    the resulting move is more than one step into the chosen direction,
    potentially altered by noise."""
    # get direction of random move
    move0 = elements[int(get_direction())]
    # calculate actual move according to required dispersal and noise
    move = dispersal*move0 + np.random.randint(-noise, noise+1, size=3)

    return move.astype(int)


def directed_move(cell_position, stem_list, progeny_list, dispersal, noise):
    """ Given a gridpoint and a grid incicating curent occupation with all
    cells combined finds the center of mass of the tumour and returns a
    migration vector which is inpired by the path of escape from the center
    of mass. To accomodate dispersal, the resulting move is more than one step
    into the chosen direction, potentially altered by noise."""
    # calculate center of mass
    center = com(stem_list, progeny_list)
    # vector from center to cell in question
    vector = cell_position - np.round(center)
    # if vector is all 0's, give random choice
    if np.sum(vector) == 0:
        move = elements[int(get_direction())]
        return move
    else:
        # normalise to the largest component and find best matching neighbour
        # via rounding of the result
        vecnorm = vector/np.max(np.abs(vector))
        move0 = np.round(vecnorm)
        # calculate actual move according to required dispersal and noise
        move = dispersal*move0 + np.random.randint(-noise, noise+1, size=3)

        return move.astype(int)


def com(stem_list, progeny_list):
    """ Returns the grid point closest to the center of mass of the tumor.
    Uses cell positions in stem_list and progeny_list to determine
    mean cell position which is equivalent to the center of mass assuming that
    all cells "weigh" the same.
    """
    # arrify
    stemarr = np.array(stem_list)
    progarr = np.array(progeny_list)
    # get means of each direction
    # it may happen that progeny list is empty
    if len(progarr) == 0:
        sum_x = np.sum(stemarr[:, 0])
        sum_y = np.sum(stemarr[:, 1])
        sum_z = np.sum(stemarr[:, 2])
    else:
        sum_x = np.sum(stemarr[:, 0])+np.sum(progarr[:, 0])
        sum_y = np.sum(stemarr[:, 1])+np.sum(progarr[:, 1])
        sum_z = np.sum(stemarr[:, 2])+np.sum(progarr[:, 2])
    lengths = len(stem_list) + len(progeny_list)
    return np.array([sum_x/lengths, sum_y/lengths, sum_z/lengths])


""" Simulation """


def run_sim_3D(
        maxtime,
        plot=True,
        migration='random',
        lambS=lambS,
        lambA=lambA,
        d1=d1,
        rhoS=rhoS,
        rhoP=rhoP,
        N=N,
        dispersal=dispersal,
        noise=noise):
    """ Runs stochastic simulation of 3D stem cell driven cancer growth.

    Input:
        maxtime (Float) - Maximum simulation time in simulated days.
        plot (bool) - whether a plot is generated or lists are returned
        migration (string) - if 'random', migrating cells choose a new position
            around their prior position at random, if 'directed', they move
            away from the center of mass of the tumour.
        Reaction rates for processes described above.
    """
    t1 = time.time()

    # prepare lists for holding positions of cells
    SCs = []
    progeny = []
    # initialize occupation grids (arrays) for stem cells and progeny
    # number signifies how many stem cells reside at this point
    stem_grid = np.zeros((N, N, N))
    progeny_grid = np.zeros((N, N, N))

    # initial condition: position one stem cell in the middle of the grid
    xstart = int(N/2)
    ystart = int(N/2)
    zstart = int(N/2)
    SCs.append(np.array([xstart, ystart, zstart]))
    stem_grid[xstart][ystart][zstart] = 1

    # book-keeping
    time_book = []  # store values of simulation time steps
    stem_book = []  # number of SCs present at each time step
    progeny_book = []  # number of progeny present at each time step
    # add starting point to books
    time_book.append(0)
    stem_book.append(len(SCs))
    progeny_book.append(len(progeny))
    # initialise time
    tnow = 0

    while tnow < maxtime:
        # calculate propensities for the five possible event types
        propvec = np.array([len(SCs)*lambS,
                            len(SCs)*lambA,
                            len(SCs)*rhoS,
                            len(progeny)*d1,
                            len(progeny)*rhoP])
        # find out which event happens next and when this is
        dt, e_type = get_next_event(propvec)
        # update time and proceed according to event type
        tnow += dt

        # if event is symmetric division of stem cell
        if e_type == 0:
            # choose a random active stem cell for division
            cell_ID = get_cell_ID(len(SCs))
            # get position of this cell
            xc, yc, zc = SCs[cell_ID]
            # check whether a division is possible, first condition is that
            # the cell is alone at its gridpoint
            if stem_grid[xc][yc][zc] == 1:
                # find out if there is an empty neighbour
                n_pos = empty_neighbour(np.array([xc, yc, zc]),
                                        stem_grid+progeny_grid, N)
                if n_pos is not None:
                    # make a new stem cell at this empty position
                    stem_grid[n_pos[0]][n_pos[1]][n_pos[2]] += 1
                    SCs.append(n_pos)
            # if one of the two conditions is not fulfilled, nothing happens

        # else if event is asymmetric division of stem cell
        elif e_type == 1:
            # choose a random active stem cell for division
            cell_ID = get_cell_ID(len(SCs))
            # get position of this cell
            xc, yc, zc = SCs[cell_ID]
            # check whether a division is possible, first condition is that
            # the cell is alone at its gridpoint
            if stem_grid[xc][yc][zc] == 1:
                # find out if there is an empty neighbour
                n_pos = empty_neighbour(np.array([xc, yc, zc]),
                                        stem_grid+progeny_grid, N)
                if n_pos is not None:
                    # make a new  progeny cell at this empty position
                    progeny_grid[n_pos[0]][n_pos[1]][n_pos[2]] += 1
                    progeny.append(n_pos)
            # if one of the two conditions is not fulfilled, nothing happens

        # else if event is migration of SC
        elif e_type == 2:
            # choose a random stem cell for movement
            cell_ID = get_cell_ID(len(SCs))
            old_pos = SCs[cell_ID]
            # get the direction of movement and calculate new position
            if migration == 'random':
                move = random_move(dispersal, noise)
            elif migration == 'directed':
                move = directed_move(old_pos, SCs, progeny, dispersal, noise)
            else:
                raise ValueError('You passed an invalid migration keyword, use "random" or "directed".')
            new_pos = old_pos + move
            # delete the old coordinates of the cell
            stem_grid[old_pos[0]][old_pos[1]][old_pos[2]] -= 1
            del SCs[cell_ID]
            # check if the new coordinates are outside the field of view
            if ((new_pos[0] > N-1) or (new_pos[0] < 0) or
                (new_pos[1] > N-1) or (new_pos[1] < 0) or
                (new_pos[2] > N-1) or (new_pos[2] < 0)):
                print('Warning: cells migrated out of the simulated area.')
                # these cells disappear from tracking and cannot reappear!
            # put stem cell to the new coordinates if they are okay
            else:
                stem_grid[new_pos[0]][new_pos[1]][new_pos[2]] += 1
                SCs.append(new_pos)

        # else if a progeny cell dies
        elif e_type == 3:
            # choose a random progeny cell for death
            cell_ID = get_cell_ID(len(progeny))
            xc, yc, zc = progeny[cell_ID]
            progeny_grid[xc][yc][zc] -= 1
            del progeny[cell_ID]

        # else if event is migration of progeny
        elif e_type == 4:
            # choose a random stem cell for movement
            cell_ID = get_cell_ID(len(progeny))
            old_pos = progeny[cell_ID]
            # get the direction of movement and calculate new position
            if migration == 'random':
                move = random_move(dispersal, noise)
            elif migration == 'directed':
                move = directed_move(old_pos, SCs, progeny, dispersal, noise)
            else:
                raise ValueError('You passed an invalid migration keyword, use "random" or "directed".')
            new_pos = old_pos + move
            # delete the old coordinates of the cell
            progeny_grid[old_pos[0]][old_pos[1]][old_pos[2]] -= 1
            del progeny[cell_ID]
            # check if the new coordinates are outside the field of view
            if ((new_pos[0] > N-1) or (new_pos[0] < 0) or
                (new_pos[1] > N-1) or (new_pos[1] < 0) or
                (new_pos[2] > N-1) or (new_pos[2] < 0)):
                print('Warning: cells migrated out of the simulated area.')
                # these cells disappear from tracking and cannot reappear!
            # put stem cell to the new coordinates if they are okay
            else:
                progeny_grid[new_pos[0]][new_pos[1]][new_pos[2]] += 1
                progeny.append(new_pos)

        # keep books
        if len(time_book) == 0 or tnow > time_book[-1] + 0.2:
            time_book.append(tnow)
            stem_book.append(len(SCs))
            progeny_book.append(len(progeny))

    t2 = time.time()
    print("Computation time = {} s".format(t2-t1))

    return (np.array(time_book), np.array(stem_book),
            np.array(progeny_book))


def timecourse_vis3D(maxtime, repeats=10):
    """ Runs stochastic simulation repeats times for each (hardcoded) parameter
    combination and saves individual simulation results to file for later
    import, interpolation and plot.
    """
    # open file stamped with systemtime if no other ID was provided in the call
    try:
        sys.argv[1]
    except IndexError:
        filepath = 'map_data/data{}.h5'.format(int(time.time()*100))
    else:
        filepath = 'map_data/data{}.h5'.format(sys.argv[1])
    print(filepath)
    datafile = pd.HDFStore(filepath)
    # dict for collecting result series
    seriesdict = {}

    # settings with which to run the simulation - each given setting will be
    # used 'repeats' times, settings will be co-written to file
    # (migration, lambS, lambA, d1, rhoS, rhoP, dispersal, noise)
    settings = [('random', 0.2, 0.4, 0.0, 0, 0, 50, 0, 0),
                ('random', 0.2, 0.4, 0.0, 0.3, 0, 200, 3, 2),
                ('directed', 0.2, 0.4, 0.0, 0.3, 0, 200, 3, 2),
                ('directed', 0.2, 0.4, 0.0, 0, 0.3, 50, 3, 2)]

    # save the result of each run as a separate series together with the
    # corresponding simulation parameters
    for i in range(len(settings)):
        print(settings[i])
        for j in range(repeats):
            time1 = time.time()
            time_book, stem_book, progeny_book = \
                run_sim_3D(maxtime, plot=False, migration=settings[i][0],
                lambS=settings[i][1], lambA=settings[i][2], d1=settings[i][3],
                rhoS=settings[i][4], rhoP=settings[i][5], N=settings[i][6],
                dispersal=settings[i][7], noise=settings[i][8])
            time2 = time.time()
            # interpolate
            t_interp = np.arange(0, maxtime, 0.1)
            # define interpolation functions
            s_interp = interp1d(time_book, stem_book, kind='slinear',
                                bounds_error=False)
            p_interp = interp1d(time_book, progeny_book, kind='slinear',
                                bounds_error=False)
            # get interpolated data
            stems = s_interp(t_interp)
            progs = p_interp(t_interp)
            sums = stems + progs
            # make settings string
            set_string = ''
            for kk in settings[i]:
                set_string += '_'
                set_string += str(kk)
            # store these results to dictionary
            series = pd.Series([time_book,
                                stem_book,
                                progeny_book,
                                maxtime,
                                t_interp,
                                stems,
                                progs,
                                sums,
                                set_string],
                                index=['time',
                                       'stem_number',
                                       'progeny_number',
                                       'maxtime',
                                       'time_interp',
                                       'stem_interp',
                                       'progeny_interp',
                                       'sum_interp',
                                       'settings'])
            ID = 'ID_{}'.format(time.time())
            seriesdict[ID] = series
            print('computation time = {}'.format(time2-time1))
    # make dataframe and store it
    df = pd.DataFrame(seriesdict)
    df = df.transpose()
    datafile['data'] = df
    # close datafile
    datafile.close()
    print('END')
