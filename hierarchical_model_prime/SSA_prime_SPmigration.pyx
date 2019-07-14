# Help from http://hplgit.github.io/teamods/MC_cython/main_MC_cython.html
# introducing typed memoryviews in this version, see also
# https://jakevdp.github.io/blog/2012/08/08/memoryview-benchmarks/

cimport cython
from libc.stdlib cimport rand, RAND_MAX
from libc.math cimport log, ceil, floor, sqrt

from cython.view cimport array as cvarray


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)  # avoids division by 0 check
cdef core_clone(double birthtime, double endtime, double lambS, double lambA,
                double lamb1, double d1, double mu1, double beta,
                int stem0, int prog0):
    # first and last recording days
    cdef int firstday
    if birthtime == 0:
        firstday = 0
    else:
        firstday = <int>birthtime + 1
    cdef int lastday = <int>endtime
    cdef int arrsize = 100
    # array for storing offspring times, use fixed size, and index for this
    # needed for stem and progeny offsprings respectively
    # stem
    cdef double [::1] offspring_S = cvarray(shape=(arrsize,),
                                         itemsize=sizeof(double),
                                         format='d',
                                         mode='c')
    offspring_S[:] = 0
    cdef int off_S = 0
    # progeny
    cdef double [::1] offspring_P = cvarray(shape=(arrsize,),
                                         itemsize=sizeof(double),
                                         format='d',
                                         mode='c')
    offspring_P[:] = 0
    cdef int off_P = 0

    # initialise populations, starting with the given starting values of stem
    # and progeny cells and 0 exhausted cells
    cdef int SC = stem0  # stem cells
    cdef int Pr = prog0  # progeny
    cdef int Ex = 0  # exhausted cells

    # initiliase tnow as birthtime
    cdef double tnow = birthtime

    # for recording the size at each full day, use index between first and last
    cdef int day = firstday

    # array for recording all sizes between day 0 and endtime
    cdef int [::1] record = cvarray(shape=(lastday+1,),
                                       itemsize=sizeof(int),
                                       format='i',
                                       mode='c')
    record[:] = 0
    # initialise containers for calculation of time increment and propensities,
    # for cumulative propensity
    cdef double [::1] props = cvarray(shape=(7,),
                                      itemsize=sizeof(double),
                                      format='d',
                                      mode='c')
    cdef double cumprop = 0  # for iterating through cumulative propensities
    cdef double dt  # time increment
    cdef double RR  # random numbers from (0,1]
    cdef unsigned int e_type  # event index
    cdef double propsum  # propensity sum
    cdef int cellsum  # sum of all cells currently in clone

    # let's loop!
    while tnow < endtime:
        # calculate current propensities
        props[0] = lambS * SC
        props[1] = lambA * SC
        props[2] = lamb1 * Pr
        props[3] = d1 * Pr
        props[4] = mu1 * Ex
        props[5] = beta * SC
        props[6] = beta * Pr

        # if no acting cells remain, exit
        propsum = props[0]+props[1]+props[2]+props[3]+props[4]+props[5]+props[6]
        if propsum == 0:
            break

        # get increment to next event time and update current time
        RR = rand()/float(RAND_MAX)
        dt =  -log(RR)/propsum
        tnow += dt

        # if this timestep is the first on a new day, record the cellnumber.
        cellsum = SC + Pr + Ex
        while (tnow >= day) and (day <= lastday):
            record[day] = cellsum
            day += 1

        # identify the event having happened with a new random number
        RR = rand()/float(RAND_MAX)

        # calculate the vector of cumulative probabilities, as soon as a value
        # larger than the drawn value emerges, identify this event and break
        # this loop
        cumprop = 0
        e_type = 0
        while True:
            cumprop += props[e_type]/propsum
            if cumprop > RR:
                break
            else:
                e_type +=1

        # proceed with the chosen event
        if e_type == 0:  # symmetric division of stem cell
            SC += 1
        elif e_type == 1:  # asymmetric division of stem cell
            Pr += 1
        elif e_type == 2:  # symmetric division of progeny
            Pr += 1
        elif e_type == 3:  # differentiation of progeny
            Pr -= 1
            Ex += 1
        elif e_type == 4:  # death of exhausted cell
            Ex -= 1
        elif e_type == 5 and cellsum > 1:  # measurable SC migration event
            if off_S < arrsize:
                offspring_S[off_S] = tnow
                off_S += 1
            else:
                pass
            SC -= 1
        elif e_type == 6 and cellsum > 1:  # measurable Pr migration event
            if off_P < arrsize:
                offspring_P[off_P] = tnow
                off_P += 1
            else:
                pass
            Pr -= 1
        elif (e_type == 5 or e_type==6) and cellsum == 1:  # immeasurable migration event
            pass


    # after completion of the loop, potential last days without action need
    # to be logged as well
    cellsum = SC + Pr + Ex
    while (tnow >= day) and (day <= lastday):
        record[day] = cellsum
        day += 1
    return record, offspring_S, off_S, offspring_P, off_P


cdef family_of_clones(double endtime, double lambS, double lambA, double lamb1,
                      double d1, double mu1, double beta):
    """ Initially labels one cell and runs the core evolution for this, but
    then proceeds also evolving all the returned new active clones.
    Merges all clonal composition into an array at the end. """
    # list for collecting clonal timeseries that have been evolved until
    # endtime
    complete = []
    # store stem and progeny clones that have yet to be simulated after having
    # left their mother clone via migration
    stem_todo = []
    progeny_todo = []
    cdef int lastday = <int>endtime
    # return values of core_clone:
    # lists of times when S and P migrated respectively
    cdef double [::1] offspring_S = cvarray(shape=(100,),
                                          itemsize=sizeof(double),
                                          format='d',
                                          mode='c')
    cdef double [::1] offspring_P = cvarray(shape=(100,),
                                          itemsize=sizeof(double),
                                          format='d',
                                          mode='c')
    # array for recording all sizes of the clone between day 0 and endtime
    cdef int [::1] record = cvarray(shape=(lastday+1,),
                                    itemsize=sizeof(int),
                                    format='i',
                                    mode='c')
    cdef int off_S = 0
    cdef int off_P = 0
    cdef int i = 0
    # size one populations of SC and Pr
    cdef int one = 1
    cdef int zero = 0
    # run first clone from day 0 and store results
    record, offspring_S, off_S, offspring_P, off_P = \
      core_clone(0, endtime, lambS, lambA, lamb1, d1, mu1, beta, one, zero)
    # process results and store
    reclist = []
    # (this is necesseray because there may be times beyonf endtime in there
    # original record)
    for i in range(lastday+1):
        reclist.append(record[i])
    complete.append(reclist)
    for i in range(off_S):
        stem_todo.append(offspring_S[i])
    for i in range(off_P):
        progeny_todo.append(offspring_P[i])
    # while there are still unprocessed clones, process them likewise if their
    # birthtime is within the time window, proceed for stem clones first and
    # for progeny clones after

    # stem seeded clones
    cdef int counter = 0
    while 1:
        try:
            birthtime = stem_todo[counter]
            if birthtime < endtime:
                record, offspring_S, off_S, offspring_P, off_P = \
                    core_clone(birthtime, endtime, lambS, lambA, lamb1, d1,
                               mu1, beta, one, zero)
                # process results and store
                reclist = []
                for i in range(lastday+1):
                    reclist.append(record[i])
                complete.append(reclist)
                for i in range(off_S):
                    stem_todo.append(offspring_S[i])
                for i in range(off_P):
                    progeny_todo.append(offspring_P[i])
            counter += 1
        except IndexError:
            break

    # progeny seeded clones
    counter = 0
    while 1:
        try:
            birthtime = stem_todo[counter]
            if birthtime < endtime:
                record, offspring_S, off_S, offspring_P, off_P = \
                    core_clone(birthtime, endtime, lambS, lambA, lamb1, d1,
                               mu1, beta, zero, one)
                # process results and store
                reclist = []
                for i in range(lastday+1):
                    reclist.append(record[i])
                complete.append(reclist)
                for i in range(off_S):
                    stem_todo.append(offspring_S[i])
                for i in range(off_P):
                    progeny_todo.append(offspring_P[i])
            counter += 1
        except IndexError:
            break
    return complete

# @cython.boundscheck(False) # turn off bounds-checking for entire function
# @cython.wraparound(False)
# @cython.nonecheck(False)
# @cython.cdivision(True)  # avoids division by 0 check
def simstats(double endtime, int famnum, double lambS, double lambA,
             double lamb1, double d1, double mu1, double beta):
    """ Runs experiment from given number of starting families until
    endtime with given parameters and returns the stats mean, CV,
    fraction of size 1 and 2 clones for this last day."""
    # declarations
    cdef int i  # loop index
    cdef int clonenum  # number of clones
    cdef int thisclone  # for holding size of one clone
    cdef int S = 0  # for summing cellnumber
    cdef float V = 0  # for summing quadratic deviations
    cdef int C1 = 0  # count of size 1 clones
    cdef int C2 = 0  # count of size 2 clones
    cdef float mean  # mean cellnumber
    cdef float CV  # CV of cellnumber per clone
    cdef float f1  # fraction of size 1 clones
    cdef float f2  # fraction of size 2 clones
    cdef int lastday = <int>endtime
    # list for collecting the clones present on the last day
    lastdaylist = []
    # evolve each family tree until endtime
    for i in range(famnum):
        complete = family_of_clones(endtime, lambS, lambA, lamb1, d1, mu1, beta)
        lastdaylist += [cl[lastday] for cl in complete if cl[lastday] != 0]

    # calculate statistics on this experiment and return them
    # 1) mean, frac1 and frac2
    clonenum = len(lastdaylist)
    for i in range(clonenum):
        thisclone = lastdaylist[i]
        S += thisclone
        if thisclone == 1:
            C1 += 1
        elif thisclone == 2:
            C2 += 1
        else:
            pass
    mean = S/float(clonenum)
    f1 = C1/float(clonenum)
    f2 = C2/float(clonenum)
    # 2) std
    for i in range(clonenum):
        V += (lastdaylist[i] - mean)**2
    CV = sqrt(V/float(clonenum))/mean

    return mean, CV, f1, f2


def family_of_clones_OUTSIDE(double endtime, double lambS, double lambA, double lamb1,
                      double d1, double mu1, double beta):
    """ Initially labels one cell and runs the core evolution for this, but
    then proceeds also evolving all the returned new active clones.
    Merges all clonal composition into an array at the end. """
    # list for collecting clonal timeseries that have been evolved until
    # endtime
    complete = []
    # store stem and progeny clones that have yet to be simulated after having
    # left their mother clone via migration
    stem_todo = []
    progeny_todo = []
    cdef int lastday = <int>endtime
    # return values of core_clone:
    # lists of times when S and P migrated respectively
    cdef double [::1] offspring_S = cvarray(shape=(100,),
                                          itemsize=sizeof(double),
                                          format='d',
                                          mode='c')
    cdef double [::1] offspring_P = cvarray(shape=(100,),
                                          itemsize=sizeof(double),
                                          format='d',
                                          mode='c')
    # array for recording all sizes of the clone between day 0 and endtime
    cdef int [::1] record = cvarray(shape=(lastday+1,),
                                    itemsize=sizeof(int),
                                    format='i',
                                    mode='c')
    cdef int off_S = 0
    cdef int off_P = 0
    cdef int i = 0
    # size one populations of SC and Pr
    cdef int one = 1
    cdef int zero = 0
    # run first clone from day 0 and store results
    record, offspring_S, off_S, offspring_P, off_P = \
      core_clone(0, endtime, lambS, lambA, lamb1, d1,
                 mu1, beta, 1, 0)
    # process results and store
    reclist = []
    # (this is necesseray because there may be times beyonf endtime in there
    # original record)
    for i in range(lastday+1):
        reclist.append(record[i])
    complete.append(reclist)
    for i in range(off_S):
        stem_todo.append(offspring_S[i])
    for i in range(off_P):
        progeny_todo.append(offspring_P[i])
    # while there are still unprocessed clones, process them likewise if their
    # birthtime is within the time window, proceed for stem clones first and
    # for progeny clones after

    # stem seeded clones
    cdef int counter = 0
    while 1:
        try:
            birthtime = stem_todo[counter]
            if birthtime < endtime:
                record, offspring_S, off_S, offspring_P, off_P = \
                    core_clone(birthtime, endtime, lambS, lambA, lamb1, d1,
                               mu1, beta, one, zero)
                # process results and store
                reclist = []
                for i in range(lastday+1):
                    reclist.append(record[i])
                complete.append(reclist)
                for i in range(off_S):
                    stem_todo.append(offspring_S[i])
                for i in range(off_P):
                    progeny_todo.append(offspring_P[i])
            counter += 1
        except IndexError:
            break

    # progeny seeded clones
    counter = 0
    while 1:
        try:
            birthtime = stem_todo[counter]
            if birthtime < endtime:
                record, offspring_S, off_S, offspring_P, off_P = \
                    core_clone(birthtime, endtime, lambS, lambA, lamb1, d1,
                               mu1, beta, zero, one)
                # process results and store
                reclist = []
                for i in range(lastday+1):
                    reclist.append(record[i])
                complete.append(reclist)
                for i in range(off_S):
                    stem_todo.append(offspring_S[i])
                for i in range(off_P):
                    progeny_todo.append(offspring_P[i])
            counter += 1
        except IndexError:
            break
    return complete
