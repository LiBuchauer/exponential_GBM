# python 3.5
# utf-8

""" ODE system for generation free model with empty qSC compartment (leftover
for historical reasons)."""

import numpy as np

""" Generation-free version with stem cell compartment parametrised by p_Sc,
p_qu and lamb0. """


def tumour_ODE(y, t, lambS, lambA, lamb1, d1, mu1):
    """ Growth dynamics of a complete tumour.

    Population order: quiescent SCs, active SCs, progeny with remaining
    division potential 4 through 0.

    Args:
        y (array): Current values of each function.
        t (float): Current time point.
        [lamb0, p_SC, p_qu, mu0, lamb1, mu1] (floats): Parameter values for
            the system.

    Returns:
        The derivative(s) of y at t (array).
    """
    dy = np.zeros(4)
    dy[0] = 0  # qSC
    dy[1] = lambS*y[1]  # aSC
    dy[2] = lambA*y[1] + lamb1*y[2] - d1*y[2]  # progeny
    dy[3] = d1*y[2] - mu1*y[3]  # exhausted
    return dy
