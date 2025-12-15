import numpy as np

"""

    :param T9: Temperature in GK
    :param a: fit coefficients for Maxwellian-average reaction rate

    a_1 : overall normalization
    a_2 : Boltzmann suppression 
    a_3 : Colomn barrier
    a_4, 5, 6 : shape correction 
    a_7 : power-law prefactor   

"""

def _E(T9, a):
    """
    Docstring for reactlib_rate

    REAClib 7-parameter rate  

    returns energy
    """

    T9 = np.asarray(T9, dtype=float)
    a1,a2,a3,a4,a5,a6,a7 = a

    return (a1 + a2*(T9**-1) + a3*(T9**(-1/3)) + a4*(T9**(1/3))
            + a5*(T9) + a6*(T9**(5/3)) + a7*np.log(T9))

def _dE_dT9(T9, a):
    """
    Docstring for _dE_dT9

    returns rate of change of the energy with respect to T9
    """

    T9 = np.asarray(T9, dtype=float)
    a0,a1,a2,a3,a4,a5,a6 = a
    
    return (-a1*T9**(-2)
            + (-1/3)*a2*T9**(-4/3)
            + ( 1/3)*a3*T9**(-2/3)
            + a4
            + ( 5/3)*a5*T9**( 2/3)
            + a6*(1/T9))

def reaclib_rate(T9, a_sets):
    """
    Full REACLIB rate = sum_k exp(E_k(T9)).
    a_sets can be shape (7,) for one set or (nset, 7).

    returns Maxwellian-average of the reaction rate
    """

    T9 = np.asarray(T9, dtype=float)
    a_sets = np.asarray(a_sets, dtype=float)

    # dimensions check of coefficients
    if a_sets.ndim == 1:
        a_sets = a_sets[None, :]

    # shape: (nset, nT)
    E = np.vstack([_E(T9, a) for a in a_sets])

    # Maxwellian average 
    return np.sum(np.exp(E), axis=0)

def dlnrate_dlnT9(T9, a_sets):
    """
    d ln(rate) / d ln(T9) for a sum of sets:
      (T9/r) * sum_k r_k * dE_k/dT9
    """

    T9 = np.asarray(T9, dtype=float)

    a_sets = np.asarray(a_sets, dtype=float)
    if a_sets.ndim == 1:
        a_sets = a_sets[None, :]

    E = np.vstack([_E(T9, a) for a in a_sets])
    rk = np.exp(E)  # (nset, nT)
    r = np.sum(rk, axis=0)

    dE = np.vstack([_dE_dT9(T9, a) for a in a_sets])
    dr_dT9 = np.sum(rk * dE, axis=0)

    return T9 * dr_dT9 / r

def dlnrate_dlnT9_fd(T9, a_sets, eps=1e-6):
    """
    Finite-difference cross-check
    """
    T9 = np.asarray(T9, dtype=float)
    rp = reaclib_rate(T9*(1+eps), a_sets)
    rm = reaclib_rate(T9*(1-eps), a_sets)
    
    return (np.log(rp) - np.log(rm)) / (2*eps)