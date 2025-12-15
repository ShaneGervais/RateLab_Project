import numpy as np

def reaclib_rate(T9, a):
    """
    Docstring for reactlib_rate

    REAClib 7-parameter rate  
    
    :param T9: Temperature in GK
    :param a: fit coefficients for Maxwellian-average reaction rate

    a_1 : overall normalization
    a_2 : Boltzmann suppression 
    a_3 : Colomn barrier
    a_4, 5, 6 : shape correction 
    a_7 : power-law prefactor

    returns reaction rate used by the libs entry (usually: cm^3 mol^-1 s^-1 for 2-body reaction)
    """

    T9 = np.asarray(T9, dtype=float)
    a1,a2,a3,a4,a5,a6,a7 = a

    return np.exp(a1 + a2*(T9**-1) + a3*(T9**(-1/3)) + a4*(T9**(1/3))
            + a5*(T9) + a6*(T9**(5/3)) + a7*np.log(T9))


def dlnrate_dlnT9(T9, a):
    """
    Docstring for dlnrate_dlnT9
    
    :param T9: Temperature in GK
    :param a: length 7 array for each

    returns temperature sensitivity
    """

    T9 = np.asarray(T9, dtype=float)
    a1,a2,a3,a4,a5,a6,a7 = a

    dE_dT9 = (-a2*(T9**-2)
        + (-1/3)*a3*(T9**-4/3)
        + ( 1/3)*a4*(T9**-2/3)
        + a5
        + (5/3)*a6*(T9**2/3)
        + a7*(1/T9))

    # d ln r / d ln T9 = T9 * dE/dT9
    return T9*dE_dT9


