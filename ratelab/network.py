import numpy as np
from .rates import reaclib_rate

# indices 
He4, O16, Ne20, Mg24, Si28, S32 = range(6)

# using molar abundances Y_i (mol/g) for two body reaction network i + j -> k

def rhs(t, Y, rho, coef, traj):
    """
    Docstring for rhs
    
    :param t: time
    :param Y: molar abundance vector (mol/g)
    :param rho: density (g/cm^3)
    :param coef: reaction fit coefficients -> a_sets
    :param traj: T9(t)
    """

    T9 = traj(t)

    #reaction rates
    r_o16 = reaclib_rate(T9, coef["o16_ag_ne20"])
    r_ne20 = reaclib_rate(T9, coef["ne20_ag_mg24"])
    r_mg24 = reaclib_rate(T9, coef["mg24_ag_si28"])
    r_si28 = reaclib_rate(T9, coef["si28_ag_s32"])

    dY = np.zeros_like(Y)

    # O16 + a -> Ne20
    f = rho * Y[He4] * Y[O16] * r_o16
    dY[He4] -= f; dY[O16] -= f; dY[Ne20] += f

    # Ne20 + a -> Mg24
    f = rho * Y[He4] * Y[Ne20] * r_ne20
    dY[He4] -= f; dY[Ne20] -= f; dY[Mg24] += f

    # Mg24 + a -> Si28
    f = rho * Y[He4] * Y[Mg24] * r_mg24
    dY[He4] -= f; dY[Mg24] -= f; dY[Si28] += f

    # Si28 + a -> S32
    f = rho * Y[He4] * Y[Si28] * r_si28
    dY[He4] -= f; dY[Si28] -= f; dY[S32] += f

    return dY