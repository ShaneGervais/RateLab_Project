import numpy as np
from .rates import reaclib_rate

# indices 
He4, O16, Ne20, Mg24, Si28, S32 = range(6)

# using molar abundances Y_i (mol/g) for two body reaction network i + j -> k

def rhs(t, Y, coef, T9_traj, rho_traj):
    """
    Docstring for rhs
    
    :param t: time
    :param Y: molar abundance vector (mol/g)
    :param rho: density (g/cm^3)
    :param coef: reaction fit coefficients -> a_sets
    :param traj: T9(t)
    """

    T9 = T9_traj(t)
    rho = rho_traj(t)

    #reaction rates
    r_o16 = reaclib_rate(T9, coef["o16_ag_ne20"])
    r_ne20 = reaclib_rate(T9, coef["ne20_ag_mg24"])
    r_mg24 = reaclib_rate(T9, coef["mg24_ag_si28"])
    r_si28 = reaclib_rate(T9, coef["si28_ag_s32"])

    dY = np.zeros_like(Y)

    # Forward O16 + a -> Ne20
    f = rho * Y[He4] * Y[O16] * r_o16
    dY[He4] -= f; dY[O16] -= f; dY[Ne20] += f

    # Reverse Ne20 -> O16 + a
    lam = reaclib_rate(T9, coef["ne20_ga_o16"])
    g = lam * Y[Ne20]
    dY[Ne20] -= g; dY[O16] += g; dY[He4] += g

    # Forward Ne20 + a -> Mg24
    f = rho * Y[He4] * Y[Ne20] * r_ne20
    dY[He4] -= f; dY[Ne20] -= f; dY[Mg24] += f

    # Reverse Mg24 -> Ne20 + a
    lam = reaclib_rate(T9, coef["mg24_ga_ne20"])
    g = lam * Y[Mg24]
    dY[Mg24] -= g; dY[Ne20] += g; dY[He4] += g

    # Forward Mg24 + a -> Si28
    f = rho * Y[He4] * Y[Mg24] * r_mg24
    dY[He4] -= f; dY[Mg24] -= f; dY[Si28] += f

    # Reverse Si28 -> Mg24 + a
    lam = reaclib_rate(T9, coef["si28_ga_mg24"])
    g = lam * Y[Si28]
    dY[Si28] -= g; dY[Mg24] += g; dY[He4] += g

    # Forward Si28 + a -> S32
    f = rho * Y[He4] * Y[Si28] * r_si28
    dY[He4] -= f; dY[Si28] -= f; dY[S32] += f

    # Reverse S32 -> Si28 + a
    lam = reaclib_rate(T9, coef["s32_ga_si28"])
    g = lam * Y[S32]
    dY[S32] -= g; dY[Si28] += g; dY[He4] += g

    return dY

def fluxes(t, Y, coef, T9_traj, rho_traj):
    T9  = T9_traj(t)
    rho = rho_traj(t)

    r_o16 = reaclib_rate(T9, coef["o16_ag_ne20"])
    r_ne20 = reaclib_rate(T9, coef["ne20_ag_mg24"])
    r_mg24 = reaclib_rate(T9, coef["mg24_ag_si28"])
    r_si28 = reaclib_rate(T9, coef["si28_ag_s32"])

    lam_ne20 = reaclib_rate(T9, coef["ne20_ga_o16"])
    lam_mg24 = reaclib_rate(T9, coef["mg24_ga_ne20"])
    lam_si28 = reaclib_rate(T9, coef["si28_ga_mg24"])
    lam_s32  = reaclib_rate(T9, coef["s32_ga_si28"])

    F = {}

    # forward (units ~ 1/s in Y-space because rho*Y*Y*rate)
    F["O16_ag_Ne20"]  = rho * Y[He4] * Y[O16]  * r_o16
    F["Ne20_ag_Mg24"] = rho * Y[He4] * Y[Ne20] * r_ne20
    F["Mg24_ag_Si28"] = rho * Y[He4] * Y[Mg24] * r_mg24
    F["Si28_ag_S32"]  = rho * Y[He4] * Y[Si28] * r_si28

    # reverse
    F["Ne20_ga_O16"]  = lam_ne20 * Y[Ne20]
    F["Mg24_ga_Ne20"] = lam_mg24 * Y[Mg24]
    F["Si28_ga_Mg24"] = lam_si28 * Y[Si28]
    F["S32_ga_Si28"]  = lam_s32  * Y[S32]

    # reverse (gamma,alpha)
    if "ne20_ga_o16" in coef:
        lam = reaclib_rate(T9, coef["ne20_ga_o16"])
        F["Ne20_ga_O16"] = lam * Y[Ne20]

    if "mg24_ga_ne20" in coef:
        am = reaclib_rate(T9, coef["mg24_ga_ne20"])
        F["Mg24_ga_Ne20"] = lam * Y[Mg24]

    if "si28_ga_mg24" in coef:
        lam = reaclib_rate(T9, coef["si28_ga_mg24"])
        F["Si28_ga_Mg24"] = lam * Y[Si28]

    if "s32_ga_si28" in coef:
        lam = reaclib_rate(T9, coef["s32_ga_si28"])
        F["S32_ga_Si28"] = lam * Y[S32]


    return F
