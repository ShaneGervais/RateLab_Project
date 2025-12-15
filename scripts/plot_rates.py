import numpy as np
import matplotlib.pyplot as plt
from ratelab.rates import reaclib_rate, dlnrate_dlnT9, dlnrate_dlnT9_fd


def main():

    # obtained from running python3 scripts/dump_reaclib_sets.py for reactions 
    coef_O16_ag_Ne20 = np.array([[[ 3.88571e+00, -1.03585e+01,  0.00000e+00,  0.00000e+00,  0.00000e+00,
    0.00000e+00, -1.50000e+00],
    [ 2.39030e+01,  0.00000e+00, -3.97262e+01, -2.10799e-01,  4.42879e-01,
    -7.97753e-02, -6.66667e-01],
    [ 9.50848e+00, -1.27643e+01,  0.00000e+00, -3.65925e+00,  7.14224e-01,
    -1.07508e-03, -1.50000e+00]]])

    coef_Ne20_ag_Mg24 = np.array([[ -8.79827,  -12.7809,     0.0,        16.9229,    -2.57325,    0.208997,
    -1.5     ],
    [  1.98307,   -9.22026,    0.0,         0.0,         0.0,         0.0,
    -1.5     ],
    [-38.7055,    -2.50605,    0.0,         0.0,         0.0,         0.0,
    -1.5     ],
    [ 24.5058,     0.0,       -46.2525,     5.58901,    7.61843,   -3.683,
    -0.666667]])
    
    coef_Mg24_ag_Si28 = np.array([[  8.03977,  -15.629,      0.0,         0.0,         0.0,         0.0,
    -1.5     ],
    [-50.5494,   -12.8332,    21.3721,    37.7649,    -4.10635,    0.249618,
    -1.5     ]])

    coef_Si28_ag_S32 = np.array([[ 47.9212,     0.0,       -59.4896,     4.47205,   -4.78989,    0.557201,
    -0.666667]])
    # for certain temperature range
    T9 = np.logspace(np.log10(0.1), np.log10(20.0), 5000)

    coef_O16_ag_Ne20 = np.squeeze(coef_O16_ag_Ne20)  # -> (3, 7)

    # Rates
    r_o16 = reaclib_rate(T9, coef_O16_ag_Ne20)
    r_ne20 = reaclib_rate(T9, coef_Ne20_ag_Mg24)
    r_mg24 = reaclib_rate(T9, coef_Mg24_ag_Si28)
    r_si  = reaclib_rate(T9, coef_Si28_ag_S32)

    # Sensitivities (analytic)
    s_o16 = dlnrate_dlnT9(T9, coef_O16_ag_Ne20)
    s_ne20 = dlnrate_dlnT9(T9, coef_Ne20_ag_Mg24)
    s_mg24 = dlnrate_dlnT9(T9, coef_Mg24_ag_Si28)
    s_si  = dlnrate_dlnT9(T9, coef_Si28_ag_S32)

    # Sensitivities (finite-difference check)
    s_o16_fd = dlnrate_dlnT9_fd(T9, coef_O16_ag_Ne20)
    s_ne20_fd = dlnrate_dlnT9_fd(T9, coef_O16_ag_Ne20)
    s_mg24_fd = dlnrate_dlnT9_fd(T9, coef_Mg24_ag_Si28)
    s_si_fd  = dlnrate_dlnT9_fd(T9, coef_Si28_ag_S32)


    plt.figure()
    plt.loglog(T9, r_o16, label=r"$^{16}\mathrm{O}(\alpha,\gamma)^{20}\mathrm{Ne}$")
    plt.loglog(T9, r_ne20, label=r"$^{20}\mathrm{Ne}(\alpha,\gamma)^{24}\mathrm{Mg}$")
    plt.loglog(T9, r_mg24,  label=r"$^{24}\mathrm{Mg}(\alpha,\gamma)^{32}\mathrm{Si}$")
    plt.loglog(T9, r_si,  label=r"$^{28}\mathrm{Si}(\alpha,\gamma)^{32}\mathrm{S}$")
    plt.xlabel(r"$T_9$")
    plt.ylabel(r"$N_A\langle\sigma v\rangle\ \mathrm{[cm^3\,mol^{-1}\,s^{-1}]}$")
    plt.legend()
    plt.tight_layout()
    plt.savefig("rate_curves.png", dpi=300)

    # ---- Plot 2: sensitivities ----
    plt.figure()
    plt.semilogx(T9, s_o16, label="O16(a,g)Ne20 analytic")
    plt.semilogx(T9, s_o16_fd, "--", label="O16(a,g)Ne20 FD")
    plt.semilogx(T9, s_ne20, label="Ne20(a,g)Mg24 analytic")
    plt.semilogx(T9, s_ne20_fd, "--", label="Ne20(a,g)Mg24 FD")
    plt.semilogx(T9, s_mg24, label="Mg24(a,g)Si28 analytic")
    plt.semilogx(T9, s_mg24_fd, "--", label="Mg24(a,g)Si28 FD")
    plt.semilogx(T9, s_si,  label="Si28(a,g)S32 analytic")
    plt.semilogx(T9, s_si_fd, "--", label="Si28(a,g)S32 FD")
    plt.xlabel(r"$T_9$")
    plt.ylabel(r"$d\ln r / d\ln T_9$")
    plt.legend()
    plt.tight_layout()
    plt.savefig("sensitivities.png", dpi=300)

if __name__ == "__main__":
    main()