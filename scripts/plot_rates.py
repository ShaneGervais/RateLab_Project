import numpy as np
import matplotlib.pyplot as plt
from ratelab.rates import reaclib_rate, dlnrate_dlnT9, dlnrate_dlnT9_fd

# obtained from running python3 scripts/dump_reaclib_sets.py for reactions 
coef_O16_ag_Ne20 = np.array([[[ 3.88571e+00, -1.03585e+01,  0.00000e+00,  0.00000e+00,  0.00000e+00,
   0.00000e+00, -1.50000e+00],
 [ 2.39030e+01,  0.00000e+00, -3.97262e+01, -2.10799e-01,  4.42879e-01,
  -7.97753e-02, -6.66667e-01],
 [ 9.50848e+00, -1.27643e+01,  0.00000e+00, -3.65925e+00,  7.14224e-01,
  -1.07508e-03, -1.50000e+00]]])

coef_Si28_ag_S32 = np.array([[ 47.9212,     0.0,       -59.4896,     4.47205,   -4.78989,    0.557201,
   -0.666667]])

def main():

    # for certain temperature range
    T9 = np.logspace(np.log10(0.05), np.log10(10.0), 500)

    coef_O16_ag_Ne20 = np.squeeze(coef_O16_ag_Ne20)  # -> (3, 7)

    # Rates
    r_o16 = reaclib_rate(T9, coef_O16_ag_Ne20)
    r_si  = reaclib_rate(T9, coef_Si28_ag_S32)

    # Sensitivities (analytic)
    s_o16 = dlnrate_dlnT9(T9, coef_O16_ag_Ne20)
    s_si  = dlnrate_dlnT9(T9, coef_Si28_ag_S32)

    # Sensitivities (finite-difference check)
    s_o16_fd = dlnrate_dlnT9_fd(T9, coef_O16_ag_Ne20)
    s_si_fd  = dlnrate_dlnT9_fd(T9, coef_Si28_ag_S32)


    plt.figure()
    plt.loglog(T9, r_o16, label=r"$^{16}\mathrm{O}(\alpha,\gamma)^{20}\mathrm{Ne}$")
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
    plt.semilogx(T9, s_si,  label="Si28(a,g)S32 analytic")
    plt.semilogx(T9, s_si_fd, "--", label="Si28(a,g)S32 FD")
    plt.xlabel(r"$T_9$")
    plt.ylabel(r"$d\ln r / d\ln T_9$")
    plt.legend()
    plt.tight_layout()
    plt.savefig("sensitivities.png", dpi=300)

if __name__ == "__main__":
    main()