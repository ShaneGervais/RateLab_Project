import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from ratelab.trajectory import shock_trajectory
from ratelab.network import rhs, He4, O16, Ne20, Mg24, Si28, S32, fluxes

#from running dump_reaclib_sets.py dependence with pynucastro
coef = {
    "o16_ag_ne20": np.array([[ 3.88571e+00, -1.03585e+01,  0.00000e+00,  0.00000e+00,  0.00000e+00,
    0.00000e+00, -1.50000e+00],
    [ 2.39030e+01,  0.00000e+00, -3.97262e+01, -2.10799e-01,  4.42879e-01,
    -7.97753e-02, -6.66667e-01],
    [ 9.50848e+00, -1.27643e+01,  0.00000e+00, -3.65925e+00,  7.14224e-01,
    -1.07508e-03, -1.50000e+00]]),
    "ne20_ag_mg24": np.array([[ -8.79827,  -12.7809,     0.0,        16.9229,    -2.57325,    0.208997,
    -1.5     ],
    [  1.98307,   -9.22026,    0.0,         0.0,         0.0,         0.0,
    -1.5     ],
    [-38.7055,    -2.50605,    0.0,         0.0,         0.0,         0.0,
    -1.5     ],
    [ 24.5058,     0.0,       -46.2525,     5.58901,    7.61843,   -3.683,
    -0.666667]]),
    "mg24_ag_si28": np.array([[  8.03977,  -15.629,      0.0,         0.0,         0.0,         0.0,
    -1.5     ],
    [-50.5494,   -12.8332,    21.3721,    37.7649,    -4.10635,    0.249618,
    -1.5     ]]),
    "si28_ag_s32": np.array([[ 47.9212,     0.0,       -59.4896,     4.47205,   -4.78989,    0.557201,
    -0.666667]]),
    "ne20_ga_o16": np.array([[ 2.86431e+01, -6.52460e+01,  0.00000e+00,  0.00000e+00,  0.00000e+00,
    0.00000e+00,  0.00000e+00],
    [ 4.86604e+01, -5.48875e+01, -3.97262e+01, -2.10799e-01,  4.42879e-01,
    -7.97753e-02,  8.33333e-01],
    [ 3.42658e+01, -6.76518e+01,  0.00000e+00, -3.65925e+00,  7.14224e-01,
    -1.07508e-03,  0.00000e+00]]),
    "mg24_ga_ne20": np.array([[  16.0203,   -120.895,       0.0,         16.9229,     -2.57325,     0.208997,
    0.0      ],
    [  26.8017,   -117.334,       0.0,          0.0,          0.0,          0.0,
     0.      ],
    [ -13.8869,   -110.62,        0.0,          0.0,          0.0,          0.0,
     0.      ],
    [  49.3244,   -108.114,     -46.2525,      5.58901,     7.61843,    -3.683,
     0.833333]]),
    "si28_ga_mg24": np.array([[  32.9006,   -131.488,       0.0,          0.0,          0.0,          0.0,
     0.      ],
    [ -25.6886,   -128.693,      21.3721,     37.7649,     -4.10635,     0.249618,
     0.0      ]]),
    "s32_ga_si28": np.array([[ 72.813,    -80.626,    -59.4896,     4.47205,   -4.78989,    0.557201,
    0.833333]])
}

def main():

    # --- model parameters (adjust as needed) ---
    rho = 1e6           # g/cm^3 (toy value) (this can either fasten or slower the burning depending on the value)
    T9_peak = 1         # can change the threshold of the X value and need for cooling
    tau = 0.2          # s (fast cooling) time for the system to cool and reach a constant X
    t_end = 0.5         # s if this number is way bigger then cooling time tau, you'll see a freeze out effect happen more constantly

    traj = lambda t: shock_trajectory(t, T9_peak, tau) # check trajectory lib for default settings

    # solve molar abundance
    Y0 = np.zeros(6)
    # Convert mass fraction X to Y via Y = X/A (mol/g), where A≈mass number
    X_he, X_o16, X_ne20, X_mg24, X_si28, X_s32 = 0.2, 0.8, 0.1, 0.05, 0.001, 0.0
    Y0[He4] = X_he / 4.0
    Y0[O16] = X_o16 / 16.0
    Y0[Ne20] = X_ne20 / 20.0
    Y0[Mg24] = X_mg24 / 24.3
    Y0[Si28] = X_si28 / 28.0
    Y0[S32] = X_s32 / 32.0

    T9_traj  = lambda t: shock_trajectory(t, T9_peak=T9_peak, tau=tau)
    rho_traj = lambda t: rho 

    sol = solve_ivp(fun=lambda t, y: rhs(t, y, coef, T9_traj, rho_traj),
        t_span=(0.0, t_end), y0=Y0, method="BDF", rtol=1e-8,
        atol=1e-14, dense_output=True)
    
    t = np.linspace(0, t_end, 10000)
    Y = sol.sol(t)

    # Convert back to mass fractions X_i = A_i * Y_i / sum(A_j Y_j)
    A = np.array([4,16,20,24,28,32], dtype=float)
    AY = (A[:,None]*Y)
    X = AY / np.sum(AY, axis=0)


    plt.figure()
    for i, lab in [ (O16,"O16"), (Ne20,"Ne20"), (Mg24,"Mg24"), (Si28,"Si28"), (S32,"S32")]:
        plt.semilogy(t, X[i], label=lab)

    plt.xlabel("t [s]")
    plt.ylabel("Mass fraction X")
    plt.legend()
    plt.tight_layout()
    plt.savefig("onezone_massfractions2.png", dpi=300)
    print("Saved onezone_massfractions2.png")

    F_list = [fluxes(tk, Y[:, k], coef, T9_traj, rho_traj) for k, tk in enumerate(t)]

    names = sorted(F_list[0].keys())
    F_series = {name: np.array([Fi[name] for Fi in F_list]) for name in names}

    plt.figure()
    for name in names:
        plt.semilogy(t, F_series[name], label=name)
    plt.xlabel("t [s]")
    plt.ylabel("Flux (~1/s in Y-space)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("onezone_fluxes.png", dpi=300)
    print("Saved onezone_fluxes.png")



if __name__ == "__main__":
    main()