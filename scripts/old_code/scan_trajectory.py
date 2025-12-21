import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from ratelab.trajectory import shock_trajectory
from ratelab.network import rhs, fluxes, He4, O16, Ne20, Mg24, Si28, S32

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

# Mass numbers for converting Y <-> X
A = np.array([4, 16, 20, 24, 28, 32], dtype=float)

def Y_to_X(Yvec):
    AY = A * Yvec
    return AY / np.sum(AY)

def baryon_invariant(Yvec):
    # Should stay constant if network bookkeeping is correct
    return float(np.sum(A * Yvec))

def run_onezone(T9_peak, tau, rho0, Y0, n_eval=800):
    """
    Runs a single one-zone burn and returns:
      X_final (6,), baryon_drift, and integrated net fluxes (optional)
    """
    T9_traj  = lambda t: shock_trajectory(t, T9_peak=T9_peak, tau=tau)
    rho_traj = lambda t: rho0

    # choose an end time that actually cools: a few tau
    t_end = max(8.0 * tau, 0.2)   # seconds
    t_span = (0.0, t_end)

    B0 = baryon_invariant(Y0)

    sol = solve_ivp(
        fun=lambda t, y: rhs(t, y, coef, T9_traj=T9_traj, rho_traj=rho_traj),
        t_span=t_span,
        y0=Y0,
        method="BDF",
        rtol=1e-8,
        atol=1e-14,
        dense_output=True,
    )

    t = np.linspace(t_span[0], t_span[1], n_eval)
    Yt = sol.sol(t)  # shape (6, n_eval)

    Yf = Yt[:, -1]
    Xf = Y_to_X(Yf)

    # baryon conservation check
    B = np.array([baryon_invariant(Yt[:, k]) for k in range(Yt.shape[1])])
    baryon_drift = float(np.max(np.abs(B - B0)) / max(abs(B0), 1e-300))

    # Integrated net fluxes (forward - reverse), robust to missing reverse keys and array-valued fluxes
    F_list = [fluxes(tk, Yt[:, k], coef, T9_traj=T9_traj, rho_traj=rho_traj) for k, tk in enumerate(t)]

    # Union of keys across all dictionaries (safe even if some keys are missing sometimes)
    all_names = sorted(set().union(*[Fi.keys() for Fi in F_list]))

    def scalarize(v):
        """Convert v (scalar/array) -> float. If array, sum all components."""
        a = np.asarray(v)
        if a.ndim == 0:
            return float(a)
        return float(np.sum(a))

    F_series = {
        name: np.array([scalarize(Fi.get(name, 0.0)) for Fi in F_list], dtype=float)
        for name in all_names
    }

    # Convenience getter (always 1D float array)
    def get(name):
        return F_series.get(name, np.zeros_like(t, dtype=float))

    net_O16  = get("O16_ag_Ne20")   - get("Ne20_ga_O16")
    net_Ne20 = get("Ne20_ag_Mg24")  - get("Mg24_ga_Ne20")
    net_Mg24 = get("Mg24_ag_Si28")  - get("Si28_ga_Mg24")
    net_Si28 = get("Si28_ag_S32")   - get("S32_ga_Si28")

    # These are now guaranteed 1D float arrays
    phi_O16  = float(np.trapezoid(net_O16,  t))
    phi_Ne20 = float(np.trapezoid(net_Ne20, t))
    phi_Mg24 = float(np.trapezoid(net_Mg24, t))
    phi_Si28 = float(np.trapezoid(net_Si28, t))


    return Xf, baryon_drift, (phi_O16, phi_Ne20, phi_Mg24, phi_Si28)

def main():
    # -----------------------------
    # 2) Initial composition (make sure sums to 1)
    # -----------------------------
    X0 = np.zeros(6)
    X0[He4] = 0.20
    X0[O16] = 0.80
    # others start at 0 (cleanest for scans)

    # convert X -> Y: Y = X/A
    Y0 = X0 / A

    # -----------------------------
    # 3) Scan grid
    # -----------------------------
    rho0 = 1e6  # g/cm^3 (hold fixed for this project)

    T9_vals  = np.linspace(1.0, 6.0, 25)          # peak temperature
    tau_vals = np.logspace(np.log10(0.02), np.log10(1.0), 25)  # cooling time in s

    # store heatmap for one interesting product (e.g., S32)
    X_S32_map = np.zeros((len(tau_vals), len(T9_vals)))

    out_csv = "scan_T9peak_tau.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "T9_peak", "tau_s", "rho0",
            "X_He4", "X_O16", "X_Ne20", "X_Mg24", "X_Si28", "X_S32",
            "baryon_drift",
            "phi_net_O16_to_Ne20",
            "phi_net_Ne20_to_Mg24",
            "phi_net_Mg24_to_Si28",
            "phi_net_Si28_to_S32",
        ])

        for itau, tau in enumerate(tau_vals):
            for iT, T9_peak in enumerate(T9_vals):
                Xf, drift, phis = run_onezone(T9_peak, tau, rho0, Y0)
                phi_O16, phi_Ne20, phi_Mg24, phi_Si28 = phis

                X_S32_map[itau, iT] = Xf[S32]

                w.writerow([
                    float(T9_peak), float(tau), float(rho0),
                    float(Xf[He4]), float(Xf[O16]), float(Xf[Ne20]),
                    float(Xf[Mg24]), float(Xf[Si28]), float(Xf[S32]),
                    float(drift),
                    phi_O16, phi_Ne20, phi_Mg24, phi_Si28
                ])

    print(f"Wrote {out_csv}")

    # -----------------------------
    # 4) Heatmap plot: final X_S32(T9_peak, tau)
    # -----------------------------
    plt.figure()
    im = plt.imshow(
        X_S32_map,
        origin="lower",
        aspect="auto",
        extent=[T9_vals[0], T9_vals[-1], np.log10(tau_vals[0]), np.log10(tau_vals[-1])]
    )
    plt.colorbar(im, label=r"final $X(^{32}\mathrm{S})$")
    plt.xlabel(r"$T_{9,\mathrm{peak}}$")
    plt.ylabel(r"$\log_{10}(\tau\ \mathrm{s})$")
    plt.tight_layout()
    plt.savefig("heatmap_X_S32.png", dpi=300)
    print("Saved heatmap_X_S32.png")

if __name__ == "__main__":
    main()





