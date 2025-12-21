import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from ratelab.trajectory import shock_trajectory
from ratelab.network import He4, O16, Ne20, Mg24, Si28, S32, AlphaChain

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

from ratelab.network import AlphaChain

net = AlphaChain(coef)

def run_onezone(T9_peak, tau, rho0, Y0):
    T9_traj  = lambda t: shock_trajectory(t, T9_peak=T9_peak, tau=tau)
    rho_traj = lambda t: rho0
    t_end = max(8.0*tau, 0.2)

    sol = solve_ivp(
        fun=lambda t,y: net.rhs(t, y, T9_traj=T9_traj, rho_traj=rho_traj),
        t_span=(0.0, t_end),
        y0=Y0,
        method="BDF",
        rtol=1e-7,
        atol=1e-12,
        t_eval=[t_end],
    )
    return sol.y[:, -1]


# To check phis of the reaction network, run this in main() instead of run_onezone
# phi = int (net_flux(t) dt) from 0 to t_end
# phi: total net amount of processing through that link over the burn
def run_onezone_final_and_phis(net, Y0, rho0, T9_peak, tau,
                               rtol=1e-7, atol=1e-12):
    """
    Fast diagnostics run: returns final X and integrated net phis
    without storing a time grid.
    """
    T9_traj  = lambda t: shock_trajectory(t, T9_peak=T9_peak, tau=tau)
    rho_traj = lambda t: rho0
    t_end = max(8.0*tau, 0.2)

    y0 = np.concatenate([Y0, np.zeros(4)])

    sol = solve_ivp(
        fun=lambda t,y: net.rhs_with_phis(t, y, T9_traj=T9_traj, rho_traj=rho_traj),
        t_span=(0.0, t_end),
        y0=y0,
        method="BDF",
        rtol=rtol,
        atol=atol,
        t_eval=[t_end],
    )

    Yf = sol.y[:6, -1]
    phis = sol.y[6:10, -1]  # (4,)

    # Convert to X (if you have A as global)
    Xf = Y_to_X(Yf)

    phi_dict = {
        "phi_O16_to_Ne20": float(phis[0]),
        "phi_Ne20_to_Mg24": float(phis[1]),
        "phi_Mg24_to_Si28": float(phis[2]),
        "phi_Si28_to_S32": float(phis[3]),
    }

    return Xf, phi_dict


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
                Yf = run_onezone(T9_peak, tau, rho0, Y0)   # (6,)
                Xf = Y_to_X(Yf)                            # (6,)
                drift = np.nan
                phi_O16 = phi_Ne20 = phi_Mg24 = phi_Si28 = np.nan

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