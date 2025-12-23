import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from ratelab.network import AlphaChain, He4, O16, S32, Si28, Mg24, Ne20
from ratelab.trajectory import shock_trajectory
from scan_trajectory import coef

# atomic masses
A = np.array([4, 16, 20, 24, 28, 32], dtype=float)

def Y_to_X(Y):
    AY = A * Y
    return AY/np.sum(AY)

def run_onezone_final(net, Y0, rho0, T9_peak, tau, rtol=1e-7, atol=1e-12):
    T9_traj  = lambda t: shock_trajectory(t, T9_peak=T9_peak, tau=tau)
    rho_traj = lambda t: rho0
    t_end = max(8.0*tau, 0.2)

    sol = solve_ivp(
        fun=lambda t, y: net.rhs(t, y, T9_traj=T9_traj, rho_traj=rho_traj),
        t_span=(0.0, t_end),
        y0=Y0,
        method="BDF",
        rtol=rtol,
        atol=atol,
        t_eval=[t_end],
    )
    return sol.y[:, -1]  # Y_final

def main():
    # Baseline conditions (tweak later)
    rho0 = 1e6
    T9_peak = 3.5
    tau = 0.2

    # Initial composition: 20% He4, 80% O16
    X0 = np.zeros(6)
    X0[He4] = 0.20
    X0[O16] = 0.80
    Y0 = X0 / A

    net = AlphaChain(coef)

    # Factors to try
    factors = [0.1, 1.0, 10.0]

    # Reactions to perturb (start with forward only; later include reverse too)
    reactions = [
        "o16_ag_ne20",
        "ne20_ag_mg24",
        "mg24_ag_si28",
        "si28_ag_s32",
        # Uncomment to explore reverse sensitivity too:
        "ne20_ga_o16",
        "mg24_ga_ne20",
        "si28_ga_mg24",
        "s32_ga_si28",
    ]

    # Baseline run
    for k in net.f:
        net.f[k] = 1.0
    Yb = run_onezone_final(net, Y0, rho0, T9_peak, tau)
    Xb = Y_to_X(Yb)

    print("Baseline final X:")
    print(f"  X(O16)={Xb[O16]:.3e}  X(Si28)={Xb[Si28]:.3e}  X(S32)={Xb[S32]:.3e}")

    results = []  # (reaction, factor, X_S32, X_Si28, X_O16)

    for rname in reactions:
        for f in factors:
            # reset all
            for k in net.f:
                net.f[k] = 1.0
            # set one
            net.f[rname] = f

            Yf = run_onezone_final(net, Y0, rho0, T9_peak, tau)
            Xf = Y_to_X(Yf)

            results.append((rname, f, Xf[S32], Xf[Si28], Xf[O16]))
            print(f"{rname:14s}  x{f:<4g}  X(S32)={Xf[S32]:.3e}  X(Si28)={Xf[Si28]:.3e}")

    # --- Plot: for each reaction, show X(S32) vs factor ---
    plt.figure()
    for rname in reactions:
        xs = [f for (rn, f, _, _, _) in results if rn == rname]
        ys = [x32 for (rn, f, x32, _, _) in results if rn == rname]
        plt.plot(xs, ys, marker="o", label=rname)

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("rate multiplier")
    plt.ylabel(r"final $X(^{32}\mathrm{S})$")
    plt.legend()
    plt.tight_layout()
    plt.savefig("sensitivity_X_S32.png", dpi=300)
    print("Saved sensitivity_X_S32.png")

if __name__ == "__main__":
    main()