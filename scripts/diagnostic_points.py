import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from ratelab.trajectory import shock_trajectory
from ratelab.network import AlphaChain, He4, O16, Ne20, Mg24, Si28, S32


from scan_trajectory import coef 

A = np.array([4, 16, 20, 24, 28, 32], dtype=float)

def Y_to_X(Y):
    AY = A * Y
    return AY / np.sum(AY)

def run_onezone_diagnostic(T9_peak, tau, rho0, Y0, n_eval=800):
    net = AlphaChain(coef)
    T9_traj  = lambda t: shock_trajectory(t, T9_peak=T9_peak, tau=tau)
    rho_traj = lambda t: rho0

    t_end = max(8.0 * tau, 0.2)

    sol = solve_ivp(
        fun=lambda t, y: net.rhs(t, y, T9_traj=T9_traj, rho_traj=rho_traj),
        t_span=(0.0, t_end),
        y0=Y0,
        method="BDF",
        rtol=1e-7,
        atol=1e-12,
        dense_output=True,   # IMPORTANT for sol.sol(t)
    )

    t = np.linspace(0.0, t_end, n_eval)
    Yt = sol.sol(t)  # shape (6, n_eval)
    Xt = np.array([Y_to_X(Yt[:, k]) for k in range(n_eval)])  # (n_eval, 6)

    # flux time series
    F_list = [net.fluxes(tk, Yt[:, k], T9_traj=T9_traj, rho_traj=rho_traj) for k, tk in enumerate(t)]
    def arr(name):
        return np.array([Fi.get(name, 0.0) for Fi in F_list], dtype=float)

    net_O16  = arr("O16_ag_Ne20")  - arr("Ne20_ga_O16")
    net_Ne20 = arr("Ne20_ag_Mg24") - arr("Mg24_ga_Ne20")
    net_Mg24 = arr("Mg24_ag_Si28") - arr("Si28_ga_Mg24")
    net_Si28 = arr("Si28_ag_S32")  - arr("S32_ga_Si28")

    # integrated net flows (phis)
    phi_O16  = float(np.trapezoid(net_O16,  t))
    phi_Ne20 = float(np.trapezoid(net_Ne20, t))
    phi_Mg24 = float(np.trapezoid(net_Mg24, t))
    phi_Si28 = float(np.trapezoid(net_Si28, t))

    out = {
        "t": t,
        "Yt": Yt,
        "Xt": Xt,
        "phis": (phi_O16, phi_Ne20, phi_Mg24, phi_Si28),
        "net_fluxes": (net_O16, net_Ne20, net_Mg24, net_Si28),
    }
    return out

def plot_diagnostic(out, tag):
    t = out["t"]
    Xt = out["Xt"]
    net_O16, net_Ne20, net_Mg24, net_Si28 = out["net_fluxes"]
    phi_O16, phi_Ne20, phi_Mg24, phi_Si28 = out["phis"]

    # mass fractions vs time
    plt.figure()
    plt.plot(t, Xt[:, O16], label="O16")
    plt.plot(t, Xt[:, Ne20], label="Ne20")
    plt.plot(t, Xt[:, Mg24], label="Mg24")
    plt.plot(t, Xt[:, Si28], label="Si28")
    plt.plot(t, Xt[:, S32], label="S32")
    plt.yscale("log")
    plt.xlabel("t [s]")
    plt.ylabel("mass fraction X")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"diag_X_{tag}.png", dpi=200)

    # net fluxes vs time
    plt.figure()
    plt.plot(t, np.abs(net_O16),  label=f"net O16→Ne20  (phi={phi_O16:.2e})")
    plt.plot(t, np.abs(net_Ne20), label=f"net Ne20→Mg24 (phi={phi_Ne20:.2e})")
    plt.plot(t, np.abs(net_Mg24), label=f"net Mg24→Si28 (phi={phi_Mg24:.2e})")
    plt.plot(t, np.abs(net_Si28), label=f"net Si28→S32  (phi={phi_Si28:.2e})")
    plt.yscale("log")
    plt.xlabel("t [s]")
    plt.ylabel("|net flux| [mol g$^{-1}$ s$^{-1}$]")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"diag_netflux_{tag}.png", dpi=200)

def main():
    rho0 = 1e6
    X0 = np.zeros(6)
    X0[He4] = 0.20
    X0[O16] = 0.80
    Y0 = X0 / A

    points = [
        (3.7, 0.03, "ridge"),
        (2.0, 0.03, "cool"),
        (5.2, 0.03, "hot"),
    ]

    for T9p, tau, tag in points:
        out = run_onezone_diagnostic(T9p, tau, rho0, Y0, n_eval=800)
        plot_diagnostic(out, tag)
        print(tag, "phis =", out["phis"], "final X32 =", out["Xt"][-1, S32])




if __name__ == "__main__":
    main()
