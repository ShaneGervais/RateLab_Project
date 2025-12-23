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
    out["net"] = net
    out["T9_traj"] = T9_traj
    out["rho_traj"] = rho_traj
    out["t"] = t
    out["Yt"] = Yt
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

def plot_forward_reverse(out, tag):
    import numpy as np
    import matplotlib.pyplot as plt

    t  = out["t"]     # (n,)
    Yt = out["Yt"]    # (6, n)
    T9_traj  = out["T9_traj"]
    rho_traj = out["rho_traj"]
    net = out["net"]  # AlphaChain instance

    # Build forward/reverse flux arrays by evaluating net.fluxes at each time
    names = [
        ("O16_ag_Ne20",  "Ne20_ga_O16",  "O16 ↔ Ne20"),
        ("Ne20_ag_Mg24", "Mg24_ga_Ne20", "Ne20 ↔ Mg24"),
        ("Mg24_ag_Si28", "Si28_ga_Mg24", "Mg24 ↔ Si28"),
        ("Si28_ag_S32",  "S32_ga_Si28",  "Si28 ↔ S32"),
    ]

    Ff = {fwd: np.zeros_like(t, dtype=float) for fwd,_,_ in names}
    Fr = {rev: np.zeros_like(t, dtype=float) for _,rev,_ in names}

    for k, tk in enumerate(t):
        F = net.fluxes(tk, Yt[:, k], T9_traj=T9_traj, rho_traj=rho_traj)
        for fwd, rev, _lab in names:
            Ff[fwd][k] = float(F[fwd])
            Fr[rev][k] = float(F[rev])

    plt.figure(figsize=(7, 5))
    for fwd, rev, lab in names:
        plt.loglog(t, np.abs(Ff[fwd]) + 1e-300, label=f"{lab} forward")
        plt.loglog(t, np.abs(Fr[rev]) + 1e-300, ls="--", label=f"{lab} reverse")

    plt.xlabel("t [s]")
    plt.ylabel(r"Flux magnitude $|F|$  [1/s in Y-space]")
    plt.title(f"Forward vs reverse fluxes ({tag})")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"onezone_fwd_rev_{tag}.png", dpi=300)
    print(f"Saved onezone_fwd_rev_{tag}.png")




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
        (4.5, 0.10, "hot_slowcool"),
    ]

    for T9p, tau, tag in points:
        out = run_onezone_diagnostic(T9p, tau, rho0, Y0, n_eval=800)
        plot_diagnostic(out, tag)
        print(tag, "phis =", out["phis"], "final X32 =", out["Xt"][-1, S32])
        phi_O16, phi_Ne20, phi_Mg24, phi_Si28 = out["phis"]
        X32_final = out["Xt"][-1, S32]

        print(f"\n[{tag}]  T9_peak={T9p:.3f}  tau={tau:.4g}  rho0={rho0:.3g}")
        print(f"  phis:")
        print(f"    phi_O16  (O16→Ne20) = {phi_O16:.6g}")
        print(f"    phi_Ne20 (Ne20→Mg24)= {phi_Ne20:.6g}")
        print(f"    phi_Mg24 (Mg24→Si28)= {phi_Mg24:.6g}")
        print(f"    phi_Si28 (Si28→S32) = {phi_Si28:.6g}")
        print(f"  final X32 = {X32_final:.6g}")
        plot_diagnostic(out, tag)
        plot_forward_reverse(out, tag)





if __name__ == "__main__":
    main()
