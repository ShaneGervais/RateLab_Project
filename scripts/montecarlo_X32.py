import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from ratelab.network import AlphaChain, He4, Mg24, Ne20, O16, Si28, S32
from ratelab.trajectory import shock_trajectory

# import reaction coefficients from scan_trajectory (need to run dump_reaclib_sets.py to get them for yourself)
from scan_trajectory import coef

# atomic masses
A = np.array([4, 16, 20, 24, 28, 32], dtype=float)

# same logic from sensitivity_oneat.py
def Y_to_X(Y):
    AY = A * Y
    s = np.sum(AY)
    if s <= 0:
        return np.full_like(Y, np.nan, dtype=float)
    return AY / s

def run_onezone_final(net, Y0, rho0, T9_peak, tau, rtol=1e-7, atol=1e-12):
    """Fast one-zone: integrate and only return final Y."""
    T9_traj  = lambda t: shock_trajectory(t, T9_peak=T9_peak, tau=tau)
    rho_traj = lambda t: rho0
    t_end = max(8.0 * tau, 0.2)

    sol = solve_ivp(
        fun=lambda t, y: net.rhs(t, y, T9_traj=T9_traj, rho_traj=rho_traj),
        t_span=(0.0, t_end),
        y0=Y0,
        method="BDF",
        rtol=rtol,
        atol=atol,
        t_eval=[t_end],
    )
    if not sol.success:
        return None
    return sol.y[:, -1]

def compute_phis(net, T9_peak, tau, rho0, Y0, n_eval=200, rtol=1e-7, atol=1e-12):
    """
    Slower diagnostic run: returns (Y(t), t, phis).
    Call this only for a handful of interesting points.
    """
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
        dense_output=True,   # key point: enables sol.sol(t)
    )

    t = np.linspace(0.0, t_end, n_eval)
    Yt = sol.sol(t)   # shape (6, n_eval)

    # flux time series
    F = [net.fluxes(tk, Yt[:,k], T9_traj=T9_traj, rho_traj=rho_traj) for k, tk in enumerate(t)]
    def get(name): return np.array([Fi.get(name, 0.0) for Fi in F], dtype=float)

    net_O16  = get("O16_ag_Ne20")  - get("Ne20_ga_O16")
    net_Ne20 = get("Ne20_ag_Mg24") - get("Mg24_ga_Ne20")
    net_Mg24 = get("Mg24_ag_Si28") - get("Si28_ga_Mg24")
    net_Si28 = get("Si28_ag_S32")  - get("S32_ga_Si28")

    phi_O16  = float(np.trapezoid(net_O16,  t))
    phi_Ne20 = float(np.trapezoid(net_Ne20, t))
    phi_Mg24 = float(np.trapezoid(net_Mg24, t))
    phi_Si28 = float(np.trapezoid(net_Si28, t))

    return t, Yt, (phi_O16, phi_Ne20, phi_Mg24, phi_Si28)

def main():
    # -----------------------------
    # Burn setup (same as your sensitivity)
    # -----------------------------
    rho0 = 1e6
    T9_peak = 3.5
    tau = 0.2

    # Initial composition
    X0 = np.zeros(6)
    X0[He4] = 0.20
    X0[O16] = 0.80
    Y0 = X0 / A

    # -----------------------------
    # Monte Carlo settings
    # -----------------------------
    N = 500                 # increase later (e.g. 2000)
    seed = 12345
    rng = np.random.default_rng(seed)

    # Uncertainty model:
    # f = 10^(sigma_dex * Normal(0,1))
    # sigma_dex = 0.3 means ~factor 2 (1-sigma) in dex space.
    sigma_dex = 0.3

    # Choose which rates to randomize (based on your sensitivity ranking)
    varied = [
        "si28_ag_s32",
        "s32_ga_si28",
        "mg24_ag_si28",
        "ne20_ga_o16",   # optional but it mattered a lot in your run
    ]

    # Output arrays
    f_draws = np.zeros((N, len(varied)), dtype=float)
    X32 = np.full(N, np.nan, dtype=float)
    X28 = np.full(N, np.nan, dtype=float)
    XO16 = np.full(N, np.nan, dtype=float)
    success = np.zeros(N, dtype=int)

    net = AlphaChain(coef)

    # -----------------------------
    # Run MC
    # -----------------------------
    for i in range(N):
        # reset all factors
        for k in net.f:
            net.f[k] = 1.0

        # draw multipliers
        z = rng.normal(0.0, 1.0, size=len(varied))
        f = 10.0 ** (sigma_dex * z)
        f_draws[i, :] = f

        for name, val in zip(varied, f):
            net.f[name] = float(val)

        Yf = run_onezone_final(net, Y0, rho0, T9_peak, tau)
        if Yf is None:
            continue

        Xf = Y_to_X(Yf)
        X32[i] = float(Xf[S32])
        X28[i] = float(Xf[Si28])
        XO16[i] = float(Xf[O16])
        success[i] = 1

    n_ok = int(np.sum(success))
    print(f"MC done: {n_ok}/{N} successful integrations")

    # -----------------------------
    # Save CSV
    # -----------------------------
    out_csv = "mc_results.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["i", "success"] + [f"f_{name}" for name in varied] + ["X32", "X28", "XO16"])
        for i in range(N):
            w.writerow([i, int(success[i])] + list(map(float, f_draws[i, :])) + [X32[i], X28[i], XO16[i]])
    print(f"Wrote {out_csv}")

    # -----------------------------
    # Correlations (log space)
    # -----------------------------
    mask = (success == 1) & np.isfinite(X32) & (X32 > 0.0)
    X32m = X32[mask]
    Fm = f_draws[mask, :]

    # avoid log(0)
    eps = 1e-40
    y = np.log10(X32m + eps)
    X = np.log10(Fm)

    # Pearson correlation corr(log f_i, log X32)
    corr = np.zeros(len(varied))
    for j in range(len(varied)):
        xj = X[:, j]
        xj = xj - np.mean(xj)
        yy = y - np.mean(y)
        denom = np.sqrt(np.sum(xj*xj) * np.sum(yy*yy))
        corr[j] = float(np.sum(xj*yy) / denom) if denom > 0 else np.nan

    print("Correlation with log10(X32):")
    for name, c in zip(varied, corr):
        print(f"  {name:14s}  corr = {c:+.3f}")

    # simple linear regression: y ~ b0 + sum b_j * X_j
    # (interpretable as sensitivities in log space)
    Areg = np.column_stack([np.ones(X.shape[0]), X])
    beta, *_ = np.linalg.lstsq(Areg, y, rcond=None)
    b0 = beta[0]
    bj = beta[1:]

    print("Linear model: log10(X32) ≈ b0 + Σ b_j log10(f_j)")
    print(f"  b0 = {b0:+.3f}")
    for name, b in zip(varied, bj):
        print(f"  {name:14s}  b = {b:+.3f}")

    # -----------------------------
    # Plots
    # -----------------------------
    plt.figure()
    plt.hist(X32m, bins=40)
    plt.yscale("log")
    plt.xlabel(r"final $X(^{32}\mathrm{S})$")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig("mc_hist_X32.png", dpi=300)
    print("Saved mc_hist_X32.png")

    plt.figure()
    idx = np.arange(len(varied))
    plt.bar(idx, corr)
    plt.xticks(idx, varied, rotation=45, ha="right")
    plt.ylim(-1.0, 1.0)
    plt.ylabel(r"corr($\log_{10} f$, $\log_{10} X_{32}$)")
    plt.tight_layout()
    plt.savefig("mc_corr_X32.png", dpi=300)
    print("Saved mc_corr_X32.png")

if __name__ == "__main__":
    main()
