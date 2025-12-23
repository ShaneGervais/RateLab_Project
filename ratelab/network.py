import numpy as np
from .rates import reaclib_rate_scalar

# indices
He4, O16, Ne20, Mg24, Si28, S32 = range(6)

class AlphaChain:
    """
    Tiny alpha-chain toy network:

      16O(a,g)20Ne(a,g)24Mg(a,g)28Si(a,g)32S
    plus the reverse photodisintegration reactions (g,a).

    This class pre-binds coefficient arrays so the ODE RHS avoids dict lookups.
    """

    def __init__(self, coef: dict):
        # forward
        self.C_O16  = np.asarray(coef["o16_ag_ne20"], dtype=float)
        self.C_Ne20 = np.asarray(coef["ne20_ag_mg24"], dtype=float)
        self.C_Mg24 = np.asarray(coef["mg24_ag_si28"], dtype=float)
        self.C_Si28 = np.asarray(coef["si28_ag_s32"], dtype=float)
        # reverse
        self.C_Ne20r = np.asarray(coef["ne20_ga_o16"], dtype=float)
        self.C_Mg24r = np.asarray(coef["mg24_ga_ne20"], dtype=float)
        self.C_Si28r = np.asarray(coef["si28_ga_mg24"], dtype=float)
        self.C_S32r  = np.asarray(coef["s32_ga_si28"], dtype=float)

        # multiplicative factors for sensitivity studies (default = 1)
        self.f = {
            "o16_ag_ne20": 1.0,
            "ne20_ag_mg24": 1.0,
            "mg24_ag_si28": 1.0,
            "si28_ag_s32": 1.0,
            "ne20_ga_o16": 1.0,
            "mg24_ga_ne20": 1.0,
            "si28_ga_mg24": 1.0,
            "s32_ga_si28": 1.0,
        }


    def rates(self, T9: float):

        # scalar rates (fast path)
        r_o16  = reaclib_rate_scalar(T9, self.C_O16)  * self.f["o16_ag_ne20"]
        r_ne20 = reaclib_rate_scalar(T9, self.C_Ne20) * self.f["ne20_ag_mg24"]
        r_mg24 = reaclib_rate_scalar(T9, self.C_Mg24) * self.f["mg24_ag_si28"]
        r_si28 = reaclib_rate_scalar(T9, self.C_Si28) * self.f["si28_ag_s32"]

        lam_ne20 = reaclib_rate_scalar(T9, self.C_Ne20r) * self.f["ne20_ga_o16"]
        lam_mg24 = reaclib_rate_scalar(T9, self.C_Mg24r) * self.f["mg24_ga_ne20"]
        lam_si28 = reaclib_rate_scalar(T9, self.C_Si28r) * self.f["si28_ga_mg24"]
        lam_s32  = reaclib_rate_scalar(T9, self.C_S32r)  * self.f["s32_ga_si28"]


        return r_o16, r_ne20, r_mg24, r_si28, lam_ne20, lam_mg24, lam_si28, lam_s32

    def rhs(self, t, Y, T9_traj, rho_traj):
        """
        RHS for solve_ivp: dY/dt.

        Parameters
        ----------
        t : float
        Y : array(6,) molar abundances (mol/g)
        T9_traj : callable t->T9
        rho_traj: callable t->rho
        """
        T9 = float(T9_traj(t))
        rho = float(rho_traj(t))

        r_o16, r_ne20, r_mg24, r_si28, lam_ne20, lam_mg24, lam_si28, lam_s32 = self.rates(T9)

        dY = np.zeros_like(Y)

        # ---- forward (a,g): i + alpha -> k ----
        f = rho * Y[He4] * Y[O16]  * r_o16
        dY[He4] -= f; dY[O16]  -= f; dY[Ne20] += f

        f = rho * Y[He4] * Y[Ne20] * r_ne20
        dY[He4] -= f; dY[Ne20] -= f; dY[Mg24] += f

        f = rho * Y[He4] * Y[Mg24] * r_mg24
        dY[He4] -= f; dY[Mg24] -= f; dY[Si28] += f

        f = rho * Y[He4] * Y[Si28] * r_si28
        dY[He4] -= f; dY[Si28] -= f; dY[S32]  += f

        # ---- reverse (g,a): k -> i + alpha ----
        g = lam_ne20 * Y[Ne20]
        dY[Ne20] -= g; dY[O16]  += g; dY[He4] += g

        g = lam_mg24 * Y[Mg24]
        dY[Mg24] -= g; dY[Ne20] += g; dY[He4] += g

        g = lam_si28 * Y[Si28]
        dY[Si28] -= g; dY[Mg24] += g; dY[He4] += g

        g = lam_s32 * Y[S32]
        dY[S32]  -= g; dY[Si28] += g; dY[He4] += g

        return dY

    def fluxes(self, t, Y, T9_traj, rho_traj):
        """
        Forward and reverse fluxes at time t for diagnostics.
        Units here are "per second in Y-space" (since Y is mol/g).
        """
        T9 = float(T9_traj(t))
        rho = float(rho_traj(t))
        r_o16, r_ne20, r_mg24, r_si28, lam_ne20, lam_mg24, lam_si28, lam_s32 = self.rates(T9)

        F = {}
        F["O16_ag_Ne20"]  = rho * Y[He4] * Y[O16]  * r_o16
        F["Ne20_ag_Mg24"] = rho * Y[He4] * Y[Ne20] * r_ne20
        F["Mg24_ag_Si28"] = rho * Y[He4] * Y[Mg24] * r_mg24
        F["Si28_ag_S32"]  = rho * Y[He4] * Y[Si28] * r_si28

        F["Ne20_ga_O16"]  = lam_ne20 * Y[Ne20]
        F["Mg24_ga_Ne20"] = lam_mg24 * Y[Mg24]
        F["Si28_ga_Mg24"] = lam_si28 * Y[Si28]
        F["S32_ga_Si28"]  = lam_s32  * Y[S32]
        return F
    
    def rhs_with_phis(self, t, y, T9_traj, rho_traj):
        """
        Augmented RHS:
          y[:6]   = Y (mol/g)
          y[6:10] = phi integrals of net fluxes:
                   [O16<->Ne20, Ne20<->Mg24, Mg24<->Si28, Si28<->S32]
        """
        Y = y[:6]
        T9  = float(T9_traj(t))
        rho = float(rho_traj(t))

        r_o16, r_ne20, r_mg24, r_si28, lam_ne20, lam_mg24, lam_si28, lam_s32 = self.rates(T9)

        # forward fluxes
        F_O16_f  = rho * Y[He4] * Y[O16]  * r_o16
        F_Ne20_f = rho * Y[He4] * Y[Ne20] * r_ne20
        F_Mg24_f = rho * Y[He4] * Y[Mg24] * r_mg24
        F_Si28_f = rho * Y[He4] * Y[Si28] * r_si28

        # reverse fluxes
        F_Ne20_r = lam_ne20 * Y[Ne20]
        F_Mg24_r = lam_mg24 * Y[Mg24]
        F_Si28_r = lam_si28 * Y[Si28]
        F_S32_r  = lam_s32  * Y[S32]

        # original Y rhs
        dY = self.rhs(t, Y, T9_traj=T9_traj, rho_traj=rho_traj)

        # phi derivatives = net flux along each link
        dphi = np.array([
            F_O16_f  - F_Ne20_r,
            F_Ne20_f - F_Mg24_r,
            F_Mg24_f - F_Si28_r,
            F_Si28_f - F_S32_r,
        ], dtype=float)

        return np.concatenate([dY, dphi])
    
    def set_factor(self, name: str, value: float):
        self.f[name] = float(value)



# Backwards-compatible function wrappers (match your scripts)

def rhs(t, Y, coef, T9_traj, rho_traj):
    """
    Convenience wrapper matching your existing call sites:
      rhs(t, Y, coef, T9_traj=..., rho_traj=...)
    """
    net = AlphaChain(coef)  # NOTE: constructing per call is slow; prefer using AlphaChain directly.
    return net.rhs(t, Y, T9_traj=T9_traj, rho_traj=rho_traj)

def fluxes(t, Y, coef, T9_traj, rho_traj):
    net = AlphaChain(coef)
    return net.fluxes(t, Y, T9_traj=T9_traj, rho_traj=rho_traj)
