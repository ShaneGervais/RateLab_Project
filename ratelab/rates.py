import math
import numpy as np

def _as_2d(a_sets: np.ndarray) -> np.ndarray:
    a = np.asarray(a_sets, dtype=float)
    if a.ndim == 1:
        a = a[None, :]
    if a.shape[1] != 7:
        raise ValueError(f"Expected REACLIB coefficients with 7 columns, got shape {a.shape}")
    return a

# ---------- Scalar (fast path for ODE RHS) ----------

def reaclib_basis_scalar(T9: float):
    """
    Precompute common temperature basis terms for REACLIB evaluation at scalar T9.

    Returns:
      invT9, T9m13, T913, T9, T953, lnT9
    """
    if T9 <= 0.0:
        raise ValueError("T9 must be > 0")
    T913 = T9 ** (1.0/3.0)
    T9m13 = 1.0 / T913
    invT9 = 1.0 / T9
    # T9^(5/3) = T9 * T9^(2/3) = T9 * (T9^(1/3))^2
    T953 = T9 * (T913 * T913)
    lnT9 = math.log(T9)
    return invT9, T9m13, T913, T9, T953, lnT9

def reaclib_rate_scalar(T9: float, a_sets) -> float:
    """
    Full REACLIB rate (scalar T9): r(T9) = sum_k exp(E_k(T9)),
    where each E_k is the 7-parameter polynomial in T9 powers and ln(T9).
    """
    a = _as_2d(a_sets)
    invT9, T9m13, T913, T9v, T953, lnT9 = reaclib_basis_scalar(float(T9))

    s = 0.0
    # tight Python loop + math.exp is typically faster than vectorized numpy for scalar RHS calls
    for (a1,a2,a3,a4,a5,a6,a7) in a:
        E = (a1 + a2*invT9 + a3*T9m13 + a4*T913 + a5*T9v + a6*T953 + a7*lnT9)
        s += math.exp(E)
    return s

def dlnrate_dlnT9_scalar(T9: float, a_sets) -> float:
    """
    Temperature sensitivity: d ln r / d ln T9 for scalar T9.
    Uses analytic derivative for each term and combines with weights exp(E_k)/r.
    """
    a = _as_2d(a_sets)
    invT9, T9m13, T913, T9v, T953, lnT9 = reaclib_basis_scalar(float(T9))

    # derivative of basis terms w.r.t T9:
    # d(invT9)/dT9 = -1/T9^2 = -invT9^2
    dinv = -invT9*invT9
    # T9^(1/3) = T913 => d/dT9 = (1/3) T9^(-2/3) = (1/3) / (T913^2)
    dT913 = (1.0/3.0) / (T913*T913)
    # T9^(-1/3) = T9m13 => d/dT9 = (-1/3) T9^(-4/3) = (-1/3) * T9m13 / T9
    dT9m13 = (-1.0/3.0) * (T9m13 * invT9)
    # d(T9)/dT9 = 1
    dT9 = 1.0
    # T9^(5/3) = T953 => d/dT9 = (5/3) T9^(2/3) = (5/3) * (T913^2)
    dT953 = (5.0/3.0) * (T913*T913)
    # d(lnT9)/dT9 = 1/T9 = invT9
    dln = invT9

    # accumulate r and dr/dT9
    r = 0.0
    dr = 0.0
    for (a1,a2,a3,a4,a5,a6,a7) in a:
        E = (a1 + a2*invT9 + a3*T9m13 + a4*T913 + a5*T9v + a6*T953 + a7*lnT9)
        ek = math.exp(E)
        r += ek
        dE = (a2*dinv + a3*dT9m13 + a4*dT913 + a5*dT9 + a6*dT953 + a7*dln)
        dr += ek * dE

    if r == 0.0:
        return float("nan")
    # d ln r / d ln T9 = (T9 / r) * dr/dT9
    return float(T9v * dr / r)

# ---------- Vectorized (for plotting / grids) ----------

def reaclib_rate(T9, a_sets):
    """
    Vectorized REACLIB evaluation.

    Parameters
    ----------
    T9 : float or array
    a_sets : shape (7,) or (nset,7)

    Returns
    -------
    r : float or ndarray
    """
    if np.isscalar(T9):
        return reaclib_rate_scalar(float(T9), a_sets)

    T9 = np.asarray(T9, dtype=float)
    a = _as_2d(a_sets)

    T913 = T9 ** (1.0/3.0)
    T9m13 = 1.0 / T913
    invT9 = 1.0 / T9
    T953 = T9 * (T913*T913)
    lnT9 = np.log(T9)

    # broadcast: (nset, nT)
    E = (a[:,0,None]
         + a[:,1,None]*invT9[None,:]
         + a[:,2,None]*T9m13[None,:]
         + a[:,3,None]*T913[None,:]
         + a[:,4,None]*T9[None,:]
         + a[:,5,None]*T953[None,:]
         + a[:,6,None]*lnT9[None,:])
    return np.sum(np.exp(E), axis=0)

def dlnrate_dlnT9(T9, a_sets):
    """
    Vectorized d ln r / d ln T9 (mainly for plotting; scalar fast path used in ODE RHS).
    """
    if np.isscalar(T9):
        return dlnrate_dlnT9_scalar(float(T9), a_sets)

    T9 = np.asarray(T9, dtype=float)
    a = _as_2d(a_sets)

    T913 = T9 ** (1.0/3.0)
    T9m13 = 1.0 / T913
    invT9 = 1.0 / T9
    T953 = T9 * (T913*T913)
    lnT9 = np.log(T9)

    E = (a[:,0,None]
         + a[:,1,None]*invT9[None,:]
         + a[:,2,None]*T9m13[None,:]
         + a[:,3,None]*T913[None,:]
         + a[:,4,None]*T9[None,:]
         + a[:,5,None]*T953[None,:]
         + a[:,6,None]*lnT9[None,:])

    ek = np.exp(E)
    r = np.sum(ek, axis=0)

    dinv = -(invT9**2)
    dT913 = (1.0/3.0) / (T913**2)
    dT9m13 = (-1.0/3.0) * (T9m13 * invT9)
    dT953 = (5.0/3.0) * (T913**2)
    dln = invT9

    dE = (a[:,1,None]*dinv[None,:]
          + a[:,2,None]*dT9m13[None,:]
          + a[:,3,None]*dT913[None,:]
          + a[:,4,None]*1.0
          + a[:,5,None]*dT953[None,:]
          + a[:,6,None]*dln[None,:])
    dr = np.sum(ek * dE, axis=0)

    return T9 * dr / r

def dlnrate_dlnT9_fd(T9, a_sets, eps=1e-6):
    """
    Finite-difference cross-check (handy for tests).
    """
    T9 = np.asarray(T9, dtype=float)
    rp = reaclib_rate(T9*(1+eps), a_sets)
    rm = reaclib_rate(T9*(1-eps), a_sets)
    return (np.log(rp) - np.log(rm)) / (2*eps)
