import numpy as np

def shock_trajectory(t, T9_peak=5.0, tau=0.15, T9_floor=0.05):
    """
    Docstring for shock_trajectory

    Simple shock exponential cooling trajectory
    
    :param t: Description
    :param T9_peak: Description
    :param tau: Description
    :param T9_floor: Description
    """

    T9 = T9_peak*np.exp(-t/tau)
    return np.maximum(T9, T9_floor)