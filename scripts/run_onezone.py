import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from ratelab.trajectory import shock_trajectory
from ratelab.network import rhs, He4, O16, Ne20, Mg24, Si28, S32

