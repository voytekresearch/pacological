from __future__ import division

from scipy.integrate import odeint, ode
from sdeint import itoint

import numpy as np
from numpy.random import normal, poisson
from copy import deepcopy

from fakespikes.rates import stim
from pacological.util import create_stim_I, ornstein_uhlenbeck
from pacological.util import phi as phi_i
# from convenience.numpy import save_hdfz, load_hdfz

def Idist(I, Iu, Isigma):
    a =  (1 / (Isigma * np.sqrt(2 * np.pi)))
    return a * np.exp(-0.5 * ((I - Iu) / (Isigma)) ** 2)


def logistic(x, x0, k, L):
    return L / (1 + np.exp(-k * (x - x0)))


def phi(Isyn, I, g0, gs, sigma):
    tau = 2e-3  # second
    Vth = -55e-03  # volt
    Vreset = -60e-3 

    # Compute the single unit respose
    g = g0 + gs
    k = (Isyn / g * sigma)
    a = ((g * sigma) / (tau * g0 * (Vth - Vreset))) 
    z = a * logistic(Isyn, I, 1/k, 2000)

    # Transform into population activity
    # Isigma = Isyn * 0.1  # TODO
    # Id = Idist(Isyn, I, Isigma)

    return z


def chance(ys, t, Istim, w_ee, w_ei, w_ie, w_ii, w_be, w_bi, rbe, rbi, 
        f=0, I_e=120e-9, I_i=85e-9):
    # --
    # Params
    tau_ampa1 = 5e-3  # seconds
    tau_gaba1 = 10e-3
    tau_m = 2e-3  # fast effective membrane conductance; try 20 too?

    Vth = -55e-3  # volts
    Vampa = 0 
    Vgaba = -80e-3
    Vreset = -60e-3 
    
    # phi for I, Pop rate tune params
    c = 1.1  # lit?
    g = 1 / 10  # lit?

    # --
    # Unpack 'ys' and reset for the next dt
    [re1, ri1, g_ee1, g_ie1, g_ei1, g_ii1, g_be1, g_bi1, s_be1, s_bi1] = ys
    ys = np.zeros_like(ys)

    # --
    # Drives
    if Istim is None:
        I = 0.0
    else:
        I = Istim(t) 

    # --
    # Internal
    ys[2] = (-g_ee1 / tau_ampa1) + (w_ee * re1) 
    ys[3] = (-g_ie1 / tau_gaba1) + (w_ie * ri1)
    ys[4] = (-g_ei1 / tau_ampa1) + (w_ei * re1)
    ys[5] = (-g_ii1 / tau_gaba1) + (w_ii * ri1)
    I_re = g_ee1 * (Vampa - Vth) + g_ie1 * (Vgaba - Vth) 
    I_ri = g_ii1 * (Vgaba - Vth) + g_ei1 * (Vampa - Vth)
    print("I {0}, I_re {1}".format(I, I_re))

    # --
    # Background
    # Calculate both mean (gb_) and variance (sb_)
    if f > 0:
        # oscillation
        rbe_t = rbe * cos(2 * pi * f * t)
        rbi_t = rbi * cos(2 * pi * f * t)
    else:
        # fixed rate
        rbe_t = deepcopy(rbe)
        rbi_t = deepcopy(rbi)

    rbe_t = poisson(rbe_t, 1)  
    rbi_t = poisson(rbi_t, 1)

    ys[6] = (-g_be1 / tau_ampa1) + (w_be * rbe_t)
    ys[7] = (-g_bi1 / tau_gaba1) + (w_bi * rbi_t)
    I_b = (g_be1 * (Vampa - Vth) + g_bi1 * (Vgaba - Vth))

    ys[8] = (2 * (-s_be1 / tau_ampa1)) + (w_be ** 2 * rbe_t)
    ys[9] = (2 * (-s_bi1 / tau_gaba1)) + (w_bi ** 2 * rbi_t)
    S_b = (s_be1 * ((Vampa - Vth) ** 2) + s_bi1 * (Vgaba - Vth) ** 2)

    # print("I_b {0}, I {1}, S_b {2}, I_re {3}, I_ri {4}, re {5}, ri {6}".format(
    #     I_b, I, S_b, I_re, I_ri, re1, ri1))

    # --
    # Rates, finally
    # Chance states ge/gi were 4-6% of g0 so
    # I use their average ge/gi max to set
    # g0
    g0 = np.mean(w_be + w_bi) / 0.05  
    gb = g_be1 + g_bi1
    # print("g0 {0}, gb {1}".format(g0, gb))

    z_re1 = phi(I_re + I, I_e, g0, gb, S_b)
    z_rei1 = phi_i(I_ri, I_i, c, g)

    # print("z_re {}".format(z_re1))

    ys[0] = (-re1 + z_re1) / tau_m 
    ys[1] = (-ri1 + z_rei1) / tau_m  # approx linear

    return ys
