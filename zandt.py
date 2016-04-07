from __future__ import division

from scipy.integrate import odeint, ode
from sdeint import itoint
import numpy as np
from numpy.random import normal, poisson
from copy import deepcopy
from fakespikes.rates import stim
from pacological.util import create_I, ornstein_uhlenbeck
from functools import partial
from convenience.numpy import save_hdfz, load_hdfz


def Idist(I, Iu, Isigma):
    a =  (1 / (Isigma * np.sqrt(2 * np.pi)))
    return a * np.exp(-0.5 * ((I - Iu) / (Isigma)) ** 2)


FIPATH = "/home/ejp/src/pacological/data/exp225/r400_wm0.hdf5"
def phi_t(I, Iu, Isigma):

    # that doesn't rely on loading the
    # data from file with each dt
    dat = load_hdfz(FIPATH)
    rates = dat['rates']
    Is = dat['Is']

    # For Is, convolve Idist with rates creating rates_t
    # and then linearly interpolate to get I.
    Id = Idist(Is, Iu, Isigma)
    rates_t = np.convolve(Id, rates, mode)

    return np.interp(I, Is, rates_t)  


def zandt(ys, t, Istim, w_ee, w_ei, w_ie, w_ii, w_be, w_bi,
        rbe, rbi, f=0, I_e=120, I_i=85):

    # --
    # Unpack 'ys' and reset for the next dt
    [re1, ri1, g_ee1, g_ie1, g_ei1, g_ii1, g_be1, g_bi1, s_be1, s_bi1] = ys
    ys = np.zeros_like(ys)

    # --
    # Params
    tau_ampa1 = 5 / 1000.  # 5 ms  
    tau_gaba1 = 10 / 1000.  # 10 ms
    tau_m = 2 / 1000.  # fast effective membrane conductance; try 20 too?

    Vth = -55
    Vampa = 0
    Vgaba = -80

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

    # --
    # Rates, finally
    ys[0] = (-re1 + fi_t(I_re + I_e + I, I_b, S_b)) / tau_m 
    ys[1] = (-ri1 + fi_t(I_ri + I_i, I_b, S_b)) / tau_m  
    # TODO do I want to use fi_t for ri, or something more traditional?
    # TODO leave it for the first runs?

    return ys


if __name__ == "__main__":
    from functools import partial

    # Run
    ys0 = np.asarray([8, 12.0, 1, 1, 1, 1, 1, 1, 1, 1])
    tmax = 2  # run time, second
    dt = 1. / 10000  # resolution, ms
    times = np.linspace(0, tmax, tmax / dt)

    # Stim params
    d = 1  # drive rate (want 0-1)
    scale = .01 * d
    Istim = create_I(tmax, d, scale, dt=dt, seed=1)

    # Simulate
    f = partial(zandt, Istim=Istim, w_ee=1, w_ei=1, w_ie=1, w_ii=1, 
            w_be=1, w_bi=1, rbe=10, rbi=10, f=0, I_e=2, I_i=1)
    g = partial(ornstein_uhlenbeck, sigma=1.0, loc=[0, 1]) 

    ys = itoint(f, g, ys0, times)

