"""A hydrid of Chance and Zandt,

a conductance neural mass system (based on Zandt 2014) whose non-linearty 
is (paritally) defined by the gain relation in Chance 2002
"""
from __future__ import division

from scipy.integrate import odeint, ode
from sdeint import itoint

import numpy as np
from numpy import random
from copy import deepcopy

from fakespikes.rates import stim
from pacological.util import create_stim_I, ornstein_uhlenbeck
from pacological.util import phi as phi_i
# from convenience.numpy import save_hdfz, load_hdfz

# Control background 
SEED = 42
prng = random.RandomState(SEED)

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
    z = 1 / a * logistic(Isyn, I, 1/k, 2000)      # supp to explore 2000?

    # Transform into population activity
    # Isigma = Isyn * 0.1  # TODO
    # Id = Idist(Isyn, I, Isigma)

    return z


def _background(t, f, rbe, rbi):
    if f > 0:
        # oscillation
        rbe_t = rbe * np.cos(2 * np.pi * f * t)
        rbi_t = rbi * np.cos(2 * np.pi * f * t)
    else:
        # fixed rate
        rbe_t = deepcopy(rbe)
        rbi_t = deepcopy(rbi)

    if rbe_t < 0:
        rbe_t = 0
    if rbi_t < 0:
        rbi_t = 0
    
    # 2nd term adds fix background
    rbe_t = prng.poisson(rbe_t, 1) + prng.poisson(135, 1) 
    rbi_t = prng.poisson(rbi_t, 1) + prng.poisson(135, 1)

    return np.asarray([rbe_t[0], rbe_t[0]]) 


def cz(ys, t, Istim, w_ee, w_ei, w_ie, w_ii, w_be, w_bi, rbe, rbi, f, 
        I_e=400e-9, I_i=300e-9):

    # print("\n>>> t : {}".format(t))

    # --
    # Params
    N = 10000

    Ne = N * 0.8
    Ni = N - Ne

    p = 0.02  # Assume connect p is constant (TODO explore var)
    Ce = Ne * p
    Ci = Ni * p    

    tau_ampa = 5e-3  # seconds
    tau_gaba = 10e-3
    tau_m = 2e-3  # fast effective membrane conductance; try 20 too?

    Vth = -55e-3  # volts
    Vampa = 0 
    Vgaba = -80e-3

    # Unpack ys
    [re, ri, g_ee, g_ie, g_ei, g_ii, g_be, g_bi, s_be, s_bi] = ys
    ys = np.zeros_like(ys)

    # print("g_ee {:.2e}, g_ei {:.2e}, g_ie {:.2e}, g_ii {:.2e}".format(g_ee, g_ie, g_ei, g_ii))
    # --    
    # Update rates, re and ri for this dt
    
    # Stim drives
    if Istim is None:
        I = 0.0
    else:
        I = Istim(t) 

    # Currents for phi
    I_re = g_ee * (Vampa - Vth) + g_ie * (Vgaba - Vth)
    I_ri = g_ii * (Vgaba - Vth) + g_ei * (Vampa - Vth)

    # Conductances for phi
    # Chance et al states that max ge/gi were 4-6% of g0 
    # (the resting conductance)so we use the max ge/gi to set g0....
    g0_e = np.abs(np.mean([Ce * w_be, Ci * w_bi]) / 0.04)
    gb_e = np.abs(g_be + g_bi + g_ee + g_ie)
    s_tot_e = s_be + s_bi

    g0_i = (Ci * w_ii + Ce * w_ei) /2/ 0.04
    gb_i = g_ii + g_ei
    s_tot_i = s_be + s_bi
    
    # print("I_syn_e : {:.2e}".format(I_re))
    # print("g0_e : {:.2e}".format(g0_e))
    # print("gb_e : {:.2e}".format(gb_e))
    # print("sb_e : {:.2e}".format(s_tot_e))

    # print("I_syn_i : {:.2e}".format(I_ri))
    # print("g0_i : {:.2e}".format(g0_i))
    # print("gb_i : {:.2e}".format(gb_i))
    # print("sb_i : {:.2e}".format(s_tot_i))

    # phi, z here
    z_re = phi(I_re + I, I_e, g0_e, gb_e, np.sqrt(s_tot_e))
    z_ri = phi(I_ri, I_i, g0_i, gb_i, np.sqrt(s_tot_i))    # should be s_bi?
    # print("phi_e: {:.2e}".format(z_re))
    # print("phi_i: {:.2e}".format(z_ri))

    ys[0] = (-re + z_re) / tau_m 
    ys[1] = (-ri + z_ri) / tau_m
    # print("rates_e : {:.2e}".format(ys[0]))
    # print("rates_i : {:.2e}".format(ys[1]))
    # --
    # Step equations for next dt
    # Internal
    ys[2] = (-g_ee / tau_ampa) + (Ce * w_ee * re) 
    ys[3] = (-g_ie / tau_gaba) + (Ci * w_ie * ri)
    ys[4] = (-g_ei / tau_ampa) + (Ce * w_ei * re)
    ys[5] = (-g_ii / tau_gaba) + (Ci * w_ii * ri)
    # print("UPDATE g_ee {:.2e}, g_ei {:.2e}, g_ie {:.2e}, g_ii {:.2e}".format(ys[2], ys[3], ys[4], ys[5]))
    # print("All the w_ie stuff - g_ie {}, tau_gaba {}, Ci {}, w_ie {}, re {}".format(g_ie, tau_gaba, Ci, w_ie, re))
    # import ipdb; ipdb.set_trace()
    
    rbe_t, rbi_t = _background(t, f, rbe, rbi)
    ys[6] = (-g_be / tau_ampa) + (Ce * w_be * rbe_t)
    ys[7] = (-g_bi / tau_gaba) + (Ci * w_bi * rbi_t)

    ys[8] = (-2 * s_be / tau_ampa) + ((Ce * w_be) ** 2 * rbe_t)
    ys[9] = (-2 * s_bi / tau_gaba) + ((Ci * w_bi) ** 2 * rbi_t)

    return ys

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    from functools import partial
    from scipy.integrate import odeint, ode
    from sdeint import itoint

    from pacological.chance_zandt import cz, phi
    from pacological.util import phi as phi_i
    from convenience.numpy import save_hdfz, load_hdfz

    from fakespikes.rates import stim
    from pacological.util import create_stim_I, ornstein_uhlenbeck

    # print(">>> Initializing constants")

    # Init time
    tmax = 1.0  # run time, second
    dt = 1e-4  # resolution, 0.1 ms
    times = np.linspace(0, tmax, int(tmax / dt))

    # Paramsz
    f=6
    I_e=400e-9
    I_i=300e-9
    # print("I_e {:.2e}, I_i {:.2e}".format(I_e, I_i))

    # Stim params
    d = 400e-9  # drive rate (want 0-1)
    scale = .001 * d
    Istim = create_stim_I(tmax, d, scale, dt=dt, seed=1)

    # Weights
    # w_ee = 0e-9
    # w_ei = 0e-9
    # w_ie = 0e-9
    # w_ii = 0e-9

    w_ee=2e-9
    w_ei=50e-9
    w_ie=23e-9
    w_ii=10e-9
    # print("w_ee {:.2e}, w_ei {:.2e}, w_ie {:.2e}, w_ii {:.2e}".format(w_ee, w_ei, w_ie, w_ii))

    w_be=400e-9
    w_bi=1600e-9
    rbe=135e2
    rbi=135e2

    # Run
    ys0 = np.asarray([8.0, 12.0, 
                      w_ee/2, w_ei/2, w_ie/2, w_ii/2, 
                      w_be/2, w_bi/2, w_be**2/2, w_bi**2/2])

    f_base = partial(cz, Istim=Istim, 
                w_ee=w_ee, w_ei=w_ei, w_ie=w_ie, w_ii=w_ii, 
                w_be=w_be, w_bi=w_bi, rbe=rbe, rbi=rbi, 
                f=0, 
                I_e=I_e, I_i=I_i)

    f = partial(cz, Istim=Istim, 
                w_ee=w_ee, w_ei=w_ei, w_ie=w_ie, w_ii=w_ii, 
                w_be=w_be, w_bi=w_bi, rbe=rbe, rbi=rbi, 
                f=f, 
                I_e=I_e, I_i=I_i)

    g = partial(ornstein_uhlenbeck, sigma=1, loc=[0, ]) 

    ys_base = itoint(f_base, g, ys0, times)
    ys = itoint(f, g, ys0, times)
