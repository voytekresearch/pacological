#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
from brian2 import *
from numpy.random import uniform, random_integers, lognormal
"""Gain modulation in a HH neuron. 

Parameters for balance taken from Chance, Abbott and Rayes, Neuron, 2002. 
"""

def gain(time, r_e=675, r_i=675, w_e=40.0, w_i=160.0, w_m=0, I_drive=0, f=0, 
        verbose=True):

    time_step = 0.01 * ms
    defaultclock.dt = time_step

    # -- params
    # User 
    w_e = w_e * usiemens  
    w_i = w_i * usiemens

    # Fixed
    # The PoissonGroup have max rates limit, which can be 
    # avoided by 'stacking' groups
    N = 10  

    r_e = (r_e / N) * Hz  
    r_i = (r_i / N) * Hz

    Et = 20 * mvolt

    # Synapse
    Ee = 0 * mvolt
    Ei = -80 * mvolt

    tau_m = 20 * ms
    tau_ampa = 5 * ms  # Setting these equal simphhies balancing (for now)
    tau_gaba = 10 * ms

    # HH specific
    Cm = 1 * uF  # /cm2

    w_m = w_m * msiemens 
    g_Na = 100 * msiemens
    g_K = 80 * msiemens
    g_l = 0.1 * msiemens

    V_Na = 50 * mV
    V_K = -100 * mV
    V_l = -67 * mV  # 67 mV 
 
    V_i = -80 * mV
    V_e = 0 * mV

    hh = """
    dV/dt = (I_Na + I_K + I_l + I_m + I_syn + I) / Cm : volt
    """ + """
    I_Na = g_Na * (m ** 3) * h * (V_Na - V) : amp
    m = a_m / (a_m + b_m) : 1
    a_m = (0.32 * (54 + V/mV)) / (1 - exp(-0.25 * (V/mV + 54))) / ms : Hz
    b_m = (0.28 * (27 + V/mV)) / (exp(0.2 * (V/mV + 27)) - 1) / ms : Hz
    h = clip(1 - 1.25 * n, 0, inf) : 1
    """ + """
    I_K = g_K * n ** 4 * (V_K - V) : amp
    dn/dt = (a_n - (a_n * n)) - b_n * n : 1
    a_n = (0.032 * (52 + V/mV)) / (1 - exp(-0.2 * (V/mV + 52))) / ms : Hz
    b_n = 0.5 * exp(-0.025 * (57 + V/mV)) / ms : Hz
    """ + """
    I_l = g_l * (V_l - V) : amp
    """ + """
    I_m = w_m * w * (V_K - V) : amp
    dw/dt = (w_inf - w) / tau_w/ms : 1
    w_inf = 1 / (1 + exp(-1 * (V/mV + 35) / 10)) : 1
    tau_w = 400 / ((3.3 * exp((V/mV + 35)/20)) + (exp(-1 * (V/mV + 35) / 20))) : 1
    """ + """
    I_syn = g_e * (V_e - V) + g_i * (V_i - V) : amp
    dg_e/dt = -g_e / tau_ampa : siemens
    dg_i/dt = -g_i / tau_gaba : siemens
    I : amp
    """
 
    # The background noise
    if f > 0:
        f = f * Hz
        P_be = NeuronGroup(N, 'rates = r_e * cos(2 * pi * f * t) : Hz',
                threshold='rand()<rates*dt')
        P_bi = NeuronGroup(N, 'rates = r_i * cos(2 * pi * f * t) : Hz',
                threshold='rand()<rates*dt')
    else:
        P_be = PoissonGroup(N, r_e)
        P_bi = PoissonGroup(N, r_i)

    # Our one neuron to gain control
    P_e = NeuronGroup(1, hh, threshold='V > Et', refractory=2 * ms)
    P_e.V = V_l
    P_e.I = I_drive * uamp

    # Set up the 'network' 
    C_be = Synapses(P_be, P_e, pre='g_e += w_e')
    C_be.connect(True)
    C_bi = Synapses(P_bi, P_e, pre='g_i += w_i')
    C_bi.connect(True)

    # Data acq
    spikes_e = SpikeMonitor(P_e)
    traces_e = StateMonitor(P_e, ['V', 'g_e', 'g_i'], record=True)

    report = 'text'
    if not verbose:
        report = None

    run(time * second, report=report)

    return {'spikes' : spikes_e, 'traces' : traces_e}


def exp(t, I, xfactor, f=0, r=675, w_e=40.0, w_i=160.0):
    """Experiments in balance.
    
    Params
    ------
    t : scalar
        Run time (seconds)
    I : scalar
        Drive current (1)
    xfactor : scalar, 2-tuple (xf_e, xf_i)
        Gain multplication factor
    f : scalar (optional)
        Oscillation frequency (set to 0 to turn off)
    r : scalar, 2-tuple (r_e, r_i)
        Background firing rate
    g_l : scalar
        Background conductance
    """
    t = float(t)
    I = float(I)
    w_e = float(w_e)
    w_i = float(w_i)

    try:
        xf1, xf2 = xfactor
    except TypeError:
        xf1, xf2 = xfactor, xfactor
    
    try:
        r_e, r_i = r
    except TypeError:
        r_e, r_i = r, r
    
    r_e = r_e * xf1  # scale rates
    r_i = r_i * xf2

    return gain(t, r_e=r_e, r_i=r_i, 
                w_e=w_e, w_i=w_i, 
                I_drive=I,
                f=f,
                verbose=False)


if __name__ == "__main__":
    import pylab as plt
    plt.ion()

    res = gain(1, I_drive=0)

