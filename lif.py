#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
from brian2 import *
from numpy.random import uniform, random_integers, lognormal
"""Gain modulation in a LIF neuron. 

Parameters taken from Chance, Abbott and Rayes, Neuron, 2002. and framework
for the model was based on Vogels, T.P. & Abbott, L.F., 2005. The Journal of 
neuroscience.
"""

def gain(time, r_e=135, r_i=135, w_e=4, w_i=16, g_l=10, I_drive=0, f=0, 
        verbose=True, fixed=False):

    # -- params
    # User 
    g_l = g_l * nsiemens
    w_e = w_e * nsiemens  
    w_i = w_i * nsiemens

    w_e = w_e / g_l
    w_i = w_i / g_l
    g_l = g_l / g_l

    r_e = r_e * Hz  
    r_i = r_i * Hz

    # Fixed
    N = 1

    Et = -54 * mvolt
    Er = -65 * mvolt
    Ereset = -60 * mvolt

    Ee = 0 * mvolt
    Ei = -80 * mvolt

    tau_m = 20 * ms
    tau_ampa = 5 * ms  # Setting these equal simplifies balancing (for now)
    tau_gaba = 10 * ms

    # --
    lif = """
    dv/dt = (g_l * (Er - v) + I_syn + I) / tau_m : volt
    I_syn = g_e * (Ee - v) + g_i * (Ei - v) : volt
    dg_e/dt = -g_e / tau_ampa : 1
    dg_i/dt = -g_i / tau_gaba : 1
    I : volt
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
    P_e = NeuronGroup(1, lif, threshold='v > Et', reset='v = Er',
        refractory=2 * ms)
    P_e.v = Ereset
    P_e.I = I_drive * mvolt

    # Set up the 'network' 
    C_be = Synapses(P_be, P_e, pre='g_e += w_e')
    C_be.connect(True)
    C_bi = Synapses(P_bi, P_e, pre='g_i += w_i')
    C_bi.connect(True)

    # Fixed background as well
    if fixed:
        P_fe = PoissonGroup(N, 50 * Hz)
        P_fi = PoissonGroup(N, 50 * Hz)

        C_fe = Synapses(P_fe, P_e, pre='g_e += w_e')
        C_fe.connect(True)
        C_fi = Synapses(P_fi, P_e, pre='g_i += w_i')
        C_fi.connect(True)

    # Data acq
    spikes_e = SpikeMonitor(P_e)
    traces_e = StateMonitor(P_e, ['v', 'g_e', 'g_i'], record=True)

    report = 'text'
    if not verbose:
        report = None

    run(time * second, report=report)

    return {'spikes' : spikes_e, 'traces' : traces_e}


def exp(t, I, xfactor, f=0, r=135, g_l=1.0, fixed=False):
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
    g_l = float(g_l)
    
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

    w_e = g_l * 0.4  # Ratio taken from Reyes 2002
    w_i = g_l * 1.6
    
    return gain(t, r_e=r_e, r_i=r_i, 
                w_e=w_e, w_i=w_i, 
                g_l=g_l, I_drive=I,
                f=f,
                verbose=False, fixed=fixed)


if __name__ == "__main__":
    import pylab as plt
    plt.ion()

    res = gain(1, I_drive=0)

