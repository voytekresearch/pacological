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

def gain(t, r_e=135, r_i=135, w_e=4, w_i=16, g_l=10, I_drive=0):
    # User params
    g_l = g_l * nsiemens
    w_e = w_e * nsiemens  
    w_i = w_i * nsiemens

    w_e = w_e / g_l
    w_i = w_i / g_l
    g_l = g_l / g_l

    r_e = r_e * Hz  # from 2002
    r_i = r_i * Hz

    I_drive = I_drive * mvolt

    # Fixed
    N = 10000
    N_e = int(N * 0.8)
    N_i = N - N_e

    p = 0.02
    p_ii = 0.02 # ?

    Et = -54 * mvolt
    Er = -65 * mvolt
    Ereset = -60 * mvolt
    Ee = 0 * mvolt
    Ei = -80 * mvolt

    tau_m = 20 * ms
    tau_ampa = 5 * ms
    tau_gaba = 10 * ms

    lif = """
    dv/dt = (g_l * (Er - v) + I_syn + I) / tau_m : volt
    I_syn = g_e * (Ee - v) + g_i * (Ei - v) : volt
    dg_e/dt = -g_e / tau_ampa : 1
    dg_i/dt = -g_i / tau_gaba : 1
    I : volt
    """

    # The background noise
    P_be = PoissonGroup(N_e, r_e)
    P_bi = PoissonGroup(N_i, r_i)

    # Our one neuron to gain control
    P_e = NeuronGroup(
        1, lif, 
        threshold='v > Et', reset='v = Ereset',
        refractory=5 * ms
    )
    P_e.v = Ereset

    # Set up the 'network' 
    C_be = Synapses(P_be, P_e, pre='g_e += w_e')
    C_be.connect(True, p=p)
    C_bi = Synapses(P_bi, P_e, pre='g_i += w_i')
    C_bi.connect(True, p=p)

    # Data acq
    spikes_e = SpikeMonitor(P_e)
    traces_e = StateMonitor(P_e, ['v', 'g_e', 'g_i'], record=True)

    run(t * second, report='text')

    return {'spikes' : spikes_e, 'traces' : traces_e}

if __name__ == "__main__":
    import pylab as plt
    plt.ion()

    res = gain(1, I_drive=0)

