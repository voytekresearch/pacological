
#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
from brian2 import *
from numpy.random import uniform, random_integers, lognormal
"""Gain modulation in a FitzHugh-Nagumo neuron. 

Parameters for balance taken from Chance, Abbott and Rayes, Neuron, 2002."""

def gain(time, r_e=135, r_i=135, w_e=.1, w_i=.4, I_drive=0, f=0, 
        verbose=True):

    time_step = 0.01 * ms
    defaultclock.dt = time_step

    # -- params
    # User 
    r_e = r_e * Hz  
    r_i = r_i * Hz

    # Fixed
    N = 1

    Et = 1
    a = 0.7 # values taken from p174 of Biophysics of Computation
    b = 0.8
    theta = 0.08
    
    # --
    tau_ampa = 5 * ms  # Setting these equal simplifies balancing (for now)
    tau_gaba = 10 * ms

    fh = """
    dv/dt = ((v - ((v ** 3) / 3) - y) + I) / msecond : 1 
    dy/dt = (theta * (v - b * y + a)) / msecond : 1
    I = I_d + I_syn : 1
    I_syn = (g_e - g_i) : 1
    dg_e/dt = -g_e / tau_ampa : 1
    dg_i/dt = -g_i / tau_gaba : 1

    # I_syn = g_e - g_i : 1
    # g_e : 1
    # g_i : 1
    I_d : 1
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
    P_e = NeuronGroup(1, fh, threshold='v > Et', refractory=20 * ms)
    P_e.I_d = I_drive 

    # Set up the 'network' 
    C_be = Synapses(P_be, P_e, pre='g_e += w_e')
    C_be.connect(True)
    C_bi = Synapses(P_bi, P_e, pre='g_i += w_i')
    C_bi.connect(True)

    # Data acq
    spikes_e = SpikeMonitor(P_e)
    traces_e = StateMonitor(P_e, ['v', 'y', 'g_e', 'g_i'], record=True)

    report = 'text'
    if not verbose:
        report = None

    run(time * second, report=report)

    return {'spikes' : spikes_e, 'traces' : traces_e}


def exp(t, I, xfactor, f=0, r=135, w_e=0.1, w_i=0.4):
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
    """
    t = float(t)
    I = float(I)
    
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

    return gain(t, r_e=r_e, r_i=r_i, w_e=w_e, w_i=w_i,
            I_drive=I, f=f, verbose=False)


if __name__ == "__main__":
    import pylab as plt
    plt.ion()

    res = gain(1, I_drive=0)

