# -*- coding: utf-8 -*-
from __future__ import division

from functools import partial
from copy import deepcopy
from scipy.integrate import odeint, ode
from sdeint import itoint
from numpy import exp, allclose, asarray
from numpy import linspace, argmax, zeros, diag, zeros_like
from numpy import abs as npabs
from numpy import mean as npmean
from numpy.random import normal
from fakespikes.rates import stim
from pacological.util import create_stim_I, ornstein_uhlenbeck

"""Jansen Rit model"""


def S(x):
    """Define the nonlinearity"""
    e0 = 2.5
    v0 = 6.
    r = 0.56

    # A variation from the orginal Jansen 1995 taken from 
    # Wang and Knoshe, ' A Realistic Neural Mass Model of the Cortex with 
    # Laminar-Specific Connections and Synaptic Plasticity – Evaluation with 
    # Auditory Habituation' PLOS one 2013
    # (may be based on an earlier Friston paper; see references.
    return (2 * e0) / (1 + exp(r * (v0 - x))) - (2 * e0) / (1 + exp(r * v0))


def jr(rs, t, Istim=None, A=3.25, B=22.0, c=60, c5=10., c6=10., p=120.):
    y0, y1, y2, y3, y4, y5 = rs[0:6]
    y6, y7, y8, y9, y10, y11 = rs[6:12]

    # Drive/signal corrupted by addtive noise
    dW = rs[12]
    pstim = p + dW 

    # 1 STIM
    # Params
    rs[13] = Istim(t)
    if Istim is not None:
        pstim *= Istim(t)  # global

    c1 = 1. * c
    c2 = 0.8 * c
    c3 = 0.25 * c
    c4 = 0.25 * c

    # David & Fristom, 'A neural mass model for MEG/EEG', Neuroimage, 2003
    e = (1 / 4.6) * 1000  # ms to s 
    i = (1 / 2.9) * 1000 

    ret = zeros_like(rs)
    ret[0] = y3
    ret[1] = y4
    ret[2] = y5
    ret[3] = (A * e * S(y1 - y2 + (c5 * y6))) - (2 * e * y3) - ((e ** 2) * y0)
    ret[4] = (A * e * (pstim + c2 * S(c1 * y0))) - (2 * e * y4) - ((e ** 2) * y1)
    ret[5] = (B * i * c4 * S((c3 * y0) + (c6 * y6))) - (2 * i * y5) - ((i ** 2) * y2)

    # # 2: THETA OSCILLATION
    C = 3.25  # mV
    D = 22.

    c = 135.  # Over rides user input; want stable oscillation
    c1 = 1. * c
    c2 = 0.8 * c
    c3 = 0.25 * c
    c4 = 0.25 * c

    # David & Fristom, 'C neural mass model for MEG/EEG', Neuroimage, 2003
    e2 = (1 / 30.) * 1000  # ms to s 
    i2 = (1 / 20.) * 1000  
    ret[6] = y9
    ret[7] = y10
    ret[8] = y11
    ret[9] = (C * e2 * S(y7 - y8)) - (2 * e2 * y9) - ((e2 ** 2) * y6)
    ret[10] = (C * e2 * (p + c2 * S(c1 * y6))) - (2 * e2 * y10) - ((e2 ** 2) * y7)
    ret[11] = (D * i2 * c4 * S(c3 * y6)) - (2 * i2 * y11) - ((i2 ** 2) * y8)

    return ret


if __name__ == "__main__":
    import pylab as plt
    plt.ion()
    from foof.util import create_psd

    # run
    seed = 42
    rs0 = asarray([0., 0., 0., 1., 1., 1.] + [0., 0., 0., 1., 1., 1.] + [0., 0])
    t = 2  # run time, ms
    dt = 1 / 10000.  # resolution, ms
    p = 130. # Jansen range was 120-320
    sigma = 0.2
    c5 = 10.  # ?
    c6 = c5 * 1

    # Stim params
    d = 1  # drive rate (want 0-1)
    scale = .01 * d
    Istim = create_stim_I(t, d, scale, dt=dt, seed=seed)

    # Integrate
    times = linspace(0, t, t / dt)
    f = partial(jr, Istim=Istim, c5=c5, c6=c6, p=p)
    g = partial(ornstein_uhlenbeck, sigma=sigma, loc=[1, 2, 6, 7, 8, 12])
    rs = itoint(f, g, rs0, times)

    # -------------------------------------
    # # Select some interesting vars and plot
    t = times >= 0.0
    eeg = rs[t,1] - rs[t,2]
    x = rs[t,13]
    dW = rs[t,12]
    osc = rs[t,7] - rs[t,8]

    # Time
    n = 6
    plt.figure(figsize=(10, 14))

    plt.subplot(n, 1, 1)
    plt.legend(loc='best')
    plt.plot(times[t], x, 'k', label='Stimulus input')
    plt.legend(loc='best')
    plt.xlabel("Time (s)")
    plt.ylabel("mV")


    plt.subplot(n, 1, 2)
    plt.plot(times[t], eeg, 'k', label='Stimulus EEG')
    plt.legend(loc='best')
    plt.xlabel("Time (s)")
    plt.ylabel("Hz")

    plt.subplot(n, 1, 3)
    plt.plot(times[t], rs[t,0], 'k', label='Stimulus Pyr')
    plt.legend(loc='best')
    plt.xlabel("Time (s)")
    plt.ylabel("mV")

    plt.subplot(n, 1, 4)
    plt.plot(times[t], osc, 'r', label='Theta EEG')
    plt.legend(loc='best')
    plt.xlabel("Time (s)")
    plt.ylabel("Hz")

    plt.subplot(n, 1, 5)
    plt.plot(times[t], rs[t,6], 'r', label='Theta Pyr')
    plt.legend(loc='best')
    plt.xlabel("Time (s)")
    plt.ylabel("Hz")  

    plt.subplot(n, 1, 6)
    plt.legend(loc='best')
    plt.plot(times[t], dW, 'k', label='Background noise')
    plt.xlabel("Time (s)")
    plt.ylabel("dW (Hz)")

    # Phase
    plt.figure(figsize=(6, 6))
    plt.plot(eeg, osc, label='phase plane (stim, eeg)', color='k')
    plt.legend(loc='best')
    plt.xlabel("Stimulus EEG")
    plt.ylabel("Theta EEG")
