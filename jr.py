# -*- coding: utf-8 -*-
from __future__ import division

from copy import deepcopy
from scipy.integrate import odeint, ode
from sdeint import itoint
from numpy import exp, allclose, asarray
from numpy import linspace, argmax, zeros, diag, zeros_like
from numpy import abs as npabs
from numpy import mean as npmean
from numpy.random import normal
from fakespikes.rates import stim
from pacological.util import create_I, ornstein_uhlenbeck

"""Jansen Rit model"""


def S(x):
    """Define the nonlinearity"""
    e0 = 2.5
    v0 = 6.
    r = 0.56

    return (2 * e0) / (1 + exp(r * (v0 - x)))


def jt(rs, t, Istim=None, c5=10):
    y0, y1, y2, y3, y4, y5 = rs[0:6]
    y6, y7, y8, y9, y10, y11 = rs[6:]

    # --
    # Common params
    A = 3.25
    B = 22.

    # 1 STIM
    # Params
    p = 150
    if Istim is not None:
        p *= Istim(t)  # global

    c = 250.
    c1 = 1. * c
    c2 = 0.8 * c
    c3 = 0.25 * c
    c4 = 0.25 * c

    # David & Fristom, 'A neural mass model for MEG/EEG', Neuroimage, 2003
    e = 217.40  # 1/tau = 1/4.6 ms
    i = 344.82  # 1/tau = 1/2.9 ms

    ret = zeros_like(rs)
    ret[0] = y3
    ret[1] = y4
    ret[2] = y5
    ret[3] = (A * e * (p + c2 * S(c1 * y2))) - (2 * e * y3) - ((e ** 2) * y0)
    ret[4] = (B * i * c4 * S(c3 * y2)) - (2 * i * y4) - ((i ** 2) * y1)
    ret[5] = (A * e * S(y0 - y1 + (c5 * y9))) - (2 * e * y5) - (e ** 2 * y2)

    # # 2: THETA OSCILLATION
    p2 = 10
    c = 135.
    c1 = 1. * c
    c2 = 0.8 * c
    c3 = 0.25 * c
    c4 = 0.25 * c

    # David & Fristom, 'A neural mass model for MEG/EEG', Neuroimage, 2003
    e2 = 33.  # 1/tau = 1/30 ms
    i2 = 50.  # 1/tau = 1/20 ms
    ret[6] = y9
    ret[7] = y10
    ret[8] = y11
    ret[9] = (A * e2 * (p2 + c2 * S(c1 * y8))) - (2 * e2 * y9) - ((e2 ** 2) * y6)
    ret[10] = (B * i2 * c4 * S(c3 * y8)) - (2 * i2 * y10) - ((i2 ** 2) * y7)
    ret[11] = (A * e2 * S(y6 - y7)) - (2 * e2 * y11) - (e2 ** 2 * y8)

    return ret


if __name__ == "__main__":
    import pylab as plt
    plt.ion()
    from functools import partial
    from foof.util import create_psd

    # run
    rs0 = asarray([0., 0., 0., 1., 1., 1.] + [0., 0., 0., 1., 1., 1.])
    tmax = 2  # run time, ms
    dt = 1 / 10000.  # resolution, ms

    # Stim params
    d = 1  # drive rate (want 0-1)
    scale = .01 * d
    Istim = create_I(tmax, d, scale, dt=dt, seed=42)
    # Istim = None

    times = linspace(0, tmax, tmax / dt)
    f = partial(jt, Istim=Istim, c5=20)
    g = partial(ornstein_uhlenbeck, sigma=0.1, loc=[1])
    rs = itoint(f, g, rs0, times)
    # rs = odeint(f, rs0, times)

    # -------------------------------------
    # # Select some interesting vars and plot
    t = times >= 1.0
    eeg = rs[t,1] - rs[t,2]
    x =  [Istim(x) for x in times[t]]
    osc = rs[t,7] - rs[t,8]

    n = 4
    plt.figure(figsize=(14, 10))

    plt.subplot(n, 1, 1)
    plt.legend(loc='best')
    plt.plot(times[t], x, 'k', label='Stim')
    plt.xlabel("Time (s)")
    plt.ylabel("Firing rate (Hz)")

    plt.subplot(n, 1, 2)
    plt.plot(times[t], eeg, 'r', label='Stimulus EEG')
    plt.legend(loc='best')
    plt.xlabel("Time (s)")
    plt.ylabel("Firing rate (Hz)")

    plt.subplot(n, 1, 3)
    plt.plot(x, eeg, label='phase plane (stim, eeg)', color='k')
    plt.legend(loc='best')
    plt.xlabel("r_e (Hz)")
    plt.ylabel("r_i (Hz)")

    plt.subplot(n, 1, 4)
    plt.plot(times[t], osc, 'r', label='Theta EEG')
    plt.legend(loc='best')
    plt.xlabel("Time (s)")
    plt.ylabel("Firing rate (Hz)")
