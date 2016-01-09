# -*- coding: utf-8 -*-
from __future__ import division

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
    # No idea why this is here, was taken from the XPP code
    # Leaving it in case I ever figure it out...
    # S(x)=1/(1+exp(-r1*(x-r2)))-1/(1+exp(r1*r­2))

    # Anyway, now define S
    e0 = 2.5
    v0 = 6.
    r = 0.56

    return (2 * e0) / (1 + exp(r * (v0 - x)))


def jt(rs, t, Istim=None):
    x1, x2, x3, x4, x5, x6 = rs

    # 1 STIM
    # Params
    p = 20
    if Istim is not None:
        p *= Istim(t)  # global

    A = 3.25
    B = 22.
    e = 100.
    i = 50.
    c = 135.

    r1 = 2.
    r2 = 1.

    c1 = 1. * c
    c2 = 0.8 * c
    c3 = 0.25 * c
    c4 = 0.25 * c

    ret = zeros_like(rs)
    ret[0] = x4
    ret[1] = x5
    ret[2] = x6
    ret[3] = (A * e * (p + c2 * S(c1 * x3))) - (2 * e * x4) - ((e ** 2) * x1)
    ret[4] = (B * i * c4 * S(c3 * x3)) - (2 * i * x5) - ((i ** 2) * x2)
    ret[5] = (A * e * S(x1 - x2)) - (2 * e * x6) - (e ** 2 * x3)

    # # 2: THETA OSCILLATION
    # TODO - read Friston for params and the layers paper?

    return ret


if __name__ == "__main__":
    import pylab as plt
    plt.ion()
    from functools import partial
    from foof.util import create_psd

    # run
    rs0 = asarray([0., 0., 0., 1., 1., 1.])
    tmax = 10  # run time, ms
    dt = 1/10000.  # resolution, ms

    # Stim params
    d = 1  # drive rate (want 0-1)
    scale = .01 * d
    Istim = create_I(tmax, d, scale, dt=dt, seed=42)
    # Istim = None

    times = linspace(0, tmax, tmax / dt)
    f = partial(jt, Istim=Istim)
    g = partial(ornstein_uhlenbeck, sigma=0.1, loc=[1])
    rs = itoint(f, g, rs0, times)
    # rs = odeint(f, rs0, times)

    # -------------------------------------
    # # Select some interesting vars and plot
    t = times
    eeg = rs[:,1] - rs[:,2]
    x =  [Istim(x) for x in t]

    n = 2
    plt.figure(figsize=(14, 10))
    plt.subplot(n, 1, 1)
    plt.plot(t, eeg, 'r', label='EEG')
    plt.plot(t, x, 'k', label='Stim')
    plt.legend(loc='best')
    plt.xlabel("Time (s)")
    plt.ylabel("Avg Pyr firing rate (Hz)")

    plt.subplot(n, 1, 2)
    plt.plot(x, eeg, label='phase plane (stim, eeg)', color='k')
    plt.legend(loc='best')
    plt.xlabel("r_e (Hz)")
    plt.ylabel("r_i (Hz)")

