# -*- coding: utf-8 -*-
from __future__ import division

from scipy.integrate import odeint, ode
from sdeint import itoint
from numpy import exp, allclose, asarray, linspace, argmax, zeros, diag
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
    v0 = 6
    r = 0.56

    return 2 * e0 / (1 + exp(r * (v0 - x)))


def jt(rs, t):
    _, x1, x2, x3, x4, x5, x6 = rs

    # Params
    p = 20
    A = 3.25
    B = 22
    e = 100
    i = 50
    c = 135

    r1 = 2
    r2 = 1
    c1 = 1 * c
    c2 = 0.8 * c
    c3 = 0.25 * c
    c4 = 0.25 * c

    # define the right-hand sides; delaying x by tau
    x1 = x4
    x2 = x5
    x3 = x6
    x4 = A * e * (p + c2 * S(c1 * x3)) - 2 * e * x4 - e ** 2 * x1
    x5 = B * i * c4 * S(c3 * x3) - 2 * i * x5 - i ** 2 * x2
    x6 = A * e * S(x1 - x2) - 2 * e * x6 - e ** 2 * x3
    #####################################
    x = x2 - x3

    return asarray([x, x1, x2, x3, x4, x5, x6])

####################################
# and the initial condition
# x1(0) = 0
# x2(0) = 0
# x3(0) = 0
# x4(0) = 1
# x5(0) = 1
# x6(0) = 1
####################################


if __name__ == "__main__":
    import pylab as plt
    plt.ion()
    from functools import partial
    from foof.util import create_psd

    # run
    rs0 = asarray([0., 0., 0., 0., 1., 1., 1.])
    tmax = 1000  # run time, ms
    dt = .1  # resolution, ms

    times = linspace(0, tmax, tmax/dt)
    g = partial(ornstein_uhlenbeck, sigma=0.5, loc=range(1, rs0.size))
    rs = itoint(jt, g, rs0, times)

    # -------------------------------------
    # Select some interesting vars and plot
    t = times
    eeg = rs[:, 1]
    
    plt.figure(figsize=(14, 10))
    plt.plot(t, eeg, 'k', label='EEG')
    plt.legend(loc='best')
    plt.xlabel("Time (ms)")
    plt.ylabel("Mass")
