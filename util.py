from numpy import exp, allclose, asarray, linspace, argmax, zeros, diag
from numpy import abs as npabs
from numpy import mean as npmean
from numpy.random import normal
from fakespikes.rates import stim

def phi(Isyn, I, c, g):
    return ((c * Isyn) - I) / (1 - exp(-g * ((c * Isyn) - I)))


def create_I(tmax, d, scale, dt=1, seed=None):
    times = linspace(0, tmax, tmax/dt)
    rates = stim(times, d, scale, seed)
    
    def I(t):
        i = (npabs(times - t)).argmin()
        return rates[i]
    
    return I

def ornstein_uhlenbeck(rs, t, sigma=0.5, loc=None):
    if loc is None:
        loc = range(rs.size)

    sigmas = zeros(rs.size)
    sigmas[loc] = sigma  # Locations of re1,ri1,re2,ri2
    
    return diag(sigmas)

