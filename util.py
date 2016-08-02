from numpy import exp, allclose, asarray, linspace, argmax, zeros, diag
from numpy import abs as npabs
from numpy import mean as npmean
from numpy.random import normal
from fakespikes.rates import stim, constant
import sys


def phi(Isyn, I, c, g):
    return ((c * Isyn) - I) / (1 - exp(-g * ((c * Isyn) - I)))


def create_stim_I(tmax, d, scale, dt=1, seed=None):
    times = linspace(0, tmax, tmax/dt)
    rates = stim(times, d, scale, seed)
    
    def I(t):
        i = (npabs(times - t)).argmin()
        return rates[i]
    
    return I


def create_constant_I(tmax, d, dt=1, seed=None):
    times = linspace(0, tmax, tmax/dt)
    rates = constant(times, d)
    
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


def progressbar(it, prefix = "", size=60):
    count = len(it)
    def _show(_i):
        x = int(size*_i / count)
        sys.stdout.write(
                "%s[%s%s] %i/%i\r" % (
                    prefix, "#" * x, "." * (size - x), _i, count))
        sys.stdout.flush()
    
    _show(0)
    for i, item in enumerate(it):
        yield item
        _show(i + 1)
    sys.stdout.write("\n")
    sys.stdout.flush()

