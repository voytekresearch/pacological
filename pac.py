"""Simulate PAC"""
import numpy as np
from copy import deepcopy
# import pyentropy as en


# Want H for drive and gain (osc) and Net
# Want MI = H_Net - H_drive
# Gain should have two forms - rate and synch
class Spikes(object):
    def __init__(self, n, t, dt=0.001, refractory=0.002, seed=None):

        self.seed = seed
        np.random.seed(self.seed)

        if n < 2:
            raise ValueError("n must be greater than 2")
        self.n = n

        # Timing
        if dt > 0.001:
            raise ValueError("dt must be less than 0.001 seconds (1 ms)")
        else:
            self.dt = dt
        self.t = t
        self.n_steps = int(self.t * (1.0 / self.dt))
        self.times = np.linspace(0, self.t, self.n_steps)

        if refractory % self.dt != 0:
            raise ValueError("refractory must be a integer multiple of dt")
        self.refractory = refractory

        # Create uniform sampling distributions for each neuron
        self.unifs = np.vstack(
            [np.random.uniform(0, 1, self.n_steps) for i in xrange(self.n)]
        ).transpose()

    def _constraints(self, drive, oscillation):
        if drive.shape != oscillation.shape:
            raise ValueError("Shape of `drive` and `oscillation' must match")
        if drive.ndim != 1:
            raise ValueError("`drive` and `oscillation` must be 1d")
        if np.all(drive < 0):
            raise ValueError("`drive` must be greater than 0")
        if np.all(oscillation < 0):
            raise ValueError("`oscillation` must be greater than 0")

    def _refractory(self, spks):
        lw = int(self.refractory / self.dt)  # len of refractory window

        # If it spiked at t, delete spikes
        # in the refractory window
        for t in xrange(spks.shape[0]):
            mask = spks[t, :]
            for t_plus in xrange(lw):
                spks[t_plus, :][mask] = 0

        return spks

    def poisson(self, rates):
        self._constraints(rates, rates)  # does no harm to check twice

        # Method taken from
        # http://www.cns.nyu.edu/~david/handouts/poisson.pdf
        spikes = np.zeros_like(self.unifs, np.int)
        for j in xrange(self.n):
            mask = self.unifs[:,j] <= (rates * self.dt)
            spikes[mask,j] = 1

        return self._refractory(spikes)

    def binary(self, rates, k=3, excitability=0.001):
        self._constraints(rates, rates)  # does no harm to check twice

        ps = rates * excitability
        ns = range(self.n)
        np.random.shuffle(ns)

        spikes = np.zeros_like(self.unifs, np.int)
        for i in xrange(int(self.t * (1.0 / self.dt))):
            for j in xrange(self.n):
                # If bernoilli success, neuron_i_j spikes
                # as does some of it's randomly selected
                # neighbors.
                # TODO - should ensure that j is not equal to jn
                if self.unifs[i, j] < ps[i]:
                    spikes[i, j] = 1

                    np.random.shuffle(ns)
                    for jn in ns[0:k]:
                        spikes[i, jn] = 1

        return self._refractory(spikes)

    def poisson_binary(self, drive, oscillation, k=3, excitability=0.001,
                       amplitude=False):
        self._constraints(drive, oscillation)

        # Renorm
        normed = oscillation / float(oscillation.max())

        # Drive is baseline poisson but...
        spks_p = self.poisson(drive * (1 - normed))

        # the oscillation increases Binomial firing
        if amplitude:
            spks_b = self.binary(
                oscillation + (drive * normed), k, excitability
            )
        else:
            spks_b = self.binary(drive * normed, k, excitability)

        # Justification:
        # 'Partitioning neural variability'
        # 'Gamma oscillations of spiking neural populations
        # enhance signal discrimination.'
        # 'Binary Spiking in Auditory Cortex'

        spks = spks_p + spks_b
        spks[spks > 1] = 1  # Clip double spikes

        return self._refractory(spks)


def osc(times, a, f):
    """Oscillating bias term"""

    return a + (a/2) * np.sin(times * f * 2 * np.pi)


def stim(times, d, scale):
    """Naturalistic bias (via diffusion model)"""

    rates = [d, ]
    for t in times[1:]:
        d += np.random.normal(0, scale)
        rates.append(d)

    rates = np.array(rates)
    rates[rates < 0] = 0

    return rates


def constant(times, d):
    """Constant drive, d"""
    return np.repeat(d, len(times))


def to_spiketimes(times, spikes):
    """Convert spikes to two 1d arrays"""

    n_steps = len(times)
    n = spikes.shape[1]

    ns, ts = [], []
    for i in xrange(n_steps):
        for j in xrange(n):
            if spikes[i,j] == 1:
                ns.append(j)  # save neuron and
                ts.append(times[i])  # look up dt time

    return np.array(ns), np.array(ts)


def to_spikedict(ns, ts):
    """Convert from arrays to a neuron-keyed dict"""
    d_sp = {}
    for n, t in zip(ns, ts):
        try:
            d_sp[n].append(t)
        except KeyError:
            d_sp[n] = [t, ]

    for k in d_sp.keys():
        d_sp[k] = np.array(d_sp[k])

    return d_sp


def fano(spikes):
    """Calculate spike-count Fano"""
    return spikes.sum(0).std() ** 2 / spikes.sum(0).mean()


def isi(d_sp):
    """ISIs, in a neuron-keyed dict"""

    d_isi = {}
    for k, v in d_sp.items():
        tlast = 0

        intervals = []
        for t in v:
            intervals.append(t - tlast)
            tlast = deepcopy(t)

        d_isi[k] = np.array(intervals)

    return d_isi
