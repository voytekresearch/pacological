"""Simulate PAC"""
import numpy as np
# import pyentropy as en


# Want H for drive and gain (osc) and Net
# Want MI = H_Net - H_drive
# Gain should have two forms - rate and synch
class Spikes(object):
    def __init__(self, n, t, dt=0.001, seed=1):

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

        # Create uniform sampling distributions for each neuron
        self.unifs = np.vstack(
            [np.random.uniform(0, 1, self.n_steps) for i in xrange(self.n)]
        ).transpose()

    def poisson(self, rates):
        # Method taken from
        # http://www.cns.nyu.edu/~david/handouts/poisson.pdf
        spikes = np.zeros_like(self.unifs, np.int)
        for j in xrange(self.n):
            mask = self.unifs[:,j] <= (rates * self.dt)
            spikes[mask,j] = 1

        return spikes

    def binmoimal_poisson(self):
        # Manipulate spikes so that gains
        # increases synchrony
        # Use binonial dist as second process?
        # It is quite a bit more syncrhonous.
        # But how to conserve rate?
        # Justification:
        # 'Partitioning neural variability'
        # 'Gamma oscillations of spiking neural populations enhance signal discrimination.'
        raise NotImplementedError("TODO")


def osc(times, a, f):
        """Oscillating bias term"""

        return a * np.sin(times * f * 2 * np.pi)


def stim(times, d, scale):
    """Naturalistic bias (via diffusion model)"""

    rates = [d, ]
    for t in times[1:]:
        d += np.random.normal(0, scale)
        rates.append(d)

    return np.array(rates)


def gain_pac(times, a, f, d, scale):
    """Stim times osc"""
    return osc(times, a, f) * stim(times, d, scale)


def to_spiketimes(times, spikes):
    """Convert spikes to 2d (dt, neuron) matrix"""

    n_steps = len(times)
    n = spikes.shape[1]

    ns, ts = [], []
    for i in xrange(n_steps):
        for j in xrange(n):
            if spikes[i,j] == 1:
                ns.append(j)  # save neuron and
                ts.append(times[i])  # look up dt time

    return ns, ts
