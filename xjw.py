from __future__ import division

from scipy.integrate import odeint, ode
from sdeint import itoint
from numpy import exp, allclose, asarray, linspace, argmax, zeros, diag
from numpy import abs as npabs
from numpy import mean as npmean
from numpy.random import normal
from fakespikes.rates import stim
from pacological.util import phi, create_stim_I, ornstein_uhlenbeck


def xjw(rs, t, Istim=None, Je_e=0, Je_i=0, Ji_e=0, Ji_i=0, k1=0.8, k2=1.2):
    # Unpack 'y'
    (re1, ri1, s_ee1, s_ie1, s_ei1, s_ii1,
     re2, ri2, s_ee2, s_ie2, s_ei2, s_ii2,
     se_e1, se_i1, si_e1, si_i1) = rs

    # Pop rate tune params
    c = 1.1  # lit?
    g = 1 / 10  # lit?

    # membrane tau
    tau_n = 2 / 1000.  # 2 ms

    # 1: EI as STIMULUS, set at fixed point driven with diffusion
    I_e = 120
    I_i = 85
    if Istim is None:
        I = 1.0
    else:
        I = Istim(t)

    J_ee1 = 1.5
    J_ie1 = 1.5
    J_ei1 = 1.5
    J_ii1 = 1.

    tau_ampa1 = 5 / 1000.  # 5 ms  
    tau_gaba1 = 20 / 1000.  # 20 ms

    # Internal
    s_ee1 = (-s_ee1 / tau_ampa1) + re1
    s_ie1 = (-s_ie1 / tau_gaba1) + ri1
    s_ei1 = (-s_ei1 / tau_ampa1) + re1
    s_ii1 = (-s_ii1 / tau_gaba1) + ri1

    # External (oscillation driven)
    se_e1 = (-se_e1 / tau_ampa1) + re2
    se_i1 = (-se_i1 / tau_ampa1) + re2
    si_e1 = (-si_e1 / tau_gaba1) + ri2
    si_i1 = (-si_i1 / tau_gaba1) + ri2

    Isyn_e1 = (s_ee1 * J_ee1) - (s_ie1 * J_ie1) + \
        (se_e1 * Je_e) - (si_e1 * Ji_e) + (I * I_e * k1)  # I into E only
    Isyn_i1 = (s_ei1 * J_ei1) - (s_ii1 * J_ii1) + \
        (se_i1 * Je_i) - (si_i1 * Ji_i) + (I_i)

    re1 = (-re1 + phi(Isyn_e1, I_e, c, g)) / tau_n  # Fast response
    ri1 = (-ri1 + phi(Isyn_i1, I_i, c, g)) / tau_n

    # 2: THETA OSCILLATION
    J_ee2 = 2.1  # lit
    J_ie2 = 1.9  # lit
    J_ei2 = 1.5  # fix
    J_ii2 = 1.1  # lit
    tau_ampa1 = 40 / 1000.  # 40 ms
    tau_gaba1 = 20 / 1000.  # 20 ms

    s_ee2 = (-s_ee2 / tau_ampa1) + re2
    s_ie2 = (-s_ie2 / tau_gaba1) + ri2
    s_ei2 = (-s_ei2 / tau_ampa1) + re2
    s_ii2 = (-s_ii2 / tau_gaba1) + ri2

    Isyn_e2 = (s_ee2 * J_ee2) - (s_ie2 * J_ie2) + I_e * k2
    Isyn_i2 = (s_ei2 * J_ei2) - (s_ii2 * J_ii2) + I_i * k2

    re2 = (-re2 + phi(Isyn_e2, I_e, c, g)) / tau_n  # Fast response
    ri2 = (-ri2 + phi(Isyn_i2, I_i, c, g)) / tau_n

    return asarray([re1, ri1, s_ee1, s_ie1, s_ei1, s_ii1,
                    re2, ri2, s_ee2, s_ie2, s_ei2, s_ii2,
                    se_e1, se_i1, si_e1, si_i1])

# demo
if __name__ == "__main__":
    import pylab as plt
    plt.ion()
    from functools import partial
    from foof.util import create_psd

    # run
    r0 = [8, 12.0]  # intial rates (re, ri)
    tmax = 2  # run time, ms
    dt = 1. / 10000 # resolution, ms

    # Stim params
    d = 1  # drive rate (want 0-1)
    scale = .01 * d
    Istim = create_stim_I(tmax, d, scale, dt=dt, seed=1)

    # Simulate
    times = linspace(0, tmax, tmax / dt)
    rs0 = asarray(r0 * 8)

    f = partial(xjw, Istim=Istim, Je_e=2.0, Je_i=2.0, Ji_e=0.0, Ji_i=0.0, 
            k1=0.5, k2=1.2)
    g = partial(ornstein_uhlenbeck, sigma=20, loc=[0, 1, 6, 7])  # re/i locs
    rs = itoint(f, g, rs0, times)

    # -------------------------------------
    # Select some interesting vars and plot
    t = times
    re1 = rs[:, 0]
    ri1 = rs[:, 1]
    re2 = rs[:, 6]
    ri2 = rs[:, 7]

    # 1
    plt.figure(figsize=(14, 10))
    plt.subplot(411)
    plt.plot(t, [Istim(x) for x in t], 'k', label='1: Stim')
    plt.legend(loc='best')
    plt.xlabel("Time (ms)")
    plt.ylabel("I")

    plt.subplot(412)
    plt.plot(t, re1, label='1: E')
    plt.plot(t, ri1, label='1: I')
    plt.ylim(0, 100)
    plt.legend(loc='best')
    plt.xlabel("Time (ms)")
    plt.ylabel("Rate (Hz)")

    plt.subplot(413)
    plt.plot(re1[500:], ri1[500:], label='phase plane', color='k')
    plt.legend(loc='best')
    plt.xlabel("r_e (Hz)")
    plt.ylabel("r_i (Hz)")

    plt.subplot(414)
    fs, psd = create_psd(re1[500:] + ri1[500:], 10000)
    plt.plot(fs[:100], psd[:100], label='r_e')
    plt.legend(loc='best')
    plt.xlabel("Freq (Hz)")
    plt.ylabel("PSD")

    print("Mean re1 : {0}".format(npmean(re1)))
    print("Mean ri1: {0}".format(npmean(ri1)))
    print("Max STIM F (Hz) {0}".format(fs[argmax(psd[:100])]))

    # 1
    plt.figure(figsize=(14, 10))
    plt.subplot(311)
    plt.plot(t, re2, label='1: E')
    plt.plot(t, ri2, label='1: I')
    plt.ylim(0, 100)
    plt.legend(loc='best')
    plt.xlabel("Time (ms)")
    plt.ylabel("Rate (Hz)")

    plt.subplot(312)
    plt.plot(re2[500:], ri2[500:], label='phase plane', color='k')
    plt.legend(loc='best')
    plt.xlabel("r_e (Hz)")
    plt.ylabel("r_i (Hz)")

    plt.subplot(313)
    fs, psd = create_psd(re2[500:] + ri2[500:], 10000)
    plt.plot(fs[:100], psd[:100], label='r_e')
    plt.legend(loc='best')
    plt.xlabel("Freq (Hz)")
    plt.ylabel("PSD")

    print("Mean re2 : {0}".format(npmean(re2)))
    print("Mean ri2: {0}".format(npmean(ri2)))
    print("Max OSC F (Hz) {0}".format(fs[argmax(psd[:100])]))
