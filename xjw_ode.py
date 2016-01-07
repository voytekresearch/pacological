from __future__ import division
import pylab as plt
%matplotlib inline
from foof.util import create_psd

from scipy.integrate import odeint, ode
from sdeint import itoint
from numpy import exp, allclose, asarray, linspace, argmax, zeros, newaxis
from numpy import abs as npabs
from numpy.random import normal
from fakespikes.rates import stim<Paste>


def phi(Isyn, I, c, g):
    return ((c * Isyn) - I) / (1 - exp(-g * ((c * Isyn) - I)))


def create_I(tmax, d, scale, dt=1, seed=None):
    times = linspace(0, tmax, tmax/dt)
    rates = stim(times, d, scale, seed)
    
    def I(t):
        i = (npabs(times - t)).argmin()
        return rates[i]
    
    return I

def g(t, rs):
    return 0.01

# Next: add dW
def f2(t, rs):
    (re1, ri1, s_ee1, s_ie1, s_ei1, s_ii1, 
     re2, ri2, s_ee2, s_ie2, s_ei2, s_ii2, 
     se_oe1, se_oi1, si_oe1, si_oi1, _) = rs
        
    # Pop rate params
    c = 1.1 # lit?
    g = 1 / 10 # lit?
    
    # membrane tau
    tau_n = 2
        
    # 1: EI as S, set at fixed point drive with diffusion
    I_e = 100 
    I_i = 85 
    k = 0.65  #  rescales I_e/i in syn to keep it from osc by itself
    I = Istim(t)
    
    J_ee1 = 2.1 
    J_ie1 = 2.5 
    J_ei1 = 1.3 
    J_ii1 = 3.1
    
    Je_oe = .6
    Je_oi = .6
    Ji_oe = .0
    Ji_oi = .0
    
    tau_ampa1 = 5 # fix
    tau_gaba1 = 20 # fix

    s_ee1 = (-s_ee1 / tau_ampa1) + re1
    s_ie1 = (-s_ie1 / tau_gaba1) + ri1
    s_ei1 = (-s_ei1 / tau_ampa1) + re1
    s_ii1 = (-s_ii1 / tau_gaba1) + ri1
    
    se_oe1 = (-se_oe1 / tau_ampa1) + re2
    se_oi1 = (-se_oi1 / tau_gaba1) + ri2
    si_oe1 = (-si_oe1 / tau_ampa1) + re2
    si_oi1 = (-si_oi1 / tau_gaba1) + ri2
    
    Isyn_e1 = (s_ee1 * J_ee1) - (s_ie1 * J_ie1) + (se_oe1 * Je_oe) - (se_oi1 * Je_oi) + (I * I_e * k)
    Isyn_i1 = (s_ei1 * J_ei1) - (s_ii1 * J_ii1) + (se_oe1 * Ji_oe) - (si_oi1 * Ji_oi) + (I * I_i * k * 0)

    re1 = (-re1 + phi(Isyn_e1, I_e, c, g)) / tau_n  # Fast response
    ri1 = (-ri1 + phi(Isyn_i1, I_i, c, g)) / tau_n
    
    # 2: EI as O
    J_ee2 = 1.5 # lit
    J_ie2 = 1.1 # fix 2
    J_ei2 = 1.1 # fix
    J_ii2 = 1.1 # lit
    tau_ampa1 = 45 # fix
    tau_gaba1 = 10 # fix
    
    s_ee2 = (-s_ee2 / tau_ampa1) + re2
    s_ie2 = (-s_ie2 / tau_gaba1) + ri2
    s_ei2 = (-s_ei2 / tau_ampa1) + re2
    s_ii2 = (-s_ii2 / tau_gaba1) + ri2
    
    Isyn_e2 = (s_ee2 * J_ee2) - (s_ie2 * J_ie2) + I_e * 1.2
    Isyn_i2 = (s_ei2 * J_ei2) - (s_ii2 * J_ii2) + I_i * 1.1
    
    re2 = (-re2 + phi(Isyn_e2, I_e, c, g)) / tau_n  # Fast response
    ri2 = (-ri2 + phi(Isyn_i2, I_i, c, g)) / tau_n
    
    return [re1, ri1, s_ee1, s_ie1, s_ei1, s_ii1, 
            re2, ri2, s_ee2, s_ie2, s_ei2, s_ii2, 
            se_oe1, se_oi1, si_oe1, si_oi1, I]

# run
r0 = [8, 12.0]  # intial rates (re, ri)
tmax = 1000  # run time, ms
dt = 1  # resolution, ms

# Stim params
d = 1  # drive rate (want 0-1)
scale = .01 * d
Istim = create_I(tmax, d, scale, seed=42)

re1, re2 = [], []
ri1, ri2 = [], []
Is = []
t = [0, ]
solver = ode(f2)
solver.set_initial_value(r0 * 8 + [0])  # set gating vars by replicating r0
while solver.successful() and t[-1] < tmax:
    solver.integrate(solver.t + dt)
    rs = solver.y
    
    re1.append(rs[0])
    ri1.append(rs[1])
    
    re2.append(rs[6])
    ri2.append(rs[7])
    Is.append(rs[-1])
    
    t.append(solver.t)
    
# plot
# 1
plt.figure(figsize=(14,10))
plt.subplot(411)
plt.plot(t[1:], [Istim(x) for x in t[1:]], 'k', label='1: Stim')
plt.legend(loc='best')
plt.xlabel("Time (ms)")
plt.ylabel("I")

plt.subplot(412)
plt.plot(t[1:], re1, label='1: E')
plt.plot(t[1:], ri1, label='1: I')
plt.legend(loc='best')
plt.xlabel("Time (ms)")
plt.ylabel("Rate (Hz)")

plt.subplot(413)
plt.plot(re1[500:], ri1[500:], label='phase plane', color='k')
plt.legend(loc='best')
plt.xlabel("r_e (Hz)")
plt.ylabel("r_i (Hz)")

plt.subplot(414)
fs, psd = create_psd(re1[500:] + ri1[500:], 1000)
plt.plot(fs[:100], psd[:100], label='r_e')
plt.legend(loc='best')
plt.xlabel("Freq (Hz)")
plt.ylabel("PSD")

print("last re1: {0}".format(re1[-1]))
print("last ri1: {0}".format(ri1[-1]))

# 1
plt.figure(figsize=(14,10))
plt.subplot(311)
plt.plot(t[1:], re2, label='1: E')
plt.plot(t[1:], ri2, label='1: I')
plt.legend(loc='best')
plt.xlabel("Time (ms)")
plt.ylabel("Rate (Hz)")

plt.subplot(312)
plt.plot(re2[500:], ri2[500:], label='phase plane', color='k')
plt.legend(loc='best')
plt.xlabel("r_e (Hz)")
plt.ylabel("r_i (Hz)")

plt.subplot(313)
fs, psd = create_psd(re2[500:] + ri2[500:], 1000)
plt.plot(fs[:100], psd[:100], label='r_e')
plt.legend(loc='best')
plt.xlabel("Freq (Hz)")
plt.ylabel("PSD")

print("last re2 : {0}".format(re2[-1]))
print("last ri2: {0}".format(ri2[-1]))
print("Peak F (Hz) {0}".format(fs[argmax(psd[:100])]))

