from __future__ import division
from brian2 import *
prefs.codegen.target = 'numpy'

from numpy.random import uniform, random_integers, lognormal
from scipy.interpolate import interp1d

import pacological as pac
from pacological.util import progressbar

from fakespikes.rates import bursts
from fakespikes.neurons import Spikes
import fakespikes.util as sp

import os
from tempfile import mkdtemp
from joblib import Memory

# Set a fixed cache so it survives across imports
CACHED = os.path.join(os.path.dirname(pac.__file__), 'cache')
try:
    os.makedirs(CACHED)
except OSError:
    if not os.path.isdir(CACHED):
        raise
memory = Memory(cachedir=CACHED, verbose=0)


def create_phi_zandt(Is, rates, delI):
    """Create the network non-linearity function, phi"""

    _interp = interp1d(Is, rates, kind='linear')

    def phi(I, u, sigma):
        """phi, the network non-linearity."""

        # Integrate fi(I) * N(I, u, sigma)
        # along the window +/- delI
        fi = np.asarray([float(_interp(I - delI)), float(_interp(I + delI))])

        di = np.asarray([
            N(I - delI, u, sigma),
            N(I + delI, u, sigma),
        ]) * Is.max()

        r = np.trapz(fi * di)

        return r

    return phi


def N(I, u, sigma):
    """Network current distribution."""

    a = (1 / (np.sqrt(sigma * 2 * np.pi)))

    return a * np.exp(-0.5 * ((I - u) / sigma)**2)


@memory.cache
def lif(time,
        Is,
        f,
        r_e=135,
        r_i=135,
        w_e=4e-9,
        w_i=16e-9,
        tau_e=5e-3,
        tau_i=10e-3,
        g_l=10e-9,
        n_bursts=None,
        verbose=True,
        min_rate=30,
        back_seed=42,
        return_trains=False):

    if verbose:
        print(">>> Creating fi.")

    # -- params
    # User
    N = len(Is)

    g_l = g_l * siemens
    w_e = w_e * siemens
    w_i = w_i * siemens

    w_e = w_e / g_l
    w_i = w_i / g_l
    g_l = g_l / g_l

    r_e = r_e * Hz
    r_i = r_i * Hz

    # Fixed
    Et = -54 * mvolt
    Er = -65 * mvolt
    Ereset = -60 * mvolt

    Ee = 0 * mvolt
    Ei = -80 * mvolt

    tau_m = 20 * ms
    tau_ampa = tau_e * second
    tau_gaba = tau_i * second

    # --
    lif = """
    dv/dt = (g_l * (Er - v) + I_syn + I) / tau_m : volt
    I_syn = g_e * (Ee - v) + g_i * (Ei - v) : volt
    dg_e/dt = -g_e / tau_ampa : 1
    dg_i/dt = -g_i / tau_gaba : 1
    I : volt
    """

    # The background noise
    if f > 0:
        nrns_e = Spikes(N, time, dt=1e-3, seed=back_seed)
        nrns_i = Spikes(N, time, dt=1e-3, seed=back_seed + 1)

        b_times = nrns_e.times

        burst_e = bursts(b_times, float(r_e), f, n_bursts, min_a=min_rate)
        burst_i = bursts(b_times, float(r_i), f, n_bursts, min_a=min_rate)

        spks_e = nrns_e.poisson(burst_e)
        spks_i = nrns_i.poisson(burst_i)

        ns_e, ts_e = sp.to_spiketimes(b_times, spks_e)
        ns_i, ts_i = sp.to_spiketimes(b_times, spks_i)

        P_be = SpikeGeneratorGroup(N, ns_e, ts_e * second)
        P_bi = SpikeGeneratorGroup(N, ns_i, ts_i * second)
    else:
        P_be = PoissonGroup(N, r_e)
        P_bi = PoissonGroup(N, r_i)

    # Our one neuron to gain control
    P_e = NeuronGroup(N,
                      lif,
                      threshold='v > Et',
                      reset='v = Er',
                      refractory=2 * ms,
                      method='rk2')

    P_e.v = Ereset
    P_e.I = Is * volt

    # Set up the 'network'
    C_be = Synapses(P_be, P_e, on_pre='g_e += w_e')
    C_be.connect('i == j')
    
    C_bi = Synapses(P_bi, P_e, on_pre='g_i += w_i')
    C_bi.connect('i == j')

    # Data acq
    spikes_e = SpikeMonitor(P_e)
    traces_e = StateMonitor(P_e, ['v', 'g_e', 'g_i'], record=True)

    report = 'text'
    if not verbose:
        report = None

    run(time * second, report=report)

    # Calc rates and gs
    trains = spikes_e.spike_trains()
    rates = np.zeros_like(Is)
    for k, v in trains.items():
        rates[k] = v.size / time

    g_es = traces_e.g_e_
    g_is = traces_e.g_i_
    vs = traces_e.v

    if return_trains:
        return rates, trains, g_es, g_is, vs
    else:
        return rates


@memory.cache
def hh(time,
       Is,
       f,
       r_e=675,
       r_i=675,
       w_e=40.0e-3,
       w_i=160.0e-3,
       w_m=0,
       tau_ampa=5e-3,
       tau_gaba=10e-3,
       verbose=True,
       fixed=12):

    if verbose:
        print(">>> Creating fi.")

    time_step = 0.01 * ms
    defaultclock.dt = time_step

    # -- params
    # User
    N = len(Is)

    w_e *= siemens
    w_i *= siemens

    r_e = r_e * Hz
    r_i = r_i * Hz
    fixed = fixed * Hz

    Et = 20 * mvolt

    # Synapse
    Ee = 0 * mvolt
    Ei = -80 * mvolt

    tau_m = 20 * ms
    tau_ampa *= second  # Setting these equal simphhies balancing (for now)
    tau_gaba *= second

    # HH specific
    Cm = 1 * uF  # /cm2

    w_m = w_m * msiemens
    g_Na = 100 * msiemens
    g_K = 80 * msiemens
    g_l = 0.1 * msiemens

    V_Na = 50 * mV
    V_K = -100 * mV
    V_l = -67 * mV  # 67 mV

    V_i = -80 * mV
    V_e = 0 * mV

    hh = """
    dV/dt = (I_Na + I_K + I_l + I_m + I_syn + I) / Cm : volt
    """ + """
    I_Na = g_Na * (m ** 3) * h * (V_Na - V) : amp
    m = a_m / (a_m + b_m) : 1
    a_m = (0.32 * (54 + V/mV)) / (1 - exp(-0.25 * (V/mV + 54))) / ms : Hz
    b_m = (0.28 * (27 + V/mV)) / (exp(0.2 * (V/mV + 27)) - 1) / ms : Hz
    h = clip(1 - 1.25 * n, 0, inf) : 1
    """ + """
    I_K = g_K * n ** 4 * (V_K - V) : amp
    dn/dt = (a_n - (a_n * n)) - b_n * n : 1
    a_n = (0.032 * (52 + V/mV)) / (1 - exp(-0.2 * (V/mV + 52))) / ms : Hz
    b_n = 0.5 * exp(-0.025 * (57 + V/mV)) / ms : Hz
    """ + """
    I_l = g_l * (V_l - V) : amp
    """ + """
    I_m = w_m * w * (V_K - V) : amp
    dw/dt = (w_inf - w) / tau_w/ms : 1
    w_inf = 1 / (1 + exp(-1 * (V/mV + 35) / 10)) : 1
    tau_w = 400 / ((3.3 * exp((V/mV + 35)/20)) + (exp(-1 * (V/mV + 35) / 20))) : 1
    """ + """
    I_syn = g_e * (V_e - V) + g_i * (V_i - V) : amp
    dg_e/dt = -g_e / tau_ampa : siemens
    dg_i/dt = -g_i / tau_gaba : siemens
    I : amp
    """

    # The background noise
    if f > 0:
        f = f * Hz
        P_be = NeuronGroup(N,
                           'rates = fixed + (r_e * cos(2 * pi * f * t)) : Hz',
                           threshold='rand()<rates*dt',
                           method='rk2')

        P_bi = NeuronGroup(N,
                           'rates = fixed + (r_i * cos(2 * pi * f * t)) : Hz',
                           threshold='rand()<rates*dt',
                           method='rk2')
    else:
        P_be = PoissonGroup(N, r_e + fixed)
        P_bi = PoissonGroup(N, r_i + fixed)

    # Our one neuron to gain control
    P_e = NeuronGroup(N,
                      hh,
                      threshold='V > Et',
                      refractory=2 * ms,
                      method='rk2')
    P_e.V = V_l
    P_e.I = Is * uamp

    # Set up the 'network'
    C_be = Synapses(P_be, P_e, on_pre='g_e += w_e')
    C_be.connect('i == j')
    C_bi = Synapses(P_bi, P_e, on_pre='g_i += w_i', delay=2 * ms)
    C_bi.connect('i == j')

    # Data acq
    spikes_e = SpikeMonitor(P_e)
    traces_e = StateMonitor(P_e, ['V', 'g_e', 'g_i'], record=True)

    report = 'text'
    if not verbose:
        report = None

    run(time * second, report=report)

    # Calc rates and gs
    trains = spikes_e.spike_trains()
    rates = np.zeros_like(Is)
    for k, v in trains.items():
        rates[k] = v.size / time

    g_es = traces_e.g_e_
    g_is = traces_e.g_i_

    return rates


if __name__ == "__main__":
    # Quick test
    time = 2
    Is = np.linspace(0, 30, 50)

    f = 10
    trains, rates, ges, gis, times = hh(time, Is, f, verbose=True)
