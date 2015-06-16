# %load test_pac.py
"""An intial PAC experiment"""
import numpy as np
import pyentropy as en
import seaborn as sns
import matplotlib.pyplot as plt; plt.ion()

from pacological import pac
from noisy import lfp
from neurosrc.spectral.pac import scpac
from brian import correlogram


# -- USER SETTINGS -----------------------------------------------------------
n = 100
t = 3

Iosc = 4
f = 12

Istim = 10
Sstim = 1

dt = 0.001
rate = 1 / dt

# -- SIM ---------------------------------------------------------------------
# Init spikers
modspikes = pac.Spikes(n, t, dt=dt)
drivespikes = pac.Spikes(n, t, dt=dt)
times = modspikes.times  # brevity

# Create biases
d_bias = {}
d_bias['osc'] = pac.osc(times, Iosc, f)
d_bias['stim'] = pac.stim(times, Istim, Sstim)
d_bias['gain'] = d_bias['osc'] * d_bias['stim']
d_bias['summed'] = d_bias['osc'] + d_bias['stim']
d_bias['silenced'] = d_bias['stim'] - d_bias['osc']

# Simulate spiking
# Create a fixed noiseless stimulus pattern, then....
stim_sp = drivespikes.poisson(d_bias['stim'])

# study how different modulation schemes interact
# with it
d_spikes = {}
d_spikes['stim_est'] = modspikes.poisson(d_bias['stim'])
d_spikes['osc'] = modspikes.poisson(d_bias['osc'])
d_spikes['gain'] = modspikes.poisson(d_bias['gain'])
d_spikes['summed'] = modspikes.poisson(d_bias['summed'])
d_spikes['silenced'] = modspikes.poisson(d_bias['silenced'])

# -- CREATE LFP --------------------------------------------------------------
d_lfps = {}
d_lfps['stim_est'] = lfp.create_lfps(d_spikes['stim_est'])
d_lfps['osc'] = lfp.create_lfps(d_spikes['osc'])
d_lfps['gain'] = lfp.create_lfps(d_spikes['gain'])
d_lfps['summed'] = lfp.create_lfps(d_spikes['summed'])
d_lfps['silenced'] = lfp.create_lfps(d_spikes['silenced'])

# -- I -----------------------------------------------------------------------
# Spikes
d_infos = {}
m = 20  # Per Ince's advice

d_infos['stim_est'] = en.DiscreteSystem(
    en.quantise_discrete(stim_sp.sum(1), m),
    (1, len(stim_sp)),
    en.quantise_discrete(d_spikes['stim_est'].sum(1), m),
    (1, len(d_spikes['stim_est']))
)
d_infos['stim_est'].calculate_entropies(method='plugin', calc=('HX', 'HY', 'HXY'))

d_infos['gain'] = en.DiscreteSystem(
    en.quantise_discrete(stim_sp.sum(1), m),
    (1, len(stim_sp)),
    en.quantise_discrete(d_spikes['gain'].sum(1), m),
    (1, len(d_spikes['gain']))
)
d_infos['gain'].calculate_entropies(method='plugin', calc=('HX', 'HY', 'HXY'))

d_infos['summed'] = en.DiscreteSystem(
    en.quantise_discrete(stim_sp.sum(1), m),
    (1, len(stim_sp)),
    en.quantise_discrete(d_spikes['summed'].sum(1), m),
    (1, len(d_spikes['summed']))
)
d_infos['summed'].calculate_entropies(method='plugin', calc=('HX', 'HY', 'HXY'))

d_infos['silenced'] = en.DiscreteSystem(
    en.quantise_discrete(stim_sp.sum(1), m),
    (1, len(stim_sp)),
    en.quantise_discrete(d_spikes['silenced'].sum(1), m),
    (1, len(d_spikes['silenced']))
)
d_infos['silenced'].calculate_entropies(method='plugin', calc=('HX', 'HY', 'HXY'))

# LFP
# TODO

# -- PAC OF LFP --------------------------------------------------------------
low_f = (f-4, f+4)
high_f = (80, 250)
method = 'plv'
filt = 'eegfilt'
kwargs = {'trans' : .15} # for eegfilt

d_pacs = {}
d_pacs['osc'] = scpac(d_lfps['osc'], low_f, high_f, rate, method, filt, **kwargs)
d_pacs['gain'] = scpac(d_lfps['gain'], low_f, high_f, rate, method, filt, **kwargs)
d_pacs['summed'] = scpac(d_lfps['summed'], low_f, high_f, rate, method, filt, **kwargs)
d_pacs['silenced'] = scpac(d_lfps['silenced'], low_f, high_f, rate, method, filt, **kwargs)

# -- AUTO CORRELATION --------------------------------------------------------
d_autocorrs = {}
d_autocorrs['osc'] = correlogram(
    pac.to_spiketimes(times, d_spikes['osc'])[1],
    pac.to_spiketimes(times, d_spikes['osc'])[1],
    width=50/1000.0,
    bin=1/1000.0
)
d_autocorrs['gain'] = correlogram(
    pac.to_spiketimes(times, d_spikes['gain'])[1],
    pac.to_spiketimes(times, d_spikes['gain'])[1],
    width=50/1000.0,
    bin=1/1000.0
)
d_autocorrs['summed'] = correlogram(
    pac.to_spiketimes(times, d_spikes['summed'])[1],
    pac.to_spiketimes(times, d_spikes['summed'])[1],
    width=50/1000.0,
    bin=1/1000.0
)
d_autocorrs['silenced'] = correlogram(
    pac.to_spiketimes(times, d_spikes['silenced'])[1],
    pac.to_spiketimes(times, d_spikes['silenced'])[1],
    width=50/1000.0,
    bin=1/1000.0
)

# -- PLOTS -------------------------------------------------------------------
# ts1, ns1 = pac.pac.to_spiketimes(times, d_spikes['silenced'])
# plt.subplot(411)
# plt.plot(ns1, ts1, 'o')
#
# ts2, ns2 = pac.pac.to_spiketimes(times, d_spikes['summed'])
# plt.subplot(412)
# plt.plot(ns2, ts2, 'o')
#
# ts2, ns2 = pac.pac.to_spiketimes(times, d_spikes['gain'])
# plt.subplot(413)
# plt.plot(ns2, ts2, 'o')
#
# plt.subplot(414)
# plt.plot(times, osc, label='osc')
# plt.plot(times, stim, label='stim')
# plt.plot(times, gain, label='gain')
# plt.plot(times, summed, label='sum')
# plt.legend(loc='best')
#
# from matplotlib.backends.backend_pdf import PdfPages
# pp = PdfPages('test_pac.pdf',)
# pp.savefig()
# pp.close()
#

