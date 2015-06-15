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
n = 500
t = 3

Iosc = 3
f = 10

Istim = 20
Sstim = .1  # Keep this fairly small

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

# Simulate spiking
# Create a fixed noiseless stimulus pattern, then....
stim_sp = drivespikes.poisson(d_bias['stim'])

# study how different modulation schemes interact
# with it
d_spikes = {}
d_spikes['stim'] = stim_sp
d_spikes['stim_est'] = modspikes.poisson(d_bias['stim'])
d_spikes['osc'] = modspikes.poisson(d_bias['osc'])
d_spikes['gain'] = modspikes.multiply(d_bias['stim'], d_bias['osc'])
d_spikes['gain'] = modspikes.multiply(d_bias['stim'], d_bias['osc'])
d_spikes['summed'] = modspikes.add(d_bias['stim'], d_bias['osc'])
d_spikes['silenced'] = modspikes.subtract(d_bias['stim'], d_bias['osc'])
d_spikes['sync'] = modspikes.poisson_binomial(
    d_bias['stim'], d_bias['osc'], amplitude=False, max_rate=1000)
d_spikes['sync_gain'] = modspikes.poisson_binomial(
    d_bias['stim'], d_bias['osc'], amplitude=True, max_rate=1000)


# -- CREATE LFP --------------------------------------------------------------
d_lfps = {}
d_lfps['stim'] = lfp.create_lfps(d_spikes['stim'])
d_lfps['osc'] = lfp.create_lfps(d_spikes['osc'])
d_lfps['gain'] = lfp.create_lfps(d_spikes['gain'])
d_lfps['summed'] = lfp.create_lfps(d_spikes['summed'])
d_lfps['silenced'] = lfp.create_lfps(d_spikes['silenced'])
d_lfps['sync'] = lfp.create_lfps(d_spikes['sync'])
d_lfps['sync_gain'] = lfp.create_lfps(d_spikes['sync_gain'])

# -- I -----------------------------------------------------------------------
# Spikes
d_infos = {}
m = 20  # Ince's advice was 8. 20 seems better. How to know what to do?
d_infos['stim_est'] = en.DiscreteSystem(
    en.quantise_discrete(stim_sp.sum(1), m),
    (1, len(stim_sp)),
    en.quantise_discrete(d_spikes['stim_est'].sum(1), m),
    (1, len(d_spikes['stim_est']))
)
d_infos['stim_est'].calculate_entropies(method='plugin', calc=('HX', 'HY', 'HXY'))

d_infos['osc'] = en.DiscreteSystem(
    en.quantise_discrete(stim_sp.sum(1), m),
    (1, len(stim_sp)),
    en.quantise_discrete(d_spikes['osc'].sum(1), m),
    (1, len(d_spikes['osc']))
)
d_infos['osc'].calculate_entropies(method='plugin', calc=('HX', 'HY', 'HXY'))

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

d_infos['sync'] = en.DiscreteSystem(
    en.quantise_discrete(stim_sp.sum(1), m),
    (1, len(stim_sp)),
    en.quantise_discrete(d_spikes['sync'].sum(1), m),
    (1, len(d_spikes['sync']))
)
d_infos['sync'].calculate_entropies(method='plugin', calc=('HX', 'HY', 'HXY'))

d_infos['sync_gain'] = en.DiscreteSystem(
    en.quantise_discrete(stim_sp.sum(1), m),
    (1, len(stim_sp)),
    en.quantise_discrete(d_spikes['sync_gain'].sum(1), m),
    (1, len(d_spikes['sync_gain']))
)
d_infos['sync_gain'].calculate_entropies(method='plugin', calc=('HX', 'HY', 'HXY'))

# LFP
# TODO norm the lfps for quantise_discrete
# d_linfos = {}
#
# d_linfos['osc'] = en.DiscreteSystem(
#     en.quantise_discrete(stim_sp, m),
#     (1, len(stim_sp)),
#     en.quantise_discrete(d_lfps['osc'], m),
#     (1, len(d_lfps['osc']))
# )
# d_linfos['osc'].calculate_entropies(method='plugin', calc=('HX', 'HY', 'HXY'))
#
# d_linfos['gain'] = en.DiscreteSystem(
#     en.quantise_discrete(stim_sp, m),
#     (1, len(stim_sp)),
#     en.quantise_discrete(d_lfps['gain'], m),
#     (1, len(d_lfps['gain']))
# )
# d_linfos['gain'].calculate_entropies(method='plugin', calc=('HX', 'HY', 'HXY'))
#
# d_linfos['summed'] = en.DiscreteSystem(
#     en.quantise_discrete(stim_sp, m),
#     (1, len(stim_sp)),
#     en.quantise_discrete(d_lfps['summed'], m),
#     (1, len(d_lfps['summed']))
# )
# d_linfos['summed'].calculate_entropies(method='plugin', calc=('HX', 'HY', 'HXY'))
#
# d_linfos['silenced'] = en.DiscreteSystem(
#     en.quantise_discrete(stim_sp, m),
#     (1, len(stim_sp)),
#     en.quantise_discrete(d_lfps['silenced'], m),
#     (1, len(d_lfps['silenced']))
# )
# d_linfos['silenced'].calculate_entropies(method='plugin', calc=('HX', 'HY', 'HXY'))
#
# d_linfos['sync'] = en.DiscreteSystem(
#     en.quantise_discrete(stim_sp, m),
#     (1, len(stim_sp)),
#     en.quantise_discrete(d_lfps['sync'], m),
#     (1, len(d_lfps['sync']))
# )
# d_linfos['sync'].calculate_entropies(method='plugin', calc=('HX', 'HY', 'HXY'))
#
# d_linfos['sync_gain'] = en.DiscreteSystem(
#     en.quantise_discrete(stim_sp, m),
#     (1, len(stim_sp)),
#     en.quantise_discrete(d_lfps['sync_gain'], m),
#     (1, len(d_lfps['sync_gain']))
# )
# d_linfos['sync_gain'].calculate_entropies(method='plugin', calc=('HX', 'HY', 'HXY'))

# -- PAC OF LFP --------------------------------------------------------------
low_f = (f-2, f+2)
high_f = (80, 250)
# method = 'plv'
method = 'mi'
# method = 'glm'
filt = 'eegfilt'
kwargs = {'trans' : .15} # for eegfilt

d_pacs = {}
d_pacs['osc'] = scpac(d_lfps['osc'], low_f, high_f, rate, method, filt, **kwargs)
d_pacs['gain'] = scpac(d_lfps['gain'], low_f, high_f, rate, method, filt, **kwargs)
d_pacs['summed'] = scpac(d_lfps['summed'], low_f, high_f, rate, method, filt, **kwargs)
d_pacs['silenced'] = scpac(d_lfps['silenced'], low_f, high_f, rate, method, filt, **kwargs)
d_pacs['sync'] = scpac(d_lfps['sync'], low_f, high_f, rate, method, filt, **kwargs)
d_pacs['sync_gain'] = scpac(d_lfps['sync_gain'], low_f, high_f, rate, method, filt, **kwargs)

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

d_autocorrs['sync'] = correlogram(
    pac.to_spiketimes(times, d_spikes['sync'])[1],
    pac.to_spiketimes(times, d_spikes['sync'])[1],
    width=50/1000.0,
    bin=1/1000.0
)

d_autocorrs['sync_gain'] = correlogram(
    pac.to_spiketimes(times, d_spikes['sync_gain'])[1],
    pac.to_spiketimes(times, d_spikes['sync_gain'])[1],
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

