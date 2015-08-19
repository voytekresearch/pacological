# %load test_pac.py
"""An intial PAC experiment"""
# import numpy as np
import pyentropy as en
import seaborn as sns; sns.__file__ # pylint
import matplotlib.pyplot as plt; plt.ion()

from pacological import pac
from noisy import lfp
from neurosrc.spectral.pac import scpac
# from brian import correlogram


# -- USER SETTINGS -----------------------------------------------------------
n = 500
t = 3

Iosc = 4
f = 12

Istim = 10
Sstim = 1

dt = 0.001
rate = 1 / dt

k = 3
excitability = 0.0001

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
d_spikes['stim_est_p'] = modspikes.poisson(d_bias['stim'])
d_spikes['osc_p'] = modspikes.poisson(d_bias['osc'])
d_spikes['gain_p'] = modspikes.poisson(d_bias['gain'])
d_spikes['summed_p'] = modspikes.poisson(d_bias['summed'])
d_spikes['silenced_p'] = modspikes.poisson(d_bias['silenced'])

d_spikes['stim_est_b'] = modspikes.binary(d_bias['stim'], k=k, excitability=excitability)
d_spikes['osc_b'] = modspikes.binary(d_bias['osc'], k=k, excitability=excitability)
d_spikes['gain_b'] = modspikes.binary(d_bias['gain'], k=k, excitability=excitability)
d_spikes['summed_b'] = modspikes.binary(d_bias['summed'], k=k, excitability=excitability)
d_spikes['silenced_b'] = modspikes.binary(d_bias['silenced'], k=k, excitability=excitability)

d_spikes['gain_bp'] = modspikes.poisson_binary(d_bias['stim'], d_bias['osc'], k=k, excitability=excitability)


# -- CREATE LFP --------------------------------------------------------------
d_lfps = {}
d_lfps['stim_est_p'] = lfp.create_lfps(d_spikes['stim_est_p'])
d_lfps['osc_p'] = lfp.create_lfps(d_spikes['osc_p'])
d_lfps['gain_p'] = lfp.create_lfps(d_spikes['gain_p'])
d_lfps['summed_p'] = lfp.create_lfps(d_spikes['summed_p'])
d_lfps['silenced_p'] = lfp.create_lfps(d_spikes['silenced_p'])

d_lfps['stim_est_b'] = lfp.create_lfps(d_spikes['stim_est_b'])
d_lfps['osc_b'] = lfp.create_lfps(d_spikes['osc_b'])
d_lfps['gain_b'] = lfp.create_lfps(d_spikes['gain_b'])
d_lfps['summed_b'] = lfp.create_lfps(d_spikes['summed_b'])
d_lfps['silenced_b'] = lfp.create_lfps(d_spikes['silenced_b'])

d_lfps['gain_bp'] = lfp.create_lfps(d_spikes['gain_bp'])

# -- I -----------------------------------------------------------------------
# Spikes
d_infos = {}
m = 20  # Per Ince's advice

# Poisson
d_infos['stim_est_p'] = en.DiscreteSystem(
    en.quantise_discrete(stim_sp.sum(1), m),
    (1, len(stim_sp)),
    en.quantise_discrete(d_spikes['stim_est_p'].sum(1), m),
    (1, len(d_spikes['stim_est_p']))
)
d_infos['stim_est_p'].calculate_entropies(method='plugin', calc=('HX', 'HY', 'HXY'))

d_infos['gain_p'] = en.DiscreteSystem(
    en.quantise_discrete(stim_sp.sum(1), m),
    (1, len(stim_sp)),
    en.quantise_discrete(d_spikes['gain_p'].sum(1), m),
    (1, len(d_spikes['gain_p']))
)
d_infos['gain_p'].calculate_entropies(method='plugin', calc=('HX', 'HY', 'HXY'))

d_infos['summed_p'] = en.DiscreteSystem(
    en.quantise_discrete(stim_sp.sum(1), m),
    (1, len(stim_sp)),
    en.quantise_discrete(d_spikes['summed_p'].sum(1), m),
    (1, len(d_spikes['summed_p']))
)
d_infos['summed_p'].calculate_entropies(method='plugin', calc=('HX', 'HY', 'HXY'))

d_infos['silenced_p'] = en.DiscreteSystem(
    en.quantise_discrete(stim_sp.sum(1), m),
    (1, len(stim_sp)),
    en.quantise_discrete(d_spikes['silenced_p'].sum(1), m),
    (1, len(d_spikes['silenced_p']))
)
d_infos['silenced_p'].calculate_entropies(method='plugin', calc=('HX', 'HY', 'HXY'))

# Binary
d_infos['stim_est_b'] = en.DiscreteSystem(
    en.quantise_discrete(stim_sp.sum(1), m),
    (1, len(stim_sp)),
    en.quantise_discrete(d_spikes['stim_est_b'].sum(1), m),
    (1, len(d_spikes['stim_est_b']))
)
d_infos['stim_est_b'].calculate_entropies(method='plugin', calc=('HX', 'HY', 'HXY'))

d_infos['gain_b'] = en.DiscreteSystem(
    en.quantise_discrete(stim_sp.sum(1), m),
    (1, len(stim_sp)),
    en.quantise_discrete(d_spikes['gain_b'].sum(1), m),
    (1, len(d_spikes['gain_b']))
)
d_infos['gain_b'].calculate_entropies(method='plugin', calc=('HX', 'HY', 'HXY'))

d_infos['summed_b'] = en.DiscreteSystem(
    en.quantise_discrete(stim_sp.sum(1), m),
    (1, len(stim_sp)),
    en.quantise_discrete(d_spikes['summed_b'].sum(1), m),
    (1, len(d_spikes['summed_b']))
)
d_infos['summed_b'].calculate_entropies(method='plugin', calc=('HX', 'HY', 'HXY'))

d_infos['silenced_b'] = en.DiscreteSystem(
    en.quantise_discrete(stim_sp.sum(1), m),
    (1, len(stim_sp)),
    en.quantise_discrete(d_spikes['silenced_b'].sum(1), m),
    (1, len(d_spikes['silenced_b']))
)
d_infos['silenced_b'].calculate_entropies(method='plugin', calc=('HX', 'HY', 'HXY'))

d_infos['gain_bp'] = en.DiscreteSystem(
    en.quantise_discrete(stim_sp.sum(1), m),
    (1, len(stim_sp)),
    en.quantise_discrete(d_spikes['gain_bp'].sum(1), m),
    (1, len(d_spikes['gain_bp']))
)
d_infos['gain_bp'].calculate_entropies(method='plugin', calc=('HX', 'HY', 'HXY'))

# -- PAC OF LFP --------------------------------------------------------------
low_f = (f-2, f+2)
high_f = (80, 250)
# method = 'plv'
method = 'mi'
# method = 'glm'
filt = 'eegfilt'
kwargs = {'trans' : .15} # for eegfilt

d_pacs = {}
d_pacs['stim_est_p'] = scpac(d_lfps['stim_est_p'], low_f, high_f, rate, method, filt, **kwargs)
d_pacs['osc_p'] = scpac(d_lfps['osc_p'], low_f, high_f, rate, method, filt, **kwargs)
d_pacs['gain_p'] = scpac(d_lfps['gain_p'], low_f, high_f, rate, method, filt, **kwargs)
d_pacs['summed_p'] = scpac(d_lfps['summed_p'], low_f, high_f, rate, method, filt, **kwargs)
d_pacs['silenced_p'] = scpac(d_lfps['silenced_p'], low_f, high_f, rate, method, filt, **kwargs)

d_pacs['stim_est_b'] = scpac(d_lfps['stim_est_b'], low_f, high_f, rate, method, filt, **kwargs)
d_pacs['osc_b'] = scpac(d_lfps['osc_b'], low_f, high_f, rate, method, filt, **kwargs)
d_pacs['gain_b'] = scpac(d_lfps['gain_b'], low_f, high_f, rate, method, filt, **kwargs)
d_pacs['summed_b'] = scpac(d_lfps['summed_b'], low_f, high_f, rate, method, filt, **kwargs)
d_pacs['silenced_b'] = scpac(d_lfps['silenced_b'], low_f, high_f, rate, method, filt, **kwargs)

d_pacs['gain_bp'] = scpac(d_lfps['gain_bp'], low_f, high_f, rate, method, filt, **kwargs)

# # -- AUTO CORRELATION --------------------------------------------------------
# d_autocorrs = {}
# d_autocorrs['osc'] = correlogram(
#     pac.to_spiketimes(times, d_spikes['osc'])[1],
#     pac.to_spiketimes(times, d_spikes['osc'])[1],
#     width=50/1000.0,
#     bin=1/1000.0
# )
# d_autocorrs['gain'] = correlogram(
#     pac.to_spiketimes(times, d_spikes['gain'])[1],
#     pac.to_spiketimes(times, d_spikes['gain'])[1],
#     width=50/1000.0,
#     bin=1/1000.0
# )
# d_autocorrs['summed'] = correlogram(
#     pac.to_spiketimes(times, d_spikes['summed'])[1],
#     pac.to_spiketimes(times, d_spikes['summed'])[1],
#     width=50/1000.0,
#     bin=1/1000.0
# )
# d_autocorrs['silenced'] = correlogram(
#     pac.to_spiketimes(times, d_spikes['silenced'])[1],
#     pac.to_spiketimes(times, d_spikes['silenced'])[1],
#     width=50/1000.0,
#     bin=1/1000.0
# )
#
# # -- PLOTS -------------------------------------------------------------------
# # ts1, ns1 = pac.pac.to_spiketimes(times, d_spikes['silenced'])
# # plt.subplot(411)
# # plt.plot(ns1, ts1, 'o')
# #
# # ts2, ns2 = pac.pac.to_spiketimes(times, d_spikes['summed'])
# # plt.subplot(412)
# # plt.plot(ns2, ts2, 'o')
# #
# # ts2, ns2 = pac.pac.to_spiketimes(times, d_spikes['gain'])
# # plt.subplot(413)
# # plt.plot(ns2, ts2, 'o')
# #
# # plt.subplot(414)
# # plt.plot(times, osc, label='osc')
# # plt.plot(times, stim, label='stim')
# # plt.plot(times, gain, label='gain')
# # plt.plot(times, summed, label='sum')
# # plt.legend(loc='best')
# #
# # from matplotlib.backends.backend_pdf import PdfPages
# # pp = PdfPages('test_pac.pdf',)
# # pp.savefig()
# # pp.close()
# #
#
