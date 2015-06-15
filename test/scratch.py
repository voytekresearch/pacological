# %load test_pac.py
"""Dev testing"""
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
t = 1

Iosc = 1
f = 10

Istim = 1
Sstim = .1  # Keep this fairly small

dt = 0.001
rate = 1 / dt

# -- SIM ---------------------------------------------------------------------
# Init spikers
modspikes = pac.Spikes(n, t, dt=dt)
drivespikes = pac.Spikes(n, t, dt=dt)
times = modspikes.times  # brevity

bias = pac.constant(times, 1)
s_p = modspikes.poisson(bias)
s_b = modspikes.binomial(bias, max_rate=1000)

# # Create biases
# d_bias = {}
# d_bias['osc'] = pac.osc(times, Iosc, f)
# # d_bias['stim'] = pac.stim(times, Istim, Sstim)
# d_bias['stim'] = pac.constant(times, Istim)
#
# # Simulate spiking
# # Create a fixed noiseless stimulus pattern, then....
# stim_sp_p = drivespikes.poisson(d_bias['stim'])
# stim_sp_b = drivespikes.binomial(d_bias['stim'], max_rate=500)
#
# stim_spks = modspikes.poisson(d_bias['stim'])
# isi = pac.isi(pac.to_spikedict(*pac.to_spiketimes(times, stim_spks)))
#
# # study how different modulation schemes interact
# # with it
# d_spikes = {}
# d_spikes['stim'] = stim_sp_p
# d_spikes['stim_b'] = stim_sp_b
# d_spikes['stim_est'] = modspikes.poisson(d_bias['stim'])
# d_spikes['osc'] = modspikes.poisson(d_bias['osc'])
# d_spikes['gain'] = modspikes.multiply(d_bias['stim'], d_bias['osc'])
# d_spikes['gain'] = modspikes.multiply(d_bias['stim'], d_bias['osc'])
# d_spikes['summed'] = modspikes.add(d_bias['stim'], d_bias['osc'])
# d_spikes['silenced'] = modspikes.subtract(d_bias['stim'], d_bias['osc'])
# d_spikes['sync'] = modspikes.poisson_binomial(
#     d_bias['stim'], d_bias['osc'], amplitude=False, max_rate=1000)
# d_spikes['sync_gain'] = modspikes.poisson_binomial(
#     d_bias['stim'], d_bias['osc'], amplitude=True, max_rate=1000)
#
# # -- PLOTS -------------------------------------------------------------------
# ts1, ns1 = pac.to_spiketimes(times, d_spikes['stim'])
# plt.subplot(211)
# plt.plot(ns1, ts1, 'o')
#
# ts2, ns2 = pac.to_spiketimes(times, d_spikes['stim_b'])
# plt.subplot(212)
# plt.plot(ns2, ts2, 'o')
#
# # ts2, ns2 = pac.to_spiketimes(times, d_spikes['gain'])
# # plt.subplot(413)
# # plt.plot(ns2, ts2, 'o')
# #
# # ts2, ns2 = pac.to_spiketimes(times, d_spikes['stim'])
# # plt.subplot(414)
# # plt.plot(ns2, ts2, 'o')
# # plt.subplot(414)
# # plt.plot(times, d_lf, label='osc')
# # plt.plot(times, stim, label='stim')
# # plt.plot(times, gain, label='gain')
# # plt.plot(times, summed, label='sum')
# # plt.legend(loc='best')
#
# from matplotlib.backends.backend_pdf import PdfPages
# pp = PdfPages('test/scratch.pdf',)
# pp.savefig()
# pp.close()
#
#
