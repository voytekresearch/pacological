"""Save some demo data for figures"""
import sys
import os
import pandas as pd
import numpy as np
from pacological import pac
from noisy import lfp

import seaborn as sns
import matplotlib.pyplot as plt; plt.ion()

path = sys.argv[1]

# -- USER SETTINGS -----------------------------------------------------------
n = 500
t = 10

Iosc = 6
f = 10

Istim = 3
Sstim = .05

dt = 0.001
rate = 1 / dt

k = 80
excitability = 0.00001
excitability_be  = 0.1

# -- SIM ---------------------------------------------------------------------
# Init spikers
modspikes = pac.Spikes(n, t, dt=dt)
drivespikes = pac.Spikes(n, t, dt=dt)
times = modspikes.times  # brevity

# Create biases
d_bias = {}
d_bias['times'] = times
d_bias['osc'] = pac.osc(times, Iosc, f)
d_bias['stim'] = pac.stim(times, Istim, Sstim)
d_bias['gain'] = d_bias['osc'] * d_bias['stim']
d_bias['summed'] = d_bias['osc'] + d_bias['stim']
d_bias['silenced'] = d_bias['stim'] - d_bias['osc']
df_bias = pd.DataFrame(d_bias)
df_bias.to_csv(os.path.join(path, "bias.csv"), index=False, header=True)

# Simulate spiking
# Create a fixed noiseless stimulus pattern, then....
stim_sp = drivespikes.poisson(d_bias['stim'])

# study how different modulation schemes interact
# with it
d_spikes = {}

# Poisson spikes
d_spikes['stim_est_p'] = modspikes.poisson(d_bias['stim'])
ns, ts = pac.to_spiketimes(times, d_spikes['stim_est_p'])
pd.DataFrame(np.vstack([ns, ts]).T, columns=['neuron', 'time']).to_csv(
    os.path.join(path, "spikes_stim_est_p.csv"), index=False)
plt.subplot(211)
plt.plot(ts, ns, 'o')

d_spikes['osc_p'] = modspikes.poisson(d_bias['osc'])
ns, ts = pac.to_spiketimes(times, d_spikes['osc_p'])
pd.DataFrame(np.vstack([ns, ts]).T, columns=['neuron', 'time']).to_csv(
    os.path.join(path, "spikes_osc_p.csv"), index=False)

d_spikes['gain_p'] = modspikes.poisson(d_bias['gain'])
ns, ts = pac.to_spiketimes(times, d_spikes['gain_p'])
pd.DataFrame(np.vstack([ns, ts]).T, columns=['neuron', 'time']).to_csv(
    os.path.join(path, "spikes_gain_p.csv"), index=False)

d_spikes['summed_p'] = modspikes.poisson(d_bias['summed'])
ns, ts = pac.to_spiketimes(times, d_spikes['summed_p'])
pd.DataFrame(np.vstack([ns, ts]).T, columns=['neuron', 'time']).to_csv(
    os.path.join(path, "spikes_summed_p.csv"), index=False)

d_spikes['silenced_p'] = modspikes.poisson(d_bias['silenced'])
ns, ts = pac.to_spiketimes(times, d_spikes['silenced_p'])
pd.DataFrame(np.vstack([ns, ts]).T, columns=['neuron', 'time']).to_csv(
    os.path.join(path, "spikes_silenced_p.csv"), index=False)

# Binary spikes
d_spikes['stim_est_b'] = modspikes.binary(d_bias['stim'], k=k, excitability=excitability)
ns, ts = pac.to_spiketimes(times, d_spikes['stim_est_b'])
pd.DataFrame(np.vstack([ns, ts]).T, columns=['neuron', 'time']).to_csv(
    os.path.join(path, "spikes_stim_est_b.csv"), index=False)

d_spikes['osc_b'] = modspikes.binary(d_bias['osc'], k=k, excitability=excitability)
ns, ts = pac.to_spiketimes(times, d_spikes['osc_b'])
pd.DataFrame(np.vstack([ns, ts]).T, columns=['neuron', 'time']).to_csv(
    os.path.join(path, "spikes_osc_b.csv"), index=False)

d_spikes['gain_b'] = modspikes.binary(d_bias['gain'], k=k, excitability=excitability)
ns, ts = pac.to_spiketimes(times, d_spikes['gain_b'])
pd.DataFrame(np.vstack([ns, ts]).T, columns=['neuron', 'time']).to_csv(
    os.path.join(path, "spikes_gain_b.csv"), index=False)

d_spikes['summed_b'] = modspikes.binary(d_bias['summed'], k=k, excitability=excitability)
ns, ts = pac.to_spiketimes(times, d_spikes['summed_b'])
pd.DataFrame(np.vstack([ns, ts]).T, columns=['neuron', 'time']).to_csv(
    os.path.join(path, "spikes_summed_b.csv"), index=False)

d_spikes['silenced_b'] = modspikes.binary(d_bias['silenced'], k=k, excitability=excitability)
ns, ts = pac.to_spiketimes(times, d_spikes['silenced_b'])
pd.DataFrame(np.vstack([ns, ts]).T, columns=['neuron', 'time']).to_csv(
    os.path.join(path, "spikes_silenced_b.csv"), index=False)

d_spikes['gain_bp'] = modspikes.poisson_binary(d_bias['stim'], d_bias['osc'], k=k, excitability=excitability)
ns, ts = pac.to_spiketimes(times, d_spikes['gain_bp'])
pd.DataFrame(np.vstack([ns, ts]).T, columns=['neuron', 'time']).to_csv(
    os.path.join(path, "spikes_gain_bp.csv"), index=False)

# Beroulli
d_spikes['stim_est_be'] = modspikes.bernoulli(d_bias['stim'], excitability=excitability_be)
ns, ts = pac.to_spiketimes(times, d_spikes['stim_est_be'])
pd.DataFrame(np.vstack([ns, ts]).T, columns=['neuron', 'time']).to_csv(
    os.path.join(path, "spikes_stim_est_be.csv"), index=False)

plt.subplot(212)
plt.plot(ts, ns, 'o')

d_spikes['osc_be'] = modspikes.bernoulli(d_bias['osc'], excitability=excitability_be)
ns, ts = pac.to_spiketimes(times, d_spikes['osc_be'])
pd.DataFrame(np.vstack([ns, ts]).T, columns=['neuron', 'time']).to_csv(
    os.path.join(path, "spikes_osc_be.csv"), index=False)

d_spikes['gain_be'] = modspikes.bernoulli(d_bias['gain'], excitability=excitability_be)
ns, ts = pac.to_spiketimes(times, d_spikes['gain_be'])
pd.DataFrame(np.vstack([ns, ts]).T, columns=['neuron', 'time']).to_csv(
    os.path.join(path, "spikes_gain_be.csv"), index=False)

d_spikes['summed_be'] = modspikes.bernoulli(d_bias['summed'], excitability=excitability_be)
ns, ts = pac.to_spiketimes(times, d_spikes['summed_be'])
pd.DataFrame(np.vstack([ns, ts]).T, columns=['neuron', 'time']).to_csv(
    os.path.join(path, "spikes_summed_be.csv"), index=False)

d_spikes['silenced_be'] = modspikes.bernoulli(d_bias['silenced'], excitability=excitability_be)
ns, ts = pac.to_spiketimes(times, d_spikes['silenced_be'])
pd.DataFrame(np.vstack([ns, ts]).T, columns=['neuron', 'time']).to_csv(
    os.path.join(path, "spikes_silenced_be.csv"), index=False)

d_spikes['gain_bep'] = modspikes.poisson_bernoulli(d_bias['stim'], d_bias['osc'], excitability=excitability_be)
ns, ts = pac.to_spiketimes(times, d_spikes['gain_bep'])
pd.DataFrame(np.vstack([ns, ts]).T, columns=['neuron', 'time']).to_csv(
    os.path.join(path, "spikes_gain_bep.csv"), index=False)

# -- CREATE LFP --------------------------------------------------------------
d_lfps = {}
d_lfps['times'] = times
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


df_lfps = pd.DataFrame(d_lfps)
df_lfps.to_csv(os.path.join(path, "lfps.csv"), index=False, header=True)
