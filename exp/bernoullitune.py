"""Tune Bernoilli firing/excitibility"""

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

Istim = 1
Sstim = .05

dt = 0.001
rate = 1 / dt

drivespikes = pac.Spikes(n, t, dt=dt)
times = drivespikes.times  # brevity

# Create biases
d_bias = {}
d_bias['times'] = times
d_bias['osc'] = pac.osc(times, Iosc, f)
d_bias['stim'] = pac.stim(times, Istim, Sstim)
d_bias['constant'] = pac.constant(times, Istim)
d_bias['gain'] = d_bias['osc'] * d_bias['stim']
d_bias['summed'] = d_bias['osc'] + d_bias['stim']
d_bias['silenced'] = d_bias['stim'] - d_bias['osc']

# sp_bin = drivespikes.binary(d_bias['stim'], k=10, excitability=0.0001)
sp_bin = drivespikes.binary(d_bias['stim'], k=20, excitability=0.0001/2)
sp_poi = drivespikes.poisson(d_bias['stim'])

print("Rates for Poi {0} and Bin {1}".format(
    sp_poi.sum() / n / float(t),
    sp_bin.sum() / n / float(t)
))

plt.subplot(211)
ns_poi, ts_poi = pac.to_spiketimes(times, sp_poi[:,:200])
plt.plot(ts_poi, ns_poi, marker='o', linestyle='None')

plt.subplot(212)
ns_bin, ts_bin = pac.to_spiketimes(times, sp_bin[:,:200])
plt.plot(ts_bin, ns_bin, marker='o', linestyle='None')
