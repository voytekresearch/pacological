"""Oscillation at 10 Hz, with a reduced sigma (HH neuron)"""
import os
import sys
from pacological.hh import gain, exp
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from convenience.numpy import save_hdfz


path = sys.argv[1]
n_jobs = 10

t = 2
n_trial = 10

Is = np.linspace(0, 50, 20)
xfactors = [1, 2, 3]
fs = [5, 8, 10, 12, 15, 20, 25, 30, 35]

w = 100
k = 5
r = 200

# --
# Do 1X first
print("1X")
rate1x = []
for i, I in enumerate(Is):
    rtmp = []

    def fn(trial):  
        res = exp(t, I, 1, f=0, r=r, w_e=k * w, w_i=k * w * 4)
        spikes = res['spikes']
        return np.mean(spikes.t_[:].shape[0] / t)

    rate1x.append(np.mean(
        Parallel(n_jobs=n_jobs)(
            delayed(fn)(trial) for trial in range(n_trial))
    ))

    print(I)
rate1x = np.asarray(rate1x)

# --
for f in fs:
    print(f)

    # Init
    rates = np.zeros((len(Is), 1 + len(xfactors)))
    rates[:, 0] = rate1x

    # Now do all the xfactors for osc
    print("osc")
    for i, I in enumerate(Is):
        for j, xfactor in enumerate(xfactors):

            def fn(trial):  # Closes globals
                res = exp(t, I, xfactor, f=f, r=r, w_e=k * w, w_i=k * w * 4)
                spikes = res['spikes']
                return np.mean(spikes.t_[:].shape[0] / t)

            rates[i, j + 1] = np.mean(
                Parallel(n_jobs=n_jobs)(
                    delayed(fn)(trial) for trial in range(n_trial))
            )

        print(I)

    # --
    save_hdfz(os.path.join(path, 'limits_{}'.format(f)),
              Is=Is, rates=rates, xfactors=xfactors,
              f=f, t=t, n_trial=n_trial)
