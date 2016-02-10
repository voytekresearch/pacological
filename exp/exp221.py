"""Generate the base FI data for the HH NMM system"""
import os
import sys
from pacological.hh import gain, exp
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from convenience.numpy import save_hdfz


path = sys.argv[1]
n_jobs = 12

t = 2
n_trial = 30

Imax = # ??
Is = np.linspace(0, Imax, 50)
xfactors = [1, ]

w = 100 
k = 5

# --
# Do 1X first, (put it in the 0th column for every f)
rate1x = []
for i, I in enumerate(Is):
    rtmp = []

    def fn(trial):  # Closes globals
        # reduce w_e/i and so sigma compared to exp217 by an
        # order of magnitude
        res = exp(t, I, 1, f=0, w_e=k * w, w_i=k * w * 4)
        spikes = res['spikes']
        return np.mean(spikes.t_[:].shape[0] / t)

    rate1x.append(np.mean(
        Parallel(n_jobs=n_jobs)(
            delayed(fn)(trial) for trial in range(n_trial))
    ))

    print(I)
rate1x = np.asarray(rate1x)

save_hdfz(os.path.join(path, 'limits_{}'.format(f)),
          Is=Is, rates=rate1x, xfactors=xfactors,
          f=f, t=t, n_trial=n_trial)
