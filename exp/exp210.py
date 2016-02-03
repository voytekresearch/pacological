
"""How does gain scale with oscillation f using the Fitzhugh-Nagumo neuron"""
import os
import sys
from pacological.fitz import gain, exp
import numpy as np
from joblib import Parallel, delayed
from convenience.numpy import save_hdfz


path = sys.argv[1]
n_jobs = 10

# --
fs = range(1, 60, 2)
for f in fs:
    t = 4

    n_trial = 20
    xfactors = np.arange(1, 6, 1)
    Is = np.arange(0, 1.1, .1)

    # --
    # Init
    rates = np.zeros((len(Is), 1 + len(xfactors)))

    # Do 1X first, put it in the 0th column
    print("1X")
    for i, I in enumerate(Is):    
        rtmp = []

        def fn(trial):  # Closes globals
            res = exp(t, I, 1, f=0)
            spikes = res['spikes']
            return np.mean(spikes.t_[:].shape[0] / t)
        
        rates[i, 0] = np.mean(
            Parallel(n_jobs=n_jobs)(delayed(fn)(trial) for trial in range(n_trial))
        )
        
        print(I)

    # Now do all the xfactors for osc
    print("osc")
    for i, I in enumerate(Is):
        for j, xfactor in enumerate(xfactors): 
            
            def fn(trial):  # Closes globals
                res = exp(t, I, xfactor, f=f)
                spikes = res['spikes']
                return np.mean(spikes.t_[:].shape[0] / t)
        
            rates[i, j + 1] = np.mean(
                Parallel(n_jobs=n_jobs)(delayed(fn)(trial) for trial in range(n_trial))
            )
        
        print(I)

    save_hdfz(os.path.join(path, 'limits_{}'.format(f)), 
            Is=Is, rates=rates, xfactors=xfactors, f=f, 
            t=t, n_trial=n_trial)

