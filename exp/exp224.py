import os
import sys
from pacological.hh import gain
import matplotlib.pyplot as plt; plt.ion()
import numpy as np
from joblib import Parallel, delayed
from convenience.numpy import save_hdfz, load_hdfz

# --
path = sys.argv[1]
n_jobs = 10

t = 5
n_trial = 20
Is = np.linspace(-10, 50, 50)

# --
print("1X")
w_m = 0
w = 100
k = 5
r = 400

rates = []
for i, I in enumerate(Is):
    # Define a fn for this iteration,
    # and run it in parallel over n_trials.
    # Save the trial averaged rate.
    def fn(trial):  
        res = gain(t, 
                   r_e=r, 
                   r_i=r, 
                   w_e=k * w, 
                   w_i=k * w * 4, 
                   w_m=w_m, 
                   I_drive=I, 
                   f=0, 
                   verbose=False)
        
        spikes = res['spikes']
        return np.mean(spikes.t_[:].shape[0] / t)  # avg rate

    rates.append( # overall trial avg
        np.mean(
            Parallel(n_jobs=n_jobs)(
                delayed(fn)(trial) for trial in range(n_trial)
            )
        )
    )
    
    print(I)
rates = np.asarray(rates)

save_hdfz(os.path.join(path, 'r{0}_wm{1}'.format(r, w_m)), 
        rate=rates, Is=Is, k=k, t=t, r=r, w_m=w_m)

# --
print("0X")
w_m = 0
w = 100
k = 5
r = 0

rates = []
for i, I in enumerate(Is):
    # Define a fn for this iteration,
    # and run it in parallel over n_trials.
    # Save the trial averaged rate.
    def fn(trial):  
        res = gain(t, 
                   r_e=r, 
                   r_i=r, 
                   w_e=k * w, 
                   w_i=k * w * 4, 
                   w_m=w_m, 
                   I_drive=I, 
                   f=0, 
                   verbose=False)
        
        spikes = res['spikes']
        return np.mean(spikes.t_[:].shape[0] / t)  # avg rate

    rates.append( # overall trial avg
        np.mean(
            Parallel(n_jobs=n_jobs)(
                delayed(fn)(trial) for trial in range(n_trial)
            )
        )
    )
    
    print(I)
rates = np.asarray(rates)

save_hdfz(os.path.join(path, 'r{0}_wm{1}'.format(r, w_m)), 
        rate=rates, Is=Is, k=k, t=t, r=r, w_m=w_m)

# --
print("0X + w_m")
w_m = 60
w = 100
k = 5
r = 0

rates = []
for i, I in enumerate(Is):
    # Define a fn for this iteration,
    # and run it in parallel over n_trials.
    # Save the trial averaged rate.
    def fn(trial):  
        res = gain(t, 
                   r_e=r, 
                   r_i=r, 
                   w_e=k * w, 
                   w_i=k * w * 4, 
                   w_m=w_m, 
                   I_drive=I, 
                   f=0, 
                   verbose=False)
        
        spikes = res['spikes']
        return np.mean(spikes.t_[:].shape[0] / t)  # avg rate

    rates.append( # overall trial avg
        np.mean(
            Parallel(n_jobs=n_jobs)(
                delayed(fn)(trial) for trial in range(n_trial)
            )
        )
    )
    
    print(I)
rates = np.asarray(rates)

save_hdfz(os.path.join(path, 'r{0}_wm{1}'.format(r, w_m)), 
        rate=rates, Is=Is, k=k, t=t, r=r, w_m=w_m)

# --
print("1X + w_m")
w_m = 60
w = 100
k = 5
r = 400

rates = []
for i, I in enumerate(Is):
    # Define a fn for this iteration,
    # and run it in parallel over n_trials.
    # Save the trial averaged rate.
    def fn(trial):  
        res = gain(t, 
                   r_e=r, 
                   r_i=r, 
                   w_e=k * w, 
                   w_i=k * w * 4, 
                   w_m=w_m, 
                   I_drive=I, 
                   f=0, 
                   verbose=False)
        
        spikes = res['spikes']
        return np.mean(spikes.t_[:].shape[0] / t)  # avg rate

    rates.append( # overall trial avg
        np.mean(
            Parallel(n_jobs=n_jobs)(
                delayed(fn)(trial) for trial in range(n_trial)
            )
        )
    )
    
    print(I)
rates = np.asarray(rates)

save_hdfz(os.path.join(path, 'r{0}_wm{1}'.format(r, w_m)), 
        rate=rates, Is=Is, k=k, t=t, r=r, w_m=w_m)
