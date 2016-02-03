"""Balance, excess re, and excess ri using an HH neuron"""
import os
import sys
from pacological.hh import gain, exp
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from convenience.numpy import save_hdfz


path = sys.argv[1]

# Null osc
f = 0
t = 4

Is = np.linspace(0, 3, 10)
xfactors = [0, 1, 2, 3, 4, 5, 6]

n_trial = 10

# --
print('classic')
rates = np.zeros((len(Is), len(xfactors)))
gs = np.zeros((len(Is), len(xfactors)))
vs = np.zeros((len(Is), len(xfactors)))
for i in range(len(Is)):
    for j in range(len(xfactors)):
        M_g, M_v, rate = [], [], []
        for n in range(n_trial):
            
            I = Is[i]
            xfactor = xfactors[j]

            res = exp(t, I, xfactor, f=f)

            spikes = res['spikes']
            traces = res['traces']
            
            rate.append(spikes.t_[:].shape[0] / t)
            
            M_g.append(np.mean(traces.g_e_[0] + traces.g_i_[0]))
            M_v.append(np.mean(traces.v_[0]))
        
        rates[i, j] = np.mean(rate)
        gs[i, j] = np.mean(M_g)
        vs[i, j] = np.mean(M_v)
        
    print(I)
    
save_hdfz(os.path.join(path, 'classic'), Is=Is, rates=rates, gs=gs,
        vs=vs, xfactors=xfactors)

# --
print('re')
rates = np.zeros((len(Is), len(xfactors)))
gs = np.zeros((len(Is), len(xfactors)))
vs = np.zeros((len(Is), len(xfactors)))
for i in range(len(Is)):
    for j in range(len(xfactors)):
        M_g, M_v, rate = [], [], []
        for n in range(n_trial):
            
            I = Is[i]
            xfactor = xfactors[j]

            res = exp(t, I, (xfactor, 1), f=f)

            spikes = res['spikes']
            traces = res['traces']
            
            rate.append(spikes.t_[:].shape[0] / t)
            
            M_g.append(np.mean(traces.g_e_[0] + traces.g_i_[0]))
            M_v.append(np.mean(traces.v_[0]))
        
        rates[i, j] = np.mean(rate)
        gs[i, j] = np.mean(M_g)
        vs[i, j] = np.mean(M_v)
        
    print(I)
    
save_hdfz(os.path.join(path, 're'), Is=Is, rates=rates, gs=gs,
        vs=vs, xfactors=xfactors)

# --
print('ri')
rates = np.zeros((len(Is), len(xfactors)))
gs = np.zeros((len(Is), len(xfactors)))
vs = np.zeros((len(Is), len(xfactors)))
for i in range(len(Is)):
    for j in range(len(xfactors)):
        M_g, M_v, rate = [], [], []
        for n in range(n_trial):
            
            I = Is[i]
            xfactor = xfactors[j]

            res = exp(t, I, (1, xfactor), f=f)

            spikes = res['spikes']
            traces = res['traces']
            
            rate.append(spikes.t_[:].shape[0] / t)
            
            M_g.append(np.mean(traces.g_e_[0] + traces.g_i_[0]))
            M_v.append(np.mean(traces.v_[0]))
        
        rates[i, j] = np.mean(rate)
        gs[i, j] = np.mean(M_g)
        vs[i, j] = np.mean(M_v)
        
    print(I)
    
save_hdfz(os.path.join(path, 'ri'), Is=Is, rates=rates, gs=gs,
        vs=vs, xfactors=xfactors)

