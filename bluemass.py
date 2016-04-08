"""Usage: bluemass.py [-t T -f F] 
    [--r_back=RATE_E,RATE_I] [--w_back=W_E,W_I] PATH PARS_FILE

Simulate the Blue Brain using gNMMs.

    Arguments:
        PATH        where to save the results
        PARS_FILE   parameters file (python code)

    Options:
        -h --help   show this screen
        -t T        simultation run time [default: 0.2] 
        -f F        background oscillation frequency [default: 10]
        --r_back=RATE_E,RATE_I  background firing rate (Hz)
        --w_back=W_E,W_I        background wieghts (nsiemens)        

"""
from __future__ import division
from docopt import docopt
import os, sys

from scipy.integrate import odeint, ode
from sdeint import itoint

import numpy as np
from functools import partial
from numpy import random
from copy import deepcopy

from fakespikes.rates import stim
from pacological.util import create_I, ornstein_uhlenbeck, progressbar

# Control background 
SEED = 42
prng = random.RandomState(SEED)


def logistic(x, x0, k, L):
    return L / (1 + np.exp(-k * (x - x0)))


def phi(Isyn, I, g0, gs, sigma):
    tau = 10e-3  # second
    Vth = -55e-03  # volt
    Vreset = -60e-3 

    # Compute the single unit respose
    g = g0 + gs
    k = (Isyn / g * sigma)
    a = ((g * sigma) / (tau * g0 * (Vth - Vreset))) 
    z = 1 / a * logistic(Isyn, I, 1/k, 500)      # supp to explore 2000?

    # Transform into population activity
    # Isigma = Isyn * 0.1  # TODO
    # Id = Idist(Isyn, I, Isigma)

    return z  


def _background(t, f, rbe, rbi):
    if f > 0:
        # oscillation
        rbe_t = rbe * np.cos(2 * np.pi * f * t)
        rbi_t = rbi * np.cos(2 * np.pi * f * t)
    else:
        # fixed rate
        rbe_t = deepcopy(rbe)
        rbi_t = deepcopy(rbi)

    if rbe_t < 0:
        rbe_t = 0
    if rbi_t < 0:
        rbi_t = 0

    rbe_t = prng.poisson(rbe_t, 1) + prng.poisson(135, 1) 
    rbi_t = prng.poisson(rbi_t, 1) + prng.poisson(135, 1)
    
    return np.asarray([rbe_t[0], rbi_t[0]])


def create_layers(f, stim, pars):
    
    # Unpack pars for readability
    Z = pars.Z
    C = pars.C
    W = pars.W
    T = pars.T
    V = pars.V
    K = pars.K
    Wb = pars.Wb
    Cb = pars.Cb
    Kb = pars.Kb 
    Rb = pars.Rb
    I_e = pars.I_e
    I_i = pars.I_i
    tau_m = pars.tau_m
    stim_index = pars.stim_i
    back_index = pars.back_i

    # Some useful sizes
    n = Z.shape[0]
    c = Z.size

    # Create indices to pack/repack the matrices
    i0 = 0
    ik = n
    idx_r = range(i0, ik)

    i0 = ik
    ik += c
    idx_g = range(i0, ik)

    i0 = ik
    ik += c
    idx_s = range(i0, ik)

    i0 = ik
    ik += 2
    idx_gb = range(i0, ik)

    i0 = ik
    ik += 2
    idx_sb = range(i0, ik)
   
    # Setup bias
    I_bias = np.zeros(len(idx_r))
    I_bias[np.diagonal(T) == 1] = I_e
    I_bias[np.diagonal(T) == -1] = I_i

    # Setup synapses
    I_syn = np.zeros(len(idx_r))
    I_tmp = np.zeros(len(idx_r))
    g0 = np.zeros_like(I_syn)
    gb = np.zeros_like(I_syn)
    sb = np.zeros_like(I_syn)

    # Define the function to integrate
    def layers(ys, t):
        """A layered gNMM model."""

        R = ys[idx_r]
        G = ys[idx_g].reshape(n, n)
        S = ys[idx_s].reshape(n, n)
        Gb = ys[idx_gb]
        Sb = ys[idx_sb]

        # Eeek, nan!
        R[np.isnan(R)] = 0
        G[np.isnan(G)] = 0
        S[np.isnan(S)] = 0
        Gb[np.isnan(Gb)] = 0
        Sb[np.isnan(Sb)] = 0

        # Reinit for this iteration
        # for safety sake
        ys = np.zeros_like(ys)

        # Step rates
        for j in idx_r:
            I_syn[j] = np.dot(G[:, j], V[:, j])

            if j in stim_index:
                I_syn[j] += stim(t)

            if j in back_index:
                g0[j] = np.dot(Wb, Cb) / len(Wb) / 0.04
                gb[j] = np.abs(np.sum(np.concatenate([Gb, G[:, j]])))
                sb[j] = np.sum(np.concatenate([Sb, S[:, j]]))
            else:
                g0[j] = np.dot(W[:, j], C[:, j]) / len(W[:, j]) / 0.04
                gb[j] = np.abs(np.sum(G[:, j]))                  
                
                # Need to def two background one mod one not.
                sb[j] = np.sum(np.concatenate([Sb, S[:, j]]))   

        ys[idx_r] = (-R + phi(I_syn, I_bias, g0, gb, np.sqrt(sb))) / tau_m

        # Step all synapses
        R2 = np.vstack([R] * R.shape[0]).T
        ys[idx_g] = ((-G / K) + (C * W * R2)).flatten()
        ys[idx_s] = (((-2 * S) / K) + (((C * W) ** 2) * R2)).flatten()
        del R2

        # Step gain control
        Rb_t = _background(t, f, Rb[0], Rb[1])
        ys[idx_gb] = ((-Gb / Kb) + (Cb * Wb * Rb_t)).flatten()
        ys[idx_sb] = (((-2 * Sb) / Kb) + ((Cb * Wb) ** 2 * Rb_t)).flatten()
        del Rb_t

        return ys
    
    idxs = {'R':idx_r, 'G':idx_g, 'S':idx_s, 'Gb':idx_gb, 'Sb':idx_sb}

    return layers, idxs

    
if __name__ == "__main__":
    args = docopt(__doc__, version='Alpha')

    # Simulation parameters ----------------------------------------- 
    print(">>> Building the model.")
    save_path = args['PATH']
    execfile(args['PARS_FILE'])  # returns 'pars'

    tmax = float(args['-t'])
    step = 0.1  # Don't go larger than 0.2
    n_step = int(np.ceil(tmax / step))

    f = float(args['-f'])

    if args['--r_back'] is not None:
        pars.Rb = np.asarray([float(a) for a in args['--r_back'].split(',')])
    if args['--w_back'] is not None:
        pars.Wb = np.asarray([float(a) for a in args['--w_back'].split(',')])

    # Simulation input ---------------------------------------------- 
    dt = 1e-4

    # Create stimulus
    scale = 0.001
    Istim = create_I(tmax, pars.I_e, scale * pars.I_e, dt=dt, seed=1)

    # Define systems to integrate
    layers_f0, idxs = create_layers(0, Istim, pars)
    layers, _ = create_layers(f, Istim, pars)
    g = partial(ornstein_uhlenbeck, sigma=0.01, loc=pars.stim_i) 

    # Init ys0
    max_n = max(max([v[:] for v in idxs.values()]))  # A dirty way to get max 
    ys0 = np.zeros(max_n + 1)
    ys0[idxs['R']] = pars.R0
    ys0[idxs['G']] = (pars.W / 2).flatten()
    ys0[idxs['S']] = ((pars.W / 2) ** 2).flatten()
    ys0[idxs['Gb']] = pars.Wb / 2 
    ys0[idxs['Sb']] = (pars.Wb ** 2) / 2

    # Run -----------------------------------------------------------
    print(">>> Running baseline.")
    ys_ts = ys0
    t0 = 0.0
    ts = t0 + step
    for k in progressbar(range(n_step)):
        times = np.linspace(t0, ts, int(step / dt))

        ys = itoint(layers_f0, g, ys_ts, times)

        np.savez(
            os.path.join(save_path, 'ys_base_{}'.format(k)), 
            ys=ys, idxs=idxs, times=times
        )

        t0 = deepcopy(ts)
        ts += step
        ys_ts = deepcopy(ys[-1, :])
        del ys

    print(">>> Running modulation.")
    ys_ts = ys0
    t0 = 0.0
    ts = t0 + step
    for k in progressbar(range(n_step)):
        times = np.linspace(t0, ts, int(step / dt))

        ys = itoint(layers, g, ys_ts, times)

        np.savez(
            os.path.join(save_path, 'ys_{}'.format(k)), 
            ys=ys, idxs=idxs, times=times
        )

        t0 = deepcopy(ts)
        ts += step
        ys_ts = deepcopy(ys[-1, :])
        del ys

    # Save params
    np.savez(os.path.join(save_path, "run_pars"), 
            tmax=tmax, 
            dt=dt,
            step=step,
            n_step=n_step,
            f=f, 
            scale=scale, 
            times=times, 
            max_n=max_n, 
            ys0=ys0,
            stim=np.asarray([Istim(t) for t in times]))

    np.savez(os.path.join(save_path, "pars"), **pars.__dict__)
