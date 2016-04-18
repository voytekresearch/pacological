"""Usage: bluemass.py [-t T] [-f F] [-s STIM_SEED]
    [--step=STEP]
    [--r_back=RATE_E,RATE_I] [--w_back=W_E,W_I] 
    NAME PARS_FILE

Simulate the Blue Brain using gNMMs.

    Arguments:
        NAME        name (and path) of the results files
        PARS_FILE   parameters file (a BMparams() instance)

    Options:
        -h --help               show this screen
        -t T                    simultation run time [default: 1.0] 
        -f F                    background modulation frequency [default: 10]
        -s STIM_SEED            seed for creating the stimulus [default: 1]
        --step=STEP             step length before writing [default: 0.1]
        --r_back=RATE_E,RATE_I  background firing rate (Hz)
        --w_back=W_E,W_I        background wieghts (nsiemens)        

"""
from __future__ import division
from docopt import docopt
import os, sys
import pudb

from scipy.integrate import odeint, ode
from sdeint import itoint

import numpy as np
from functools import partial
from numpy import random
from copy import deepcopy

from fakespikes.rates import stim
from pacological.util import create_I, ornstein_uhlenbeck, progressbar
from pacological.pars import perturb_params


# Control background 
SEED = 42
prng = random.RandomState(SEED)


# def logistic(x, x0, k, L):
#     return L / (1 + np.exp(-k * (x - x0)))
#
#
# # TODO : use tanh for the non-linearity instead of logistic
# # scaling the latter is part of the problem?
# # TODO compare eq 13,14 to 20,21 in Zandt 2014, you are missing 
# # something important?
# def phi(Isyn, I, g0, gs, sigma, L=500):
#     tau = 20e-3  # second
#     Vth = -55e-03  # volt
#     Vreset = -60e-3 
#
#     # Compute the single unit respose
#     g = g0 + gs
#     k = Isyn / (g * sigma)
#     a = (g * sigma) / (tau * g0 * (Vth - Vreset))
#     # z = (1 / a) * logistic(Isyn, I, 1/k, L)      # supp to explore 2000?
#     z = (1 / a) * np.tanh(k)
#     # Transform into population activity
#     # Isigma = Isyn * 0.1  # TODO
#     # Id = Idist(Isyn, I, Isigma)
#
#     return z  


def phi(Isyn, g0, gs, sigma):
    tau = 2e-3  # second
    Vth = -55e-03  # volt
    Vreset = -60e-3 

    # Compute the single unit respose
    g = g0 + gs
    a = (g * sigma) / (tau * g0 * (Vth - Vreset))
    x = Isyn / (g * sigma)
    z = a * np.tanh(x)
    
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

    rbe_t = prng.poisson(rbe_t, 1) + 12   # rate can't fall below 12
    rbi_t = prng.poisson(rbi_t, 1) + 12
    
    return np.asarray([rbe_t[0], rbi_t[0]])


def create_layers(f, stim, pars):
    
    # Unpack pars for readability
    stim_index = pars.stim_i
    back_index = pars.back_i

    Z = pars.Z
    C = pars.C
    W = pars.W
    T = pars.T
    V = pars.V
    K = pars.K

    rbe = pars.rbe
    rbi = pars.rbi

    Kbe = pars.Kbe
    Cbe = pars.Cbe
    Wbe = pars.Wbe
    Kbi = pars.Kbi
    Cbi = pars.Cbi
    Wbi = pars.Wbi

    I_e = pars.I_e
    I_i = pars.I_i
    tau_m = pars.tau_m

    # Define a connection index
    idx_conn = Z == 1
    n_s = np.sum(idx_conn)

    # Create indices to pack/repack the matrices
    n = Z.shape[0]
    idx_r = range(n)

    i0 = n 
    ik = i0 + n_s
    idx_g = range(i0, ik)

    i0 = ik
    ik += n_s
    idx_s = range(i0, ik)

    i0 = ik
    ik += n
    idx_gbe = range(i0, ik)

    i0 = ik
    ik += n
    idx_sbe = range(i0, ik)

    i0 = ik
    ik += n
    idx_gbi = range(i0, ik)

    i0 = ik
    ik += n
    idx_sbi = range(i0, ik)

    # Setup bias
    I_bias = np.zeros(len(idx_r))
    I_bias[np.diagonal(T) == 1] = I_e
    I_bias[np.diagonal(T) == -1] = I_i

    # Setup synapses
    I_syn = np.zeros(len(idx_r))
    I_tmp = np.zeros(len(idx_r))

    G = np.zeros_like(W)
    S = np.zeros_like(W)

    # Fixed/constant background conductances
    Gbe = np.zeros(n)
    Sbe = np.zeros(n)
    Gbi = np.zeros(n)
    Sbi = np.zeros(n)
    g0 = ((Wbe.mean() * Cbe.mean()) + (Wbi.mean() * Cbi.mean())) / n / 0.04

    # Define the function to integrate
    def layers(ys, t):
        pu.db
        """A layered gNMM model."""

        # --
        # Unpack
        R = ys[idx_r]
        G[idx_conn] = ys[idx_g]
        S[idx_conn] = ys[idx_s]

        Gbe = ys[idx_gbe]
        Sbe = ys[idx_sbe]

        Gbi = ys[idx_gbi]
        Sbi = ys[idx_sbi]

        # --
        # Step population rates
        for j in idx_r:
            I_syn[j] = np.dot(G[:, j], V[:, j]) + I_bias[j]
            
            if j in stim_index:
                I_syn[j] += stim(t)
            
            gb = abs(Gbe[j] + Gbi[j])
            sb = abs(Sbe[j] + Sbi[j])
            rt = phi(I_syn[j], g0, gb, np.sqrt(sb)) 

            ys[j] = (-R[j] + rt) / tau_m

        # --
        # Step network synapses
        R2 = np.vstack([R] * R.shape[0]).T

        # (Ugly one liners needed to conserve memory)
        ys[idx_g] = (  
            (-G / K) + (C * W * R2)
        )[idx_conn]

        ys[idx_s] = (  
            ((-2 * S) / K) + (((C * W) ** 2) * R2)
        )[idx_conn]

        # --
        # TODO Gb/Sb should be col vec not mats
        # Setup background firing for this t
        Rbe = prng.poisson(rbe, size=Gbe.shape)
        Rbi = prng.poisson(rbi, size=Gbi.shape)

        # Modulation effects whole populations not individual synapses
        for i in back_index:
            Rmode, Rmodi = _background(t, f, rbe, rbi)
            Rbe[i] = Rmode
            Rbi[i] = Rmodi

        # --
        # Step background gain control
        ys[idx_gbe] = (-Gbe / Kbe) + (Cbe * Wbe * Rbe)
        ys[idx_sbe] = ((-2 * Sbe) / Kbe) + (((Cbe * Wbe) ** 2) * Rbe)
        
        ys[idx_gbi] = (-Gbi / Kbi) + (Cbi * Wbi * Rbi)
        ys[idx_sbi] = ((-2 * Sbi) / Kbi) + (((Cbi * Wbi) ** 2) * Rbi)
        
        # If anything goes NaN we need to know NOW.
        if np.any(np.logical_not(np.isfinite(ys))):
            raise TypeError("ys is not finite at {} seconds.".format(t))
            # print("ys is not finite at {} seconds.".format(t))
            pass

        return ys
    
    idxs = {
        'R' : idx_r, 
        'G' : idx_g, 'S' : idx_s, 
        'Gbe' : idx_gbe, 'Sbe' : idx_sbe,
        'Gbi': idx_gbi, 'Sbi' : idx_sbi,
        'Z': idx_conn
    }

    return layers, idxs

    
if __name__ == "__main__":
    args = docopt(__doc__, version='tlpha')

    # Simulation parameters ----------------------------------------- 
    print(">>> Building the model.")
    save_path = args['NAME']
    execfile(args['PARS_FILE'])  # returns 'pars'

    # Override defaults
    pars.I_e = 1e-3
    pars.I_i = 0.7e-3

    # Pertub defaults
    # pars = perturb_params(pars, 'W', sd='Wstd', prng=prng)
    
    # W can't be less then 0 (or a really small number, in this case)
    # pars.W[pars.W < 1e-20] = 0

    # Set up time
    tmax = float(args['-t'])
    step = float(args['--step']) # 0.1  # Don't go larger than 0.2
    n_step = int(np.ceil(tmax / step))

    f = float(args['-f'])

    if args['--r_back'] is not None:
        pars.Rb = np.asarray([float(a) for a in args['--r_back'].split(',')])
    if args['--w_back'] is not None:
        pars.Wb = np.asarray([float(a) for a in args['--w_back'].split(',')])

    # Simulation input ---------------------------------------------- 
    I_stim = pars.I_e * 0.75
    scale = 0.001 * I_stim

    dt = 1e-4

    # Create stimulus
    stim_seed = int(args['-s'])
    Istim = create_I(tmax, I_stim, scale, dt=dt, seed=stim_seed)

    # Define systems to integrate
    layers, idxs = create_layers(f, Istim, pars)
    g = partial(ornstein_uhlenbeck, sigma=0.01, loc=pars.stim_i) 

    # Init ys
    # A dirty way to get max
    max_n = max(max([v[:] for k, v in idxs.items() if k != 'Z']))  

    ys0 = np.zeros(max_n + 1)
    idx_conn = idxs['Z']

    ys0[idxs['R']] = pars.R0
    
    ys0[idxs['G']] = (pars.W / 2)[idx_conn]
    ys0[idxs['S']] = ((pars.W / 2) ** 2)[idx_conn]
    
    ys0[idxs['Gbe']] = pars.Wbe / 2
    ys0[idxs['Sbe']] = (pars.Wbe ** 2) / 2
    ys0[idxs['Gbi']] = pars.Wbi / 2
    ys0[idxs['Sbi']] = (pars.Wbi ** 2) / 2

    # Run -----------------------------------------------------------
    print(">>> Running the model.")
    ys_ts = ys0
    t0 = 0.0
    ts = t0 + step
    for k in progressbar(range(n_step)):
        times = np.linspace(t0, ts, int(step / dt))

        ys = itoint(layers, g, ys_ts, times)

        np.savez(
            save_path + '_k{}'.format(k), 
            ys=ys, idxs=idxs, times=times
        )

        t0 = deepcopy(ts)
        ts += step
        ys_ts = deepcopy(ys[-1, :])
        del ys

    # Save params
    np.savez(save_path + "_run_pars", 
            tmax=tmax, 
            dt=dt,
            step=step,
            n_step=n_step,
            f=f, 
            stim_seed=stim_seed,
            scale=scale, 
            times=times, 
            max_n=max_n, 
            ys0=ys0,
            stim=np.asarray([
                Istim(t) for t in np.linspace(0, tmax, int(tmax/dt))]))

    np.savez(save_path + "_pars", **pars.__dict__)
