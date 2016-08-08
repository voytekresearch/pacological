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


def create_layers(f, stim, pars):
    # TODO def mod_index in pars
    
    # Unpack pars for readability
    stim_index = pars.stim_i
    back_index = pars.back_i

    Z = pars.Z
    C = pars.C
    W = pars.W
    T = pars.T
    V = pars.V
    K = pars.K

    phi_back = pars.fi_back
    phi_mod = pars.fi_mod
    
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
    
    # Setup bias
    I_bias = np.zeros(len(idx_r))
    I_bias[np.diagonal(T) == 1] = I_e
    I_bias[np.diagonal(T) == -1] = I_i

    # Setup synapses
    I_syn = np.zeros(len(idx_r))
    G = np.zeros_like(W)
    S = np.zeros_like(W)
    
    def layers(ys, t):
        """A layered gNMM model."""

        # --
        # Unpack
        R = ys[idx_r]
        G[idx_conn] = ys[idx_g]
        S[idx_conn] = ys[idx_s]
        
        # --
        # Step population rates
        for j in idx_r:
            I_syn[j] = np.dot(G[:, j], V[:, j]) + I_bias[j]
            
            if j in stim_index:
                I_syn[j] += stim(t)
            
            phi = phi_back
            if j in mod_index:
                phi = phi_mod
                    
            gb = abs(Gbe[j] + Gbi[j])
            sb = abs(Sbe[j] + Sbi[j])
            rt = phi(I_syn[j], g0, gb, np.sqrt(sb)) 

            ys[j] = (-R[j] + rt) / tau_m
        
        # If anything goes NaN we need to know NOW.
        if np.any(np.logical_not(np.isfinite(ys))):
            raise TypeError("ys is not finite at {} seconds.".format(t))
            # print("ys is not finite at {} seconds.".format(t))
            pass
            
        return ys
    
    idxs = {
        'R' : idx_r, 
        'G' : idx_g, 'S' : idx_s, 
        'Z' : idx_conn
    }
    
    return layers, idxs