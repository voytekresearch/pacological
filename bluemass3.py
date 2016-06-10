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
from copy import deepcopy

import numpy as np
from functools import partial
from numpy import random
from scipy.integrate import odeint, ode
from fakespikes.rates import stim
from pacological.util import create_I, ornstein_uhlenbeck, progressbar
from pacological.pars import perturb_params
from pacological.fi import lif
from pacological.fi import N as normal
from numba import autojit


def background(t, f, rbe, rbi, min_rate=12, prng=None):
    if prng is None:
        prng = random.RandomState()

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

    rbe_t = prng.poisson(rbe_t, 1) + min_rate
    rbi_t = prng.poisson(rbi_t, 1) + min_rate

    return rbe_t[0], rbi_t[0], prng


def create_layers(stim, pars, seed=42, verbose=True):
    global prng
    prng = random.RandomState(seed)

    if verbose:
        print(">>> Creating layers.")

    I_max = pars.I_max
    background_res = pars.background_res
    t_back = pars.t_back

    # Unpack pars for readability
    # Network
    Z = pars.Z  # Input
    C = pars.C  # Total synapse number
    W = pars.W  # Weights
    T = pars.T  # Connection type E:1, I:-1
    V = pars.V  # Eff. voltage drive at synapses
    K = pars.K  # Taus
    tau_m = pars.tau_m

    # Input
    Zi = pars.Zi  # Input
    Ci = pars.Ci  # Total synapse number
    Wi = pars.Wi  # Weights
    Ti = pars.Ti  # Connection type E:1, I:-1
    Ki = pars.Ki  # Taus
    Id = np.identity(Zi.shape[0])  # And identity matrix

    I_e = pars.I_e
    I_i = pars.I_i

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
    G = np.zeros_like(W)
    S = np.zeros_like(W)

    # Setup valid FI window
    I_fis = np.linspace(0, I_max, 500)

    def layers(t, ys):
        """A layered gNMM model."""
        global prng

        # unpack ys
        R = ys[idx_r]
        G[idx_conn] = ys[idx_g]  # g
        S[idx_conn] = ys[idx_s]  # sigma_{g}

        # the step
        dh = np.zeros_like(ys)

        if verbose:
            print ">>> t: {}".format(t)

        for j in idx_r:
            if verbose:
                print ">>> Pop: {}".format(pars.names[j])

                # I(t)
            I = np.dot(G[:, j], V[:, j])

            if I > I_max:
                raise ValueError("I became to large.")

            # sigma_{I}(t)
            S_I = np.dot(S[:, j], V[:, j]**2)

            # f_back(t) - background firing rate
            b = pars.backs[j][1]
            f = b['f']
            r_e = b['r_e']
            r_i = b['r_i']
            w_e = b['w_e']
            w_i = b['w_i']
            tau_e = b['tau_e']
            tau_i = b['tau_e']

            rbe, rbi, prng = background(t, f, r_e, r_i, prng=prng)

            # Round the poisson rate output so `lif` is not 
            # called with EVERY iteration.
            rbe = np.round(rbe, background_res)
            rbi = np.round(rbi, background_res)

            # Use background at t to define a fi(t).
            fi = lif(t_back,
                     I_fis,
                     f,
                     rbe,
                     rbi,
                     w_e,
                     w_i,
                     tau_e=tau_e,
                     tau_i=tau_i,
                     verbose=verbose)

            # Calculate network variance, g(t)
            g = normal(I_fis, I, S_I)

            # Estimate network firing rate, r_t(t)
            # (this is PHI, the network non-linearity)
            rt = np.trapz(fi, g)

            # Network noise(t)
            rn = prng.poisson(pars.sigma, 1)[0]

            # Update R(t)
            dh[j] = (-R[j] / tau_m) + rt + rn

            if verbose:
                print("I : {}, rt : {}, rn : {}".format(I, rt, rn))

        # stim(t)
        rs = prng.poisson(stim(t), 1)[0]
        if verbose:
            print("rs : {}".format(rs))

        # dg/dt
        # TODO think thourgh the orientation of R * W* C math in relation to G
        Gnet = (W * C * R)
        Gi = Id * (Wi * Ci * (Zi * rs))
        dh[idx_g] = (-(G / K) + Gnet + Gi)[idx_conn].flatten()

        # ds/dt
        Gnet = (W**2 * C * R)
        Gi = Id * (Wi**2 * Ci * (Zi * rs))
        dh[idx_s] = (-2 * (G / K) + Gnet + Gi)[idx_conn].flatten()

        # If anything goes NaN we need to know NOW.
        if np.any(np.logical_not(np.isfinite(dh))):
            raise TypeError("y is not finite at {} seconds.".format(t))

        return dh

    if verbose:
        print(">>> Done.\n>>> Running the model....")

    idxs = {'R': idx_r, 'G': idx_g, 'S': idx_s, 'Z': idx_conn}

    return layers, idxs


if __name__ == "__main__":
    seed = 42

    pars_file = sys.argv[1]
    print(">>> Running {}".format(pars_file))

    # Time
    t = .3
    dt = 1 / 1000.  # resolution, 1 ms
    # n_step = int(np.ceil(t / dt))
    # times = np.linspace(0, t, n_step)

    # Get model parameters
    pars = None
    execfile(pars_file)  # adds data to pars

    # Create model input
    d = 100  # drive rate (want 0-30)
    scale = .01 * d
    stim = create_I(t, d, scale, dt=dt, seed=seed)

    # Init network
    f, idxs = create_layers(stim, pars, seed=seed)

    # Init ys0
    max_n = max(max([v[:] for k, v in idxs.items() if k != 'Z']))
    ys0 = np.zeros(max_n + 1)
    idx_conn = idxs['Z']

    ys0[idxs['R']] = pars.R0
    ys0[idxs['G']] = (pars.W / 10)[idx_conn]
    ys0[idxs['S']] = ((pars.W / 10)**2)[idx_conn]

    # Integrate!
    # ys = odeint(f, ys0, times)
    times = []
    ys = []
    r = ode(f).set_integrator('vode', method='bdf')
    r.set_initial_value(ys0, 0.0)
    while r.successful() and r.t < t:
        r.integrate(r.t + dt)
        ys.append(r.y)
        times.append(r.t)

    np.savez("bluemass3",
             ys=ys,
             idxs=idxs,
             ys0=ys0,
             times=times,
             stim=np.asarray([stim(t) for t in times]),
             t=t,
             dt=dt,
             seed=seed)
