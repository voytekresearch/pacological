"""Usage: bluemass3.py NAME PARS_FILE
    [-t T] 
    [--seed STIM_SEED]
    [--dt DT]
    [--r_stim=RATE] 
    
Simulate the Blue Brain using gNMMs.

    Arguments:
        NAME        name (and path) of the results files
        PARS_FILE   parameters file (a BMparams() instance)

    Options:
        -h --help               show this screen
        -t T                    simultation run time [default: 1.0] 
        --seed STIM_SEED        seed for creating the stimulus [default: 1]
        --dt DT                 time resolution [default: 1e-3]
        --r_stim=RATE           stimulus firing rate firing rate (Hz) [default: 10]

"""
from __future__ import division
from docopt import docopt
import os, sys
from copy import deepcopy
from sdeint import itoint

import numpy as np
from numpy import random
from scipy.integrate import odeint, ode
from fakespikes.rates import stim
from pacological.util import create_I, ornstein_uhlenbeck, progressbar
from pacological.pars import perturb_params
from pacological.fi import lif
from pacological.fi import N as normal


def background(t, f, rbe, rbi, min_rate=12, prng=None):
    if prng is None:
        prng = random.RandomState()

    if f > 0:
        # oscillation
        rbe0 = rbe * np.cos(2 * np.pi * f * t)
        rbi0 = rbi * np.cos(2 * np.pi * f * t)
    else:
        # fixed rate
        rbe0 = deepcopy(rbe)
        rbi0 = deepcopy(rbi)

    if rbe0 < min_rate:
        rbe0 = min_rate
    if rbi0 < min_rate:
        rbi0 = min_rate

    rbe_t = prng.poisson(rbe0, 1)
    rbi_t = prng.poisson(rbi0, 1)

    return rbe_t[0], rbi_t[0], rbe0, rbi0, prng


def create_layers(stim, pars, seed=42, verbose=True):
    global prng
    prng = random.RandomState(seed)

    if verbose:
        print(">>> Creating layers.")

    # Unpack pars for readability
    # Limits for single unit fi
    I_max = pars.I_max
    background_res = pars.background_res
    t_back = pars.t_back

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

    # Setup synapses
    G = np.zeros_like(W)
    S = np.zeros_like(W)

    # Setup valid FI window
    I_fis = np.linspace(0, I_max, 500)

    def layers(ys, t):
        """A layered gNMM model."""
        global prng

        # unpack ys
        R = ys[idx_r]
        G[idx_conn] = ys[idx_g]  # g
        S[idx_conn] = ys[idx_s]  # sigma_{g}

        # the step
        dh = np.zeros_like(ys)

        if verbose:
            print "---\n>>> t: {}".format(t)

        for j in idx_r:
            if verbose:
                print ">>> Pop: {}".format(pars.names[j])

                # I(t)
            I = np.dot(G[:, j], V[:, j])

            if I > (2 * I_max):
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

            # Stocastic
            # rbe, rbi, _, _, prng = background(t, f, r_e, r_i, prng=prng)
            # Deterministic
            _, _, rbe, rbi, prng = background(t, f, r_e, r_i, prng=prng)

            # Round the poisson rate output so `lif` is not 
            # called with EVERY iteration.
            rbe = np.round(rbe, background_res)
            rbi = np.round(rbi, background_res)

            # Use background at t to define a fi(t).
            if verbose:
                print("rbe/i : {}/{}".format(rbe, rbi))
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
        Gnet = (W * C * R)
        Gi = Id * (Wi * Ci * (Zi * rs))
        dg = (-(G / K) + Gnet + Gi)[idx_conn].flatten()
        dh[idx_g] = dg

        # ds/dt
        Gnet = ((W * C)**2 * R)
        Gi = Id * ((Wi * Ci)**2 * (Zi * rs))
        ds = (-2 * (S / K) + Gnet + Gi)[idx_conn].flatten()
        dh[idx_s] = ds

        # If anything goes NaN we need to know NOW.
        if np.any(np.logical_not(np.isfinite(dh))):
            raise TypeError("y is not finite at {} seconds.".format(t))

        return dh

    if verbose:
        print(">>> Done.\n>>> Running the model....")

    idxs = {'R': idx_r, 'G': idx_g, 'S': idx_s, 'Z': idx_conn}

    return layers, idxs


def create_ys0(pars, idxs, frac=0.1):
    """Init the intial value, ys0"""

    max_n = max(max([v[:] for k, v in idxs.items() if k != 'Z']))
    ys0 = np.zeros(max_n + 1)
    idx_conn = idxs['Z']

    ys0[idxs['R']] = [p[1]['r_0'] for p in pars.pops]
    ys0[idxs['G']] = (pars.W * frac)[idx_conn]
    ys0[idxs['S']] = ((pars.W * frac)**2)[idx_conn]

    return ys0


if __name__ == "__main__":
    args = docopt(__doc__, version='alpha')

    # Simulation parameters ------------------------------------
    print(">>> Building the model.")
    seed = int(args['--seed'])
    save_path = args['NAME']

    # Load parameters
    pars = None
    execfile(args['PARS_FILE'])  # returns 'pars'

    # Setup time
    t = float(args['-t'])
    dt = float(args['--dt'])
    n_step = int(np.ceil(t / dt))
    times = np.linspace(0, t, n_step)

    # Setup stimulus
    d = float(args['--r_stim'])
    scale = .01 * d
    stim = create_I(t, d, scale, dt=dt, seed=seed)

    # Setup the network
    gn = partial(ornstein_uhlenbeck, sigma=0.01, loc=[])
    fn, idxs = create_layers(stim, pars, seed=seed)

    # Init the intial values
    ys0 = create_ys0(pars, idxs, frac=0.1)

    # Integrate!
    ys = itoint(fn, gn, ys0, times)

    # TODO save into h/kdf instead...
    np.savez("{}".format(save_path),
             ys=ys,
             idxs=idxs,
             ys0=ys0,
             times=times,
             stim=np.asarray([stim(t) for t in times]),
             t=t,
             dt=dt,
             seed=seed)
