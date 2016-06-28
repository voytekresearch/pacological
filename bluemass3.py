"""Usage: bluemass3.py NAME PARS_FILE
    [-t T] 
    [--seed STIM_SEED]
    [--dt DT]
    [--r_stim=RATE] [--verbose] [--debug]
    
Simulate the Blue Brain using gNMMs.

    Arguments:
        NAME        name (and path) of the results files
        PARS_FILE   parameters file (a BMparams() instance)

    Options:
        -h --help               show this screen
        -t T                    simultation run time [default: 1.0] 
        --seed STIM_SEED        seed for creating the stimulus [default: 1]
        --dt DT                 time resolution [default: 1e-3]
        --r_stim=RATE           stimulus firing rate (Hz) [default: 10]
        --debug                 write data to disk for all t
        --verbose               print progress

"""
from __future__ import division
from docopt import docopt
from copy import deepcopy
from sdeint import itoint
from scipy.interpolate import interp1d

import numpy as np
from numpy import random
from fakespikes.rates import stim
from pacological.util import create_I, ornstein_uhlenbeck
from pacological.fi import lif
from pacological.fi import N as normal
from pacological.pars import BMparams
from functools import partial

import csv


def background(t, f, rbe, rbi, min_rate=12, prng=None):
    """Poissonic background activity"""

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


def create_layers(stim, pars, seed=42, verbose=True, debug=False):
    """Create the cortical model"""

    global prng
    prng = random.RandomState(seed)

    if verbose:
        print(">>> Creating layers.")

    # Unpack pars for readability
    # Limits for single unit fi
    I_max = pars.I_max
    background_res = pars.background_res
    t_back = pars.t_back
    tau_m = pars.tau_m

    # Network
    Z = pars.Z  # Input
    C = pars.C  # Connectivty
    C_var = pars.C_var
    W = pars.W  # Weights
    V = pars.V  # Eff. voltage drive at synapses
    K = pars.K  # Taus

    # Input
    Zi = pars.Zi  # Input
    Wi = pars.Wi  # Weights
    Ki = pars.Ki  # Taus

    Id = np.identity(Zi.shape[0])  # identity for broadcasting fun

    # Define a connection index
    idx_conn = Z == 1
    n_s = np.sum(idx_conn)

    # Create indices to pack/repack the matrices
    n = Z.shape[0]
    idx_r = range(n)

    i0 = n
    ik = i0 + n_s
    idx_h = range(i0, ik)

    i0 = ik
    ik += n_s
    idx_hs = range(i0, ik)

    # Setup synapses
    H = np.zeros_like(W)
    H_var = np.zeros_like(W)

    # Setup valid FI window
    I_fis = np.linspace(0, I_max, 500)

    def layers(ys, t):
        """A layered gNMM model."""
        global prng

        # unpack ys
        R = ys[idx_r]
        H[idx_conn] = ys[idx_h]  # TODO
        H_var[idx_conn] = ys[idx_hs]  # TODO

        # the step
        dh = np.zeros_like(ys)

        if verbose:
            print "---\n>>> t: {}".format(t)

        for j in idx_r:
            if verbose:
                print ">>> Pop: {}".format(pars.names[j])

            # ---------------------------------------------
            # I(t) - network currents
            g = H[:, j] * R[j]
            G = C[:, j] * g
            I = np.dot(G, V[:, j])
            I = max(I, 0)  # Rectify

            Np = C_var[:, j]  # approx valid for small p; revist?
            S = (C_var[:, j] * g**2) + (Np * H_var[:, j] * R[j])
            I_var = np.dot(S, V[:, j]**2)

            if I > I_max:
                print("!!! I became to large. Reset to I_max !!!")
                I = I_max

            # ---------------------------------------------
            # f_back(t) - background firing rate
            b = pars.backs[j][1]
            f = b['f']
            r_e = b['r_e']
            r_i = b['r_i']
            w_e = b['w_e']
            w_i = b['w_i']
            tau_e = b['tau_e']
            tau_i = b['tau_i']

            # stocastic
            # rbe, rbi, _, _, prng = background(t, f, r_e, r_i, prng=prng)

            # Deterministic
            _, _, rbe, rbi, prng = background(t, f, r_e, r_i, prng=prng)

            # Round the poisson rate output so `lif` can be cached
            rbe = np.round(rbe, background_res)
            rbi = np.round(rbi, background_res)

            # Use background at t to define a fi(t).
            if verbose:
                print("rbe/i : {}/{}".format(rbe, rbi))
            fi = lif(t_back,
                     I_fis,
                     rbe,
                     rbi,
                     w_e,
                     w_i,
                     tau_e=tau_e,
                     tau_i=tau_i,
                     verbose=verbose)

            # Est current distribution
            Idist = normal(I_fis, I, I_var)

            # Estimate network firing rate, r_t(t)
            if I_var < 1e-10:
                rt = interp1d(I_fis, fi)(I)
                st = 0.0

                if verbose:
                    print("!!! Interpolating rate. !!!")
            else:
                rt = np.trapz(Idist * fi, I_fis)
                st = np.trapz((fi - rt)**2 * Idist, I_fis)

            # Network noise(t)
            rn = prng.poisson(pars.sigma, 1)[0]

            # Update R(t)
            dh[idx_r[j]] = (-R[j] + rt + rn) / tau_m

            if verbose:
                print("I : {}, rt : {}, rn : {}".format(I * 1000, rt, rn))
            if debug:
                with open('I.csv', 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow([I, ])
                with open('I_var.csv', 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow([I_var, ])
                with open('S.csv', 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow(S)
                with open('G.csv', 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow(G)
                with open('fi.csv', 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow(fi)
                with open('Idist.csv', 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow(Idist)
                with open('rt.csv', 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow([rt, ])
                with open('st.csv', 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow([st, ])

        # stim(t)
        rs = prng.poisson(stim(t), 1)[0]
        if verbose:
            print("rs : {}".format(rs))

        # dg/dt
        # TODO - input is going through the diag, which
        # means the Ki term is never used. Need to change
        # synapses to use Ki, or remove Ki from pars.
        # The former seems preferable.
        Hnet = W * R[:, None]
        Hi = Id * (Wi * (Zi * rs))
        dh[idx_h] = (-(H / K) + Hnet + Hi)[idx_conn].flatten()

        # ds/dt
        Hnet = W**2 * R[:, None]
        Hi = Id * (Wi**2 * (Zi * rs))
        dh[idx_hs] = ((-2 * (H_var / K)) + Hnet + Hi)[idx_conn].flatten()

        # import ipdb
        # ipdb.set_trace()

        # If anything goes NaN we need to know NOW.
        if np.any(np.logical_not(np.isfinite(dh))):
            raise TypeError("y is not finite at {} seconds.".format(t))

        return dh

    if verbose:
        print(">>> Done.\n>>> Running the model....")

    idxs = {'R': idx_r, 'H': idx_h, 'H_var': idx_hs, 'Z': idx_conn}

    return layers, idxs


def create_ys0(pars, idxs, frac=0.1, frac_var=0.1):
    """Init the intial value, ys0"""

    max_n = max(max([v[:] for k, v in idxs.items() if k != 'Z']))
    ys0 = np.zeros(max_n + 1)
    idx_conn = idxs['Z']

    ys0[idxs['R']] = [p[1]['r_0'] for p in pars.pops]
    ys0[idxs['H']] = (pars.W * frac)[idx_conn]
    ys0[idxs['H_var']] = (pars.W * frac)[idx_conn]

    return ys0


if __name__ == "__main__":
    args = docopt(__doc__, version='alpha')

    verbose = False
    if args['--verbose']:
        verbose = True

    debug = False
    if args['--debug']:
        debug = True

    # Simulation parameters ------------------------------------
    if verbose:
        print(">>> Building the model.")

    seed = int(args['--seed'])
    save_path = args['NAME']

    # Load parameters
    pops, inputs, backs, conns = None, None, None, None
    execfile(args['PARS_FILE'])  # returns into the above

    pars = BMparams(pops, conns, backs, inputs, sigma=1, background_res=0)

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
    fn, idxs = create_layers(stim,
                             pars,
                             seed=seed,
                             debug=debug,
                             verbose=verbose)

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
             d=d,
             seed=seed)
