"""Usage: bluemass5.py NAME PARS_FILE
    [-t T] 
    [--seed STIM_SEED]
    [--dt DT]
    [--r_stim=RATE] 
    [--verbose]

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
        --verbose               display progress

"""
from __future__ import division
import os
import sys
import csv

import warnings
warnings.simplefilter("always")

from docopt import docopt
from copy import deepcopy
from sdeint import itoint
import numpy as np
from numpy import random
from scipy.interpolate import interp1d

from fakespikes.rates import bursts, constant, stim
from fakespikes.util import create_times
from fakespikes.neurons import Spikes

from pacological.util import create_stim_I, create_constant_I
from pacological.util import ornstein_uhlenbeck
from pacological.pars import perturb_params
from pacological.pars import BMparams
from pacological.fi import lif
from pacological.fi import N as normal

from functools import partial
from pykdf.kdf import save_kdf, load_kdf


def create_background(times, f, r_e, r_i, min_rate=30, n_bursts=None, 
        prng=None):

    if prng is None:
        prng = random.RandomState()

    # The background drives rates
    if f > 0:
        rates_e = bursts(times, float(r_e), f, n_bursts, min_a=min_rate)
        rates_i = bursts(times, float(r_i), f, n_bursts, min_a=min_rate)
    else: 
        rates_e = constant(times, r_e)
        rates_i = constant(times, r_i)

    def time_index(t):
        return (np.abs(times - t)).argmin()

    # Define a function to sample rates at t
    def background(t):
        i = time_index(t)

        r_e_0 = rates_e[i]
        r_i_0 = rates_i[i]

        r_e_t = prng.poisson(r_e_0, 1)[0]
        r_i_t = prng.poisson(r_i_0, 1)[0]

        return r_e_t, r_i_t, r_e_0, r_i_0, prng

    return background, prng


def create_layers(times, stim, pars, seed=42, verbose=True, debug=False):
    global prng
    prng = random.RandomState(seed)

    # if verbose:
        # print(">>> Creating layers.")

    # Unpack pars
    n_pop = pars.n_pop
    names = pars.names
    I_max = pars.I_max
    background_res = pars.background_res
    t_back = pars.t_back

    # Network
    Z = pars.Z  # Input
    C = pars.C  # Connection number
    Cstd = pars.Cstd
    W = pars.W  # Weights
    T = pars.T  # Connection type E:1, I:-1
    V = pars.V  # Eff. voltage drive at synapses
    K = pars.K  # Taus
    tau_m = pars.tau_m
    I_bias = pars.I_bias

    # Input
    Zi = pars.Zi  # Input
    Ci = pars.Ci  # Total synapse number
    Cistd = pars.Cistd 
    Wi = pars.Wi  # Weights
    Ti = pars.Ti  # Connection type E:1, I:-1
    Ki = pars.Ki  # Taus
    Vi = pars.Vi  # Input
    Id = np.identity(Zi.shape[0])  # And identity matrix

    # Define a connection index
    idx_conn = Z == 1
    idx_conn_in = Zi == 1

    n_s = np.sum(idx_conn)
    n_in = np.sum(idx_conn_in)

    # Define idx into ys
    idx_in = range(n_in)

    i0 = n_in
    ik = i0 + n_in
    idx_in_sigma = range(i0, ik)

    i0 = ik
    ik += n_s
    idx_h = range(i0, ik)

    i0 = ik
    ik += n_s
    idx_h_sigma = range(i0, ik)

    i0 = ik
    ik += n_pop
    idx_r = range(i0, ik)

    # Setup synapses
    H = np.zeros_like(W)
    Hsigma = np.zeros_like(W)

    IN = np.zeros_like(Wi)
    INsigma = np.zeros_like(Wi)

    # Setup valid FI window
    I_fis = np.linspace(0, I_max, 100)

    # Create background fns
    backgrounds = []
    for j in range(n_pop):
        b = pars.backs[j][1]

        f = b['f']
        r_e = b['r_e']
        r_i = b['r_i']
        min_r = b['min_r']
        n_bursts = b['n_bursts']

        b, prng = create_background(times, f, r_e, r_i, 
                min_rate=min_r, n_bursts=n_bursts, prng=prng)

        backgrounds.append(b)

    def layers(ys, t):
        """A layered gNMM model."""
        global prng

        # Unpack ys
        IN[idx_conn_in] = ys[idx_in]
        INsigma[idx_conn_in] = ys[idx_in_sigma]

        H[idx_conn] = ys[idx_h]
        Hsigma[idx_conn] = ys[idx_h_sigma]

        G = H * C
        G = G.T

        Gi = IN * Ci

        Cprime = Cstd  # approximatly
        S = (Cstd * G**2) + (Cprime * Hsigma)  # Use S**2 instead?

        Ciprime = Cistd
        Si = (Cistd * Gi**2) + (Ciprime * INsigma)  # Use S**2 instead?

        R = np.zeros(n_pop)
        Rsigma = np.zeros(n_pop)

        for j in range(n_pop):
            dy = np.zeros_like(ys)

            # I(t)
            I = np.dot(G[:, j], V[:, j]) + np.dot(Gi[j], Vi[j]) + I_bias[j]
            if I > I_max:
                warnings.warn("Clipping I to {} in {} at {} seconds.".format(
                    I_max, names[j], t), RuntimeWarning)
                I = I_max
            # I = np.min([I, I_max])
            I = np.max([0, I])

            # I_sigma(t)
            # Isigma = np.abs(I / 10)   # TODO; correct math fron Zandt
            Isigma = 5e-3
            # Isigma = np.dot(S[:, j], V[:, j]**2) + np.dot(Si[j], Vi[j]**2)

            # print "--"
            # print j
            # print I * 1000, Isigma * 1000

            # - FI(t)
            # - Params
            b = pars.backs[j][1]
            w_e = b['w_e']
            w_i = b['w_i']
            tau_e = b['tau_e']
            tau_i = b['tau_e']

            # Background params
            # Stocastic
            rbe, rbi, _, _, prng = backgrounds[j](t)

            # Deterministic
            _, _, rbe, rbi, prng = backgrounds[j](t)
            
            # Round the poisson rate output so `lif` is not 
            # called with EVERY iteration.
            rbe = np.round(rbe, background_res)
            rbi = np.round(rbi, background_res)

            # Calculate single neuron fi curve
            fi = lif(t_back,
                     I_fis,
                     0,  # Osc is taken care of by backgrounds
                     rbe,
                     rbi,
                     w_e,
                     w_i,
                     tau_e=tau_e,
                     tau_i=tau_i,
                     min_rate=0,  # min_rate taken care by backgrounds
                     verbose=verbose)
            
            # Calculate network variance, g(t)
            g = normal(I_fis, I, Isigma)
            # g /= g.max()

            # Estimate network firing rate, r_t(t)
            # (this is PHI, the network non-linearity)
            if Isigma < 1e-10:
                R[j] = interp1d(I_fis, fi)(I)
                Rsigma[j] = 0.0
            else:
                R[j] = np.trapz(fi * g, I_fis)
                Rsigma[j] = np.trapz((fi - R[j])**2 * g, I_fis)

            if debug:
                with open('I.csv', 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow([I, ])
                with open('Isigma.csv', 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow([Isigma, ])
                with open('fi.csv', 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow(fi)
                with open('g.csv', 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow(g)
        # H
        dy[idx_h] = ((-H / K) + (W * C * R))[idx_conn].flatten()

        # Hsigma
        dy[idx_h_sigma] = ((-2 * (Hsigma / K)) + (
            (W * C)**2 * Rsigma))[idx_conn].flatten()

        # Stim
        R_stim = prng.poisson(stim(t), 1)[0] * Zi

        # IN
        dy[idx_in] = ((-IN / Ki) + (Wi * Ci * R_stim))[idx_conn_in].flatten()

        # INsigma
        dy[idx_in_sigma] = (-2 * (INsigma / Ki) + (
            (Wi * Ci)**2 * R_stim))[idx_conn_in].flatten()

        # Rate
        dy[idx_r] = (-ys[idx_r] / tau_m) + R

        # If anything goes numerically funky, die NOW.
        if np.any(np.logical_not(np.isfinite(dy))):
            raise TypeError("y is not finite at {} seconds.".format(t))

        return dy

    idxs = {'IN': idx_in,
            'INsigma': idx_in_sigma,
            'H': idx_h,
            'Hsigma': idx_h_sigma,
            'R': idx_r,
            'Z': idx_conn,
            'Zi': idx_conn_in}

    return layers, idxs


def create_ys0(pars, idxs, frac=0.1):
    """Init the intial value, ys0"""

    exclude = ['Z', 'Zi']
    max_n = 0
    for k, idx in idxs.items():
        if k not in exclude:
            max_n += len(idx)

    ys0 = np.zeros(max_n + 1)

    idx_conn = idxs['Z']
    idx_conn_in = idxs['Zi']

    # ys0[idxs['R']] = [p[1]['r_0'] for p in pars.pops]
    ys0[idxs['H']] = (pars.W * frac)[idx_conn]
    ys0[idxs['Hsigma']] = ((pars.W * frac)**2)[idx_conn]

    ys0[idxs['IN']] = (pars.Wi * frac)[idx_conn_in]
    ys0[idxs['INsigma']] = ((pars.Wi * frac)**2)[idx_conn_in]

    return ys0


if __name__ == "__main__":
    args = docopt(__doc__, version='alpha')

    # Simulation parameters ------------------------------------
    # print(">>> Starting the model.")
    seed = int(args['--seed'])
    save_path = args['NAME']

    verbose = False
    if args['--verbose']:
        verbose = True

    # Load parameters
    execfile(args['PARS_FILE'])  # returns 'pars'
    pars = BMparams(pops, conns, backs, inputs, sigma=0, background_res=0,
            I_max=50e-3)

    # Setup time
    t = float(args['-t'])
    dt = float(args['--dt'])
    times = create_times(t, dt)

    # Setup stimulus
    d = float(args['--r_stim'])
    scale = .01 * d
    stim = create_stim_I(times, d, scale,  seed=seed)
    # stim = create_constant_I(times, d, seed=seed)

    # Setup the network
    gn = partial(ornstein_uhlenbeck, sigma=0.01, loc=[])
    fn, idxs = create_layers(times, stim, pars, seed=seed, verbose=False, debug=True)

    # Init the intial values
    ys0 = create_ys0(pars, idxs, frac=0.1)

    # Integrate!
    # print(">>> Running the model.")
    ys = itoint(fn, gn, ys0, times)

    save_kdf("{}".format(save_path),
         ys=ys,
         ys0=ys0,
         idx_R=idxs['R'],
         idx_IN=idxs['IN'],
         idx_INsigma=idxs['INsigma'],
         idx_H=idxs['H'],
         idx_Hsigma=idxs['Hsigma'],
         times=times,
         names=pars.names,
         t=t,
         dt=dt,
         d=d,
         stim=np.asarray([stim(t) for t in times]),
         seed=seed)

