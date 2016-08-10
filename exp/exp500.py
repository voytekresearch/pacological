"""Usage: exp500.py NAME PARS_FILE 
    [-t T] 
    [-n N]
    [--sigma SIGMA]
    [--seed STIM_SEED]
    [--dt DT]

Simulate the Blue Brain using gNMMs.

    Arguments:
        NAME        name (and path) of the results files
        PARS_FILE   parameters file (a BMparams() instance)

    Options:
        -h --help               show this screen
        -t T                    simultation run time [default: 1.0] 
        -n N                    number of trials [default: 100]
        --sigma SIGMA           Noise added to E [default: 0.01]
        --seed STIM_SEED        seed for creating the stimulus [default: 1]
        --dt DT                 time resolution [default: 1e-3]

"""

"""Explore the bluemass5 parameter space"""
from __future__ import division
import os
import sys

import pyentropy as en

from copy import deepcopy
from functools import partial
from sdeint import itoint
from itertools import product

from fakespikes.util import create_times
from pacological.pars import BMparams
from pacological.bluemass5 import create_layers, create_ys0
from pacological.util import create_stim_I, ornstein_uhlenbeck
from joblib import Parallel, delayed
from pykdf.kdf import save_kdf, load_kdf

def exp(save_path, i, seed, sigma):

    # Reinit stim
    d = 10  #Hz
    scale = .01 * d
    stim = create_stim_I(times, d, scale, seed=seed)
    # stim = create_constant_I(times, d, seed=seed)

    # Network
    pars = BMparams(pops, conns, backs, inputs, sigma=sigma, background_res=0)

    # System
    gn = partial(ornstein_uhlenbeck, sigma=sigma, loc=[0])
    fn, idxs = create_layers(times,
                             stim,
                             pars,
                             seed=seed,
                             verbose=False,
                             debug=False)

    # Init the intial values
    ys0 = create_ys0(pars, idxs, frac=0.1)

    # Integrate!
    ys = itoint(fn, gn, ys0, times)

    # MI
    x_in = np.asarray([stim(t) for t in times])

    to_calc = ('HX', 'HY', 'HXY')
    m = 8  # Per Ince's advice
    q_in, _, _ = en.quantise(x_in, m)
    q_e, _, _ = en.quantise(ys[:, idx_H[0]], m)
    info = en.DiscreteSystem(
        q_in,
        (1, m),
        q_e,
        (1, m)
    )
    info.calculate_entropies(method='pt', calc=to_calc)
    mutual_info = info.I()

    # Save results
    save_kdf("{}".format(os.path.join(save_path, str(i))),
         ys=ys,
         ys0=ys0,
         idx_R=idxs['R'],
         idx_IN=idxs['IN'],
         idx_INsigma=idxs['INsigma'],
         idx_H=idxs['H'],
         idx_Hsigma=idxs['Hsigma'],
         times=times,
         mutual_info=mutual_info,
         sigma=sigma,
         stim=x_in,
         seed=seed
    )



if __name__ == "__main__":
    args = docopt(__doc__, version='alpha')

    # Random seeding
    seed = int(args['--seed'])

    save_path = args['NAME']

    # Returns conns, inputs, backs, pops
    execfile(args['PARS_FILE']) 

    # Noise level
    sigma = float(args['--sigma'])

    # Setup time
    t = float(args['-t'])
    dt = float(args['--dt'])
    times = create_times(t, dt)

    # Setup trials
    n_trials = int(args['-n'])
    trials = range(n_trials)

    Parallel(n_jobs=10, verbose=6)(delayed(exp)(
        times, save_path, i, i + seed, sigma
    ) for i in trials)
