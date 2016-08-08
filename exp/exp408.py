"""Explore the bluemass5 parameter space"""
from __future__ import division
import os, sys
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


def exp(times, save_path, i, seed, d, w_e, w_ie, w_ei, w_ii, w_ee):
    # Input
    inputs[0][1]['w'] = w_e

    # Reinit stim
    scale = .01 * d
    stim = create_stim_I(times, d, scale, seed=seed)
    # stim = create_constant_I(times, d, seed=seed)

    # Network
    conns[0][2]['w'] = w_ee
    conns[1][2]['w'] = w_ei
    conns[2][2]['w'] = w_ie
    conns[3][2]['w'] = w_ii
    pars = BMparams(pops, conns, backs, inputs, sigma=0,
            I_max=100e-3,
            background_res=0)

    # System
    gn = partial(ornstein_uhlenbeck, sigma=0.01, loc=[])
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
             d=d,
             stim=np.asarray([stim(t) for t in times]),
             seed=seed,
             w_e=w_e,
             w_ee=w_ee,
             w_ei=w_ei,
             w_ie=w_ie,
             w_ii=w_ii)


if __name__ == "__main__":
    seed = 42

    execfile(sys.argv[1])

    save_path = sys.argv[2]

    # Setup time
    t = 1.0
    dt = 1e-3
    times = create_times(t, dt)

    # Init params
    r_stims = [10, ]
    w_es = [3 / 1e3, ]
    w_ies = np.linspace(2, 20.0, 10) / 1e3
    w_eis = np.linspace(2, 20.0, 10) / 1e3
    w_iis = np.linspace(2, 14.0, 5) / 1e3
    w_ees = np.linspace(2, 14.0, 5) / 1e3
    params = product(r_stims, w_es, w_ies, w_eis, w_iis, w_ees)

    Parallel(n_jobs=10, verbose=6)(delayed(exp)(times, save_path, i, seed, *p)
                                   for i, (p) in enumerate(params))
