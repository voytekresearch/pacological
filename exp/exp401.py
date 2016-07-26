"""Explore the bluemass5 parameter space"""
from __future__ import division
import os, sys
from copy import deepcopy
from functools import partial
from sdeint import itoint
from itertools import product

from pacological.pars import BMparams
from pacological.bluemass5 import create_layers, create_ys0
from pacological.util import create_stim_I, ornstein_uhlenbeck
from joblib import Parallel, delayed
from pykdf.kdf import save_kdf, load_kdf


def exp(t, dt, save_path, i, seed, d, w_e, w_ie, w_ei, w_ii, w_ee):
    # Input
    inputs[0][1]['w'] = w_e

    # Reinit stim
    scale = .01 * d
    stim = create_stim_I(t, d, scale, dt=dt, seed=seed)

    # Network
    conns[0][2]['w'] = w_ee
    conns[1][2]['w'] = w_ei
    conns[2][2]['w'] = w_ie
    conns[3][2]['w'] = w_ii

    pars = BMparams(pops, conns, backs, inputs, sigma=0, background_res=0)

    # System
    gn = partial(ornstein_uhlenbeck, sigma=0.01, loc=[])
    fn, idxs = create_layers(stim, pars, seed=seed)

    # Init the intial values
    ys0 = create_ys0(pars, idxs, frac=0.1)

    # Integrate!
    ys = itoint(fn, gn, ys0, times)

    # TODO save into h/kdf instead...
    # import pdb; pdb.set_trace()

    save_kdf("{}".format(os.path.join(save_path, str(i))),
             ys=ys,
             ys0=ys0,
             idx_R=idxs['R'],
             idx_IN=idxs['IN'],
             idx_INsigma=idxs['INsigma'],
             idx_H=idxs['H'],
             idx_Hsigma=idxs['Hsigma'],
             times=times,
             t=t,
             dt=dt,
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
    n_step = int(np.ceil(t / dt))
    times = np.linspace(0, t, n_step)

    # Init params
    r_stims = [10, ]
    w_es = [2 / 1e3]
    w_eis = [10 / 1e3, ]
    w_ies = np.linspace(1, 90.0, 20) / 1e3 
    w_iis = np.linspace(1, 20.0, 2) / 1e3 
    w_ees = np.linspace(1, 20.0, 2) / 1e3 
    params = product(r_stims, w_es, w_ies, w_eis, w_iis, w_ees)

    Parallel(n_jobs=1, verbose=5)(
        delayed(exp)(t, dt, save_path, i, seed, *p)
        for i, (p) in enumerate(params))
