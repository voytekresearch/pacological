"""Explore the bluemass3 parameter space"""
from __future__ import division
import os, sys
from copy import deepcopy
from functools import partial
from sdeint import itoint
from itertools import product

from pacological.bluemass3 import create_layers, create_ys0
from pacological.util import create_I, ornstein_uhlenbeck
from joblib import Parallel, delayed


def exp(t, dt, pars, save_path, i, seed, d, w_e, w_ie, w_ei, w_ii, w_ee):
    # Input
    pars.inputs[0][1]['w'] = w_e

    # Reinit stim
    scale = .01 * d
    stim = create_I(t, d, scale, dt=dt, seed=seed)

    # Network
    pars.conns[0][2]['w'] = w_ee
    pars.conns[1][2]['w'] = w_ei
    pars.conns[2][2]['w'] = w_ie
    pars.conns[3][2]['w'] = w_ii

    # System
    gn = partial(ornstein_uhlenbeck, sigma=0.01, loc=[])
    fn, idxs = create_layers(stim, pars, seed=seed)

    # Init the intial values
    ys0 = create_ys0(pars, idxs, frac=0.1)

    # Integrate!
    ys = itoint(fn, gn, ys0, times)

    # TODO save into h/kdf instead...
    np.savez("{}".format(os.path.join(save_path, i)),
             ys=ys,
             ys0=ys0,
             idxs=idxs,
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

    pars = None
    execfile(sys.argv[1])

    save_path = sys.argv[2]

    # Setup time
    t = 3.0
    dt = 1e-3
    n_step = int(np.ceil(t / dt))
    times = np.linspace(0, t, n_step)

    # Init params
    r_stims = [1, 10, 100]
    w_es = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
    w_ies = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
    w_eis = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
    w_iis = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
    w_ees = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
    params = product(r_stims, w_es, w_ies, w_eis, w_iis, w_ees)

    Parallel(n_jobs=1, verbose=5)(
        delayed(exp)(t, dt, pars, save_path, i, seed, *p)
        for i, (p) in enumerate(params))
