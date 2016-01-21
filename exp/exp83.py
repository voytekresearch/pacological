#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""PAC as selective amplification and information transmission tested
using model based on XJW's neural mass models in Nueron 2015"""
import sys
import os
import pandas as pd
import numpy as np
import pyentropy as en

from sdeint import itoint
from scipy.integrate import odeint, ode
from functools import partial

from pacpy.pac import plv as pacfn
from pacological.xjw import xjw
from pacological.util import create_I, ornstein_uhlenbeck, phi

from joblib import Parallel, delayed
from itertools import product
from collections import defaultdict


def exp(name, t, dt, Je_e=0, Je_i=0, Ji_e=0, Ji_i=0, k1=0.9, k2=1.2,
        save=True, no_return=True, seed=42, t_min=1, sigma=0.2):
    
    i = 0   # Hold over from jr that had many ce/ci 
            # pairs for which i was a unique index

    if not save and no_return:
        raise ValueError("Save must be True, or no_return must be False")

    # -- Create driving stimulus
    d = 1  # mean drive rate (want 0-1)
    scale = .01 * d
    Istim = create_I(t, d, scale, dt=dt, seed=seed)

    # -- Intial values (all integrations)
    times = np.linspace(0, t, t / dt)

    r0 = [8, 12.0]  # intial rates (re, ri)
    rs0 = np.asarray(r0 * 8)

    # -- Integrate for stimulus only (no PAC)
    f1 = partial(xjw, Istim=Istim, Je_e=0.0, Je_i=0.0, Ji_e=0.0, Ji_i=0.0,
            k1=k1, k2=k2)
    g1 = partial(ornstein_uhlenbeck, sigma=sigma, loc=[0, 1, 6, 7]) 

    rs1 = itoint(f1, g1, rs0, times)
    y0_stim = rs1[:, 0]

    metrics = []
    rss = []

    # -- Integrate and extract data
    f2 = partial(xjw, Istim=Istim, Je_e=Je_e, Je_i=Je_i, Ji_e=Ji_e, Ji_i=Ji_i, 
                 k1=k1, k2=k2)
    g2 = partial(ornstein_uhlenbeck, sigma=sigma, loc=[0, 1, 6, 7]) 

    rs2 = itoint(f2, g2, rs0, times)
    y0_pac = rs2[:,0]
    y_lfp = rs2[:,0] - rs2[:,1]

    # create mode labels
    if (Je_e == 0) and (Je_i == 0):
        mode = 'stim'
    elif Je_i == 0:
        mode = 'summed'
    elif Je_e == 0:
        mode = 'silenced'
    elif Je_e == Je_i:
        mode = 'gain'
    else:
        mode = 'unbalanced'

    # Select data for metrics
    select = times >= t_min

    # -- I
    to_calc = ('HX', 'HY', 'HXY')
    # m = 8  # Per Ince's advice
    m = 30
    info = en.DiscreteSystem(
        en.quantise(y0_stim[select], m)[0],
        (1, m),
        en.quantise(y0_pac[select], m)[0],
        (1, m)
    )
    info.calculate_entropies(method='pt', calc=to_calc)
    I = info.I()
    H = info.H['HX']

    # -- Spectral
    f = 7
    low_f = (f - 5, f + 5)
    high_f = (30, 250)
    pac = pacfn(y_lfp[select], y_lfp[select], low_f, high_f)

    # -- Gather the results
    metrics.append((i, mode, Je_e, Je_i, Ji_e, Ji_i, k1, k2, I, H, pac))
    print(name, metrics[-1])

    n = y0_stim.shape[0]
    rss.append(np.vstack([
        np.repeat(i, n), y0_stim, y0_pac, times,
    ]).T)

    metrics = np.vstack(metrics)
    rss = np.vstack(rss)

    if save:
        df_m = pd.DataFrame(metrics)
        df_m.to_csv(name + "_metrics.csv", index=False,
                header=['i', 'mode', 'Je_e', 'Je_i', 'Ji_e', 'Ji_i', 
                    'k1', 'k2', 'I', 'H', 'pac'])

        df_rs = pd.DataFrame(rss)
        df_rs.to_csv(name + "_ys.csv", index=False,
                header=['i', 'stim', 'pac', 'times'])

    # Prevents memory overruns when running inside Parallel()
    if no_return:
        return None
    else:
        return metrics, rss


if __name__ == "__main__":
    path = sys.argv[1]
    name = os.path.join(path, "xjw")

    n_trials = 20
    t = 2.  # run time, ms
    dt = 1. / 10000  # resolution, ms  

    Js = np.arange(0.0, 3.5, 0.5)
    
    trials = range(n_trials)
    params = product(trials, Js, Js, Js, Js)
 
    # Set noise
    sigma = 20.0  # Eyeballed to look like LFP
    
    # go
    Parallel(n_jobs=11)(
        delayed(exp)(name + "_sigma-{0}_Js-".format(sigma) +  # awful name code
            "_".join([str(Je_e), str(Je_i), str(Ji_e), str(Ji_i), 
                str(k)]),
            t, dt, Je_e=Je_e, Je_i=Je_i, Ji_e=Ji_e, Ji_i=Ji_i, 
            k1=0.5, k2=1.2,
            save=True, no_return=True, 
            seed=k, t_min=1, sigma=sigma
        ) for k, Je_e, Je_i, Ji_e, Ji_i in params
    )
