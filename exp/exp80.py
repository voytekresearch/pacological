#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""PAC as selective amplification and information transmission."""
import sys
import os
import pandas as pd
import numpy as np
import pyentropy as en
from pacpy.pac import plv as pacfn
from pacological.jr import jr, S, run
from joblib import Parallel, delayed
from itertools import product
from collections import defaultdict


def exp(name, t, dt, p, sigma, cs, save=True, seed=42, t_min=1):
    print("Running experiment {0}".format(name))

    if (0 not in cs):
        raise ValueError("cs must contain 0")

    rs0 = np.asarray([0., 0., 0., 1., 1., 1.] + 
            [0., 0., 0., 1., 1., 1.] + [0., 0])

    # Stim only
    _, stim_rs = run(t, dt, rs0, p, 0, 0, sigma, seed=seed)
    pyramidal_stim = stim_rs[:,0]

    # Init for PAC
    metrics = []

    n = stim_rs.shape[0]
    rss = [np.vstack([
        np.repeat(-1, n), np.repeat('stim', n), 
        np.repeat(0, n), np.repeat(0, n), 
        stim_rs.T
    ]).T]

    # go PAC
    for i, cpair in enumerate(product(cs, repeat=2)):
        # -- Integrate and extract data --------------------------------------
        ce, ci = cpair
        times, rs = run(t, dt, rs0, p, ce, ci, sigma, seed=seed)
        
        select = times >= t_min

        pyramidal_pac = rs[select,0]
        pyramidal_lfp = rs[select,1] - rs[select,2]  # Mock EEG
        
        # create labels
        if (ce == 0) and (ci == 0):
            lab = 'stim'
        elif ce == 0:
            lab = 'summed'
        elif ci == 0:
            lab = 'silenced'
        elif ce == ci:
            lab = 'gain'
        else:
            lab = 'unbalamced'

        # -- I ---------------------------------------------------------------
        to_calc = ('HX', 'HY', 'HXY')
        m = 8  # Per Ince's advice
        info = en.DiscreteSystem(
            en.quantise(pyramidal_stim[select], m)[0],
            (1, m),
            en.quantise(pyramidal_pac, m)[0],
            (1, m)
        )
        info.calculate_entropies(method='pt', calc=to_calc)
        I = info.I()
        H = info.H['HY']

        # -- Spectral --------------------------------------------------------
        f = 7
        low_f = (f-3, f+3)
        high_f = (80, 250)
        pac = pacfn(pyramidal_lfp, pyramidal_lfp, low_f, high_f)
        
        # -- Gather the results ----------------------------------------------
        metrics.append((i, lab, ce, ci, I, H, pac))
        print(name, metrics[-1])

        n = rs.shape[0]
        rss.append(np.vstack([
            np.repeat(i, n), np.repeat(lab, n), 
            np.repeat(ce, n), np.repeat(ci, n), rs.T
        ]).T)

    metrics = np.vstack(metrics)
    rss = np.vstack(rss)
    if save:
        df_m = pd.DataFrame(metrics)
        df_m.to_csv(name + "_metrics.csv", index=False, 
                header=['i', 'mode', 'ce', 'ci', 'MI', 'H', 'pac'])

        df_rs = pd.DataFrame(rss)
        df_rs.to_csv(name + "_r.csv", index=False, header=False)

    return metrics, rss


if __name__ == "__main__":
    path = sys.argv[1]
    name = os.path.join(path, "jr")

    t = 2  # run time, ms
    dt = 1 / 10000.  # resolution, ms
    p = 130. # Jansen range was 120-320
    sigma = 0.05

    # O-S connection strength weights (don't use 0).
    cs = range(0, 32, 2)

    n_trials = 20

    Parallel(n_jobs=11)(
        delayed(exp)(
                name + "_{0}".format(k), 
                t, dt, 
                p, sigma, cs, 
                save=True, seed=k
            ) for k in range(n_trials)
    )

