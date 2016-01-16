#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""PAC as selective amplification and information transmission tested
using the Jansen Rit [1995] neural mass model."""
import sys
import os
import pandas as pd
import numpy as np
import pyentropy as en

from sdeint import itoint
from scipy.integrate import odeint, ode
from functools import partial

from pacpy.pac import plv as pacfn
from pacological.jr import jr, S
from pacological.util import create_I, ornstein_uhlenbeck

from joblib import Parallel, delayed
from itertools import product
from collections import defaultdict


def exp(name, t, dt, cs, A=3.25, B=22.0, c=60, p=130.,
        save=True, no_return=True, seed=42, t_min=1):

    if (0 not in cs):
        raise ValueError("cs must contain 0")

    # -- Create driving stimulus
    d = 1  # mean drive rate (want 0-1)
    scale = .01 * d
    Istim = create_I(t, d, scale, dt=dt, seed=seed)

    # -- Intial values (all integrations)
    times = np.linspace(0, t, t / dt)
    rs0 = np.asarray([0., 0., 0., 1., 1., 1.] +  # y set 1
            [0., 0., 0., 1., 1., 1.] +  # y set 2
            [0., 0])  # dW, Istim

    # -- Integrate for stimulus only (no PAC)
    f = partial(jr, Istim=Istim, c=c, c5=0, c6=0, p=p, A=A, B=B)
    g = partial(ornstein_uhlenbeck, sigma=sigma, loc=[1, 2, 6, 7, 8, 12])
    stim_rs = itoint(f, g, rs0, times)
    y0_stim = stim_rs[:,0]

    # -- Integrate for PAC (exploring Stim-Osc coupling, i.e. PAC modes)
    metrics = []
    rss = []
    for i, cpair in enumerate(product(cs, repeat=2)):
        # -- Integrate and extract data
        ce, ci = cpair
        f = partial(jr, Istim=Istim, c=c, c5=ce, c6=ci, p=p, A=A, B=B)
        g = partial(ornstein_uhlenbeck, sigma=sigma, loc=[1, 2, 6, 7, 8, 12])
        rs = itoint(f, g, rs0, times)

        y0_pac = rs[:, 0]
        y_lfp = rs[:, 1] - rs[:, 2]  # Mock EEG

        # create mode labels
        if (ce == 0) and (ci == 0):
            mode = 'stim'
        elif ce == 0:
            mode = 'summed'
        elif ci == 0:
            mode = 'silenced'
        elif ce == ci:
            mode = 'gain'
        else:
            mode = 'unbalanced'

        # Select data for metrics
        select = times >= t_min

        # -- I
        to_calc = ('HX', 'HY', 'HXY')
        m = 8  # Per Ince's advice
        info = en.DiscreteSystem(
            en.quantise(y0_stim[select], m)[0],
            (1, m),
            en.quantise(y0_pac[select], m)[0],
            (1, m)
        )
        info.calculate_entropies(method='pt', calc=to_calc)
        I = info.I()
        H = info.H['HY']

        # -- Spectral
        f = 7
        low_f = (f - 5, f + 5)
        high_f = (30, 250)
        pac = pacfn(
            y_lfp[select], y_lfp[select], low_f, high_f)

        # -- Gather the results
        metrics.append((i, mode, ce, ci, I, H, pac))
        print(name, metrics[-1])

        n = y0_stim.shape[0]
        rss.append(np.vstack([
            np.repeat(i, n), np.repeat(mode, n),
            np.repeat(ce, n), np.repeat(ci, n),
            y0_stim, y0_pac, times,
        ]).T)

    metrics = np.vstack(metrics)
    rss = np.vstack(rss)

    if save:
        df_m = pd.DataFrame(metrics)
        df_m.to_csv(name + "_metrics.csv", index=False,
                header=['i', 'mode', 'ce', 'ci', 'MI', 'H', 'pac'])

        df_rs = pd.DataFrame(rss)
        df_rs.to_csv(name + "_ys.csv", index=False,
                header=['i', 'mode', 'ce', 'ci', 'stim', 'pac', 'times'])

    # Prevents memory overruns when running inside Parallel()
    if no_return:
        return None
    else:
        return metrics, rss


if __name__ == "__main__":
    path = sys.argv[1]
    name = os.path.join(path, "jr")

    n_trials = 20
    t = 2  # run time, ms
    dt = 1 / 10000.  # resolution, ms
    p = 130.  # Jansen range was 120-320

    # O-S connection strength weights (must use 0).
    c56s = range(0, 36, 6) + [1,]

    # Search c (stim only), relative A/B (i.e. g)
    cs = [20, 80, 100, 120, 200]  # base is 100 C
    gs = [1, 2, 3, 4, 5, 6, 7, 8]
    A = 3.25  # Baseline synaptic weight
    trials = range(n_trials)
    params = product(trials, cs, gs)

    # Try three levels of noise
    sigma = 0.2
    Parallel(n_jobs=11)(
        delayed(exp)(
            name + "_sigma-{0}_c-{1}_g-{2}_{3}".format(sigma, c, g, k),
            t, dt, c56s,
            A=A, B=A * g, c=c, p=p,
            save=True, no_return=True, 
            seed=k, t_min=1
        ) for k, c, g in params
    )

