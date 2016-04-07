#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""gNNM run 1, what do all the parameters do?"""

import matplotlib as mpl
mpl.use('Agg')  # No Xll dependency

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import numpy as np
import pyentropy as en

from functools import partial
from scipy.integrate import odeint, ode
from sdeint import itoint

from pacological.util import create_I, ornstein_uhlenbeck
from pacological.chance_zandt import cz, phi
from fakespikes.rates import stim

from joblib import Parallel, delayed
from convenience.numpy import save_hdfz


def exp(tmax, f, d, w_ee, w_ei, w_ie, w_ii, sigma, 
        I_e = 400e-9, I_i=300e-9,
        w_be=400e-9, w_bi=1600e-9, 
        rbe=135e1, rbi=135e1, 
        dt=1e-4, stim_seed=1):
    """gNMM basic exp"""
    
    # --
    # Init time
    times = np.linspace(0, tmax, int(tmax / dt))

    # Create stimulus
    scale = .001 * d
    Istim = create_I(tmax, d, scale, dt=dt, seed=stim_seed)

    # --
    # Run
    ys0 = np.asarray([8.0, 12.0, 
                      w_ee/2, w_ei/2, w_ie/2, w_ii/2, 
                      w_be/2, w_bi/2, w_be/2, w_bi/2])

    f_base = partial(cz, Istim=Istim, 
                w_ee=w_ee, w_ei=w_ei, w_ie=w_ie, w_ii=w_ii, 
                w_be=w_be, w_bi=w_bi, rbe=rbe, rbi=rbi, 
                f=0, 
                I_e=I_e, I_i=I_i)

    f = partial(cz, Istim=Istim, 
                w_ee=w_ee, w_ei=w_ei, w_ie=w_ie, w_ii=w_ii, 
                w_be=w_be, w_bi=w_bi, rbe=rbe, rbi=rbi, 
                f=f, 
                I_e=I_e, I_i=I_i)

    g = partial(ornstein_uhlenbeck, sigma=sigma, loc=[0, 1]) 

    ys_base = itoint(f_base, g, ys0, times)
    ys = itoint(f, g, ys0, times)

    # --
    # MI between stim, and with base of osc
    m = 40
    to_calc = ('HX', 'HY', 'HXY')
    stim = np.asarray([Istim(t) for t in times])
    qstim = en.quantise(stim, m)[0]

    qre_base = en.quantise(ys_base[:, 0], m)[0]
    qre = en.quantise(ys[:, 0], m)[0]

    mi_base = en.DiscreteSystem(
        qstim,
        (1, m),
        qre_base,
        (1, m)
    )
    mi_base.calculate_entropies(method='pt', calc=to_calc)
    
    mi = en.DiscreteSystem(
        qstim,
        (1, m),
        qre,
        (1, m)
    )
    mi.calculate_entropies(method='pt', calc=to_calc)

    mis = [mi_base.I(), mi.I()]
    
    return ys_base, ys, mis, times, stim


if __name__ == "__main__":
    import os, sys
    import csv
    from itertools import product

    path = sys.argv[1]
   
    # Run time
    t = 2
    n_trials = 20

    # Layout search params,
    # FOR TEST
    # fs = np.linspace(4, 40, 1)
    # ds = [400e-9, ]
    # w_ee = 2e-9  # Any larger and no stim makes it though
    # w_eis = np.linspace(0, 80e-9, 2)
    # w_ies = np.linspace(0, 80e-9, 2)
    # w_iis = np.linspace(0, 40e-9, 2)
    # sigmas = np.linspace(40, 160, 1)     

    # # Layout search params,
    # RUN
    fs = np.linspace(4, 20, 5)
    ds = [400e-9, ]
    w_ees = np.linspace(0, 2e-9, 2)
    w_eis = np.linspace(0, 80e-9, 9)
    w_ies = np.linspace(0, 80e-9, 9)
    w_iis = np.linspace(0, 20e-9, 2)
    sigmas = np.linspace(100, 150, 2)
     
    # 5 is temp and because a run crashed.
    for k in range(n_trials):
        print(">>> Trial {0}".format(k))
        for f in fs:
            print(">>> Freq {0}".format(f))

            # and join em. (Call list() becuase we need to reuse params 
            # a couple times)
            params = list(product(ds, w_ees, w_eis, w_ies, w_iis, sigmas))

            # Run.
            exp_k = partial(exp, stim_seed=k)
            res = Parallel(n_jobs=10, verbose=6)(
                delayed(exp_k)(t, f, d, w_ee, w_ei, w_ie, w_ii, sigma) for 
                    d, w_ee, w_ei, w_ie, w_ii, sigma in params
            )

            # --
            # Save params and info one csv file, a summary
            head = ['code', 'fs', 'ds', 'w_ee', 'w_ei', 'w_ie', 'w_ii', 
                    'sigma', 'MI_base', 'MI']

            sum_path = os.path.join(path, "summary_f{1}_{0}.csv".format(k, f))
            with open(sum_path, 'wb') as fi:
                wr = csv.writer(fi)
                wr.writerow(head)
                for i, param in enumerate(params):
                    ys_base, ys, mis, times, stim = res[i]  # Unpack data
                    wr.writerow([i] + list(param) + mis)

            # And raw data in another. 
            # Code defined in the first let's you decode the second
            code = 0
            for ys_base, ys, mis, times, stim in res:
                data_path = os.path.join(path, "data_code{0}_f{2}_{1}".format(
                    code, k, f))
                save_hdfz(data_path, ys_base=ys_base, ys=ys, times=times, 
                        stim=stim, mis=list(mis))
                code += 1
