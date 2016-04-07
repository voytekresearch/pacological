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


def plot_exp(title, pp, tmax, times, stim, ys, ys_base):
    # --
    plt.figure(figsize=(14, 25))
    nplot = 7

    n = 1
    plt.subplot(nplot, 1, n)
    plt.title(title)
    plt.plot(times, stim, label='Stimulus', color='k')
    plt.ylabel("Stimulus drive (mV)")

    n += 1
    plt.subplot(nplot, 1, n)
    plt.plot(times, ys_base[:, 0], color='red', alpha=0.5)
    plt.plot(times, ys_base[:, 1], color='blue', alpha=0.5)
    plt.xlim(0, tmax)
    plt.ylabel("Rate (Hz)")
    plt.legend()

    n += 1
    plt.subplot(nplot, 1, n)
    plt.plot(times, ys[:, 0], label='re', color='red')
    plt.plot(times, ys_base[:, 0], color='red', alpha=0.5)
    plt.xlim(0, tmax)
    plt.ylabel("Rate (Hz)")
    plt.legend()

    n += 1
    plt.subplot(nplot, 1, n)
    plt.plot(times, ys[:, 1], label='ri', color='blue')
    plt.plot(times, ys_base[:, 1], color='blue', alpha=0.5)
    plt.xlim(0, tmax)
    plt.ylabel("Rate (Hz)")
    plt.legend()

    n += 1
    plt.subplot(nplot, 1, n)
    plt.plot(times, ys[:, 2], label='g_ee')
    plt.plot(times, ys[:, 3], label='g_ie')
    plt.plot(times, ys[:, 4], label='g_ei')
    plt.plot(times, ys[:, 5], label='g_ii')
    plt.xlim(0, tmax)
    plt.ylabel("Conductance")
    plt.legend()

    n += 1
    plt.subplot(nplot, 1, n)
    plt.plot(times, ys[:, 6], label='g_be')
    plt.plot(times, ys[:, 7], label='g_bi')
    plt.plot(times, ys[:, 6] - ys[:, 7], label='g_be - g_bi')
    plt.xlim(0, tmax)
    plt.axhline(0, color='k')
    plt.ylabel("Conductance")
    plt.legend()

    n += 1
    plt.subplot(nplot, 1, n)
    Sb = ys[:, 8] + ys[:, 9]
    plt.plot(times, Sb, label='s_b')
    plt.xlim(0, tmax)
    plt.axhline(0, color='k')
    plt.ylabel("Conductance variance")
    plt.legend()

    plt.xlabel("Time (s)")
    pp.savefig()

    plt.close()


def exp(tmax, f, d, w_ee, w_ei, w_ie, w_ii, sigma, 
        I_e = 400e-9, I_i=200e-9,
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

    # Layout search params,
    # FOR TEST
    # fs = np.linspace(4, 40, 1)
    # ds = [400e-9, ]
    # w_ees = np.linspace(0, 40e-9, 2)
    # w_eis = np.linspace(0, 80e-9, 2)
    # w_ies = np.linspace(0, 80e-9, 2)
    # w_iis = np.linspace(0, 40e-9, 2)
    # sigmas = np.linspace(40, 160, 1)     
    
    # # Layout search params,
    # RUN
    fs = np.linspace(4, 40, 5)
    ds = [400e-9, 500e-9]
    w_ees = np.linspace(0, 40e-9, 3)
    w_eis = np.linspace(0, 80e-9, 4)
    w_ies = np.linspace(0, 80e-9, 4)
    w_iis = np.linspace(0, 40e-9, 3)
    sigmas = np.linspace(40, 160, 3)
    
    # and join em. (Call list() becuase we need to reuse params 
    # a couple times)
    params = list(product(fs, ds, w_ees, w_eis, w_ies, w_iis, sigmas))

    # Run.
    res = Parallel(n_jobs=10, verbose=5)(
        delayed(exp)(t, f, d, w_ee, w_ei, w_ie, w_ii, sigma) for 
            f, d, w_ee, w_ei, w_ie, w_ii, sigma in params
    )

    # --
    # Save params and info one csv file, a summary
    head = ['code', 'fs', 'ds', 'w_ee', 'w_ei', 'w_ie', 'w_ii', 
            'sigma', 'MI_base', 'MI']
    with open(os.path.join(path, "summary.csv"), 'wb') as fi:
        wr = csv.writer(fi)
        wr.writerow(head)
        for i, param in enumerate(params):
            ys_base, ys, mis, times, stim = res[i]  # Unpack data
            wr.writerow([i] + list(param) + mis)

    # And raw data in another. 
    # Code defined in the first let's you decode the second
    code = 0
    for ys_base, ys, mis, times, stim in res:
        np.savez(os.path.join(path, "data_{0}".format(code)), 
                ys_base=ys_base, ys=ys, times=times, stim=stim,
                mis=mis)
        code += 1
