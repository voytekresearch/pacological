#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""An intial PAC experiment"""
# import numpy as np
import pyentropy as en
import seaborn as sns
import matplotlib.pyplot as plt; plt.ion()

from pacological import pac
from noisy import lfp
from neurosrc.spectral.pac import scpac
# from brian import correlogram


def main(n, t, Iosc, f, Istim, Sstim, dt, k_spikes, excitability):
    rate = 1 / dt

    # -- SIM ---------------------------------------------------------------------
    # Init spikers
    modspikes = pac.Spikes(n, t, dt=dt)
    drivespikes = pac.Spikes(n, t, dt=dt)
    times = modspikes.times  # brevity

    # Create biases
    d_bias = {}
    d_bias['osc'] = pac.osc(times, Iosc, f)
    d_bias['stim'] = pac.stim(times, Istim, Sstim)

    d_bias['gain'] = d_bias['osc'] * d_bias['stim']
    d_bias['summed'] = d_bias['osc'] + d_bias['stim']
    d_bias['silenced'] = d_bias['stim'] - d_bias['osc']

    # Simulate spiking
    # Create a fixed noiseless stimulus pattern, then....
    stim_sp = drivespikes.poisson(d_bias['stim'])

    # study how different modulation schemes interact
    # with it
    d_spikes = {}
    for k in d_bias.keys():
        d_spikes[k + "_p"] = modspikes.poisson(d_bias[k])
        d_spikes[k + "_b"] = modspikes.binary(
            d_bias[k], k=k_spikes, excitability=excitability
        )

    d_spikes['gain_bp'] = modspikes.poisson_binary(
        d_bias['stim'], d_bias['osc'], k=k_spikes, excitability=excitability
    )

    # -- CREATE LFP --------------------------------------------------------------
    d_lfps = {}
    for k in d_spikes.keys():
        d_lfps[k] = lfp.create_lfps(d_spikes[k])

    # -- I -----------------------------------------------------------------------
    # Spikes
    d_infos = {}
    m = 20  # Per Ince's advice

    for k in d_spikes.keys():
        d_infos[k] = en.DiscreteSystem(
            en.quantise_discrete(stim_sp.sum(1), m),
            (1, len(stim_sp)),
            en.quantise_discrete(d_spikes[k].sum(1), m),
            (1, len(d_spikes[k]))
        )
        d_infos[k].calculate_entropies(method='plugin', calc=('HX', 'HY', 'HXY'))

    # MI
    d_mis = {}
    for k, mi in d_infos.items():
        d_mis[k] = mi.I()

    # H
    d_hs = {}
    for k, mi in d_infos.items():
        d_hs[k] = mi.H

    # -- PAC OF LFP --------------------------------------------------------------
    low_f = (f-2, f+2)
    high_f = (80, 250)
    method = 'plv'
    filt = 'eegfilt'
    kwargs = {'trans' : .15} # for eegfilt

    d_pacs = {}
    for k in d_lfps.keys():
        d_pacs[k] = scpac(d_lfps[k], low_f, high_f, rate, method, filt, **kwargs)

    return {
        'MI' : d_mis,
        'H' : d_hs,
        'PAC' : d_pacs
    }


if __name__ == "__main__":
    import sys
    import pandas as pd
    # import glob
    import os
    # import numpy as np
    from itertools import product

    path = sys.argv[1]
    if os.path.exists(path):
        raise ValueError("Path exists. Won't overwrite")
    os.mkdir(path)

    # -- USER SETTINGS -----------------------------------------------------------
    n = 500
    t = 3
    dt = 0.001
    Sstim = .1

    Ioscs = [2, 30]
    fs = [6, 12]
    Istims = [2, 30]
    ks = [5, 50]
    excitabilities = [0.0001, 0.001]
    iterations = range(100)
    params = product(iterations, Ioscs, fs, Istims, ks, excitabilities)

    for i, Iosc, f, Istim, k, excitability in params:
        # Create basename for the data
        basename = os.path.join(
            path,
            "_".join([str(Iosc), str(f),
                      str(Istim), str(k),
                      str(excitability), str(i)])
        )

        # Run
        res = main(n, t, Iosc, f, Istim, Sstim, dt, k, excitability)

        # Process and save the result
        hys = {}
        for b in res['H'].keys():
            hys[b] = res['H'][b]['HY']

        df1 = pd.DataFrame(hys)
        df1.to_csv(
            "{0}_HY.csv".format(basename),
            index=False, header=True
        )

        df2 = pd.DataFrame(res['MI'], index=[0])
        df2.to_csv(
            "{0}_MI.csv".format(basename),
            index=False, header=True
        )

        df3 = pd.DataFrame(res['PAC'], index=[0])
        df3.to_csv(
            "{0}_PAC.csv".format(basename),
            index=False, header=True
        )

