#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""PAC as selective amplification and information transmission."""
import numpy as np
import pyentropy as en
import matplotlib.pyplot as plt; plt.ion()

from pacological import pac
from noisy import lfp
from neurosrc.pac.pac_tools import pac as scpac
# from brian import correlogram

def run(n, n_b, t, Iosc, f, Istim, Sstim, Iback, Ipub, Ipri,
        dt, pac_type='plv', stim_seed=None):
    # rate = 1.0 / dt

    # -- SIM ---------------------------------------------------------------------
    # Init spikers
    backspikes = pac.Spikes(n_b, t, dt=dt)
    pacspikes = pac.Spikes(n, t, dt=dt, private_stdev=Ipri)
    drivespikes = pac.Spikes(n, t, dt=dt, private_stdev=Ipri)
    times = pacspikes.times  # brevity

    # --
    # Create biases
    d_bias = {}
    d_bias['back'] = pac.stim(times, Istim, Sstim, seed=stim_seed)
    d_bias['public'] = pac.constant(times, Ipub)
    d_bias['osc'] = pac.osc(times, Iosc, f)
    d_bias['stim'] = pac.stim(times, Istim, Sstim, seed=stim_seed)

    d_bias['gain'] = d_bias['osc'] * d_bias['stim']
    d_bias['summed'] = d_bias['osc'] + d_bias['stim']
    d_bias['silenced'] = d_bias['stim'] - d_bias['osc']

    # --
    # Simulate spiking
    #
    # Create the background pool.
    b_spks = backspikes.poisson(d_bias['back'])

    # Create a non-PAC stimulus pattern for MI inqualities
    stim_sp = np.hstack([
        drivespikes.poisson(d_bias['stim']),
        b_spks
    ])

    # and then create PAC spikes.
    d_spikes = {}
    for k in d_bias.keys():
        d_spikes[k + "_p"] = np.hstack([
            pacspikes.poisson(d_bias[k]),
            b_spks
        ])

    # -- CREATE LFP -----------------------------------------------------------
    d_lfps = {}
    for k in d_spikes.keys():
        d_lfps[k] = lfp.create_synaptic_lfps(d_spikes[k])

    # -- I --------------------------------------------------------------------
    to_calc = ('HX', 'HY', 'HXY')
    m = 8  # Per Ince's advice
    d_infos = {}
    for k in d_spikes.keys():
        d_infos[k] = en.DiscreteSystem(
            en.quantise_discrete(stim_sp.sum(1), m),
            (1, m),
            en.quantise_discrete(d_spikes[k].sum(1), m),
            (1, m)
        )
        d_infos[k].calculate_entropies(method='pt', calc=to_calc)

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
    method = pac_type

    d_pacs = {}
    for k in d_lfps.keys():
        _, _, d_pacs[k], _ = scpac(d_lfps[k], low_f, high_f, method)

    return {
        'MI' : d_mis,
        'H' : d_hs,
        'PAC' : d_pacs,
        'spikes' : d_spikes,
        'times' : times
    }


if __name__ == "__main__":
    import sys
    import pandas as pd
    import os
    from itertools import product
    from collections import defaultdict

    path = sys.argv[1]

    # -- USER SETTINGS --------------------------------------------------------
    N = 500
    pn = 0.75
    n = int(pn * N)
    n_b = int((1 - pn) * N)

    Iback = 2
    Ipub = 1
    Ipri = 0.1
    Sstim = .05

    t = 5
    dt = 0.001
    f = 10

    # Drives and iteration counter
    Ioscs = range(2, 32, 2)
    Istims = range(2, 32, 2)
    iterations = range(20)

    params = product(Ioscs, Istims)
    for Iosc, Istim in params:
        # Create basename for the data
        basename = "Iosc-{0}_Istim-{1}".format(
                Iosc, Istim)
        print(basename)
        basepath = os.path.join(path, basename)

        # Tmp dicts for each param set
        d_H = defaultdict(list)
        d_MI = defaultdict(list)
        d_PAC = defaultdict(list)
        d_rate = defaultdict(list)

        # -- Run
        for i in iterations:
            print(i)
            res = run(n, n_b, t, Iosc, f, Istim, Sstim * Istim, Iback, Ipub, Ipri,
                dt, pac_type='plv', stim_seed=i)

            # Process the result
            hys = {}
            for b in res['H'].keys():
                hys[b] = res['H'][b]['HY']
            for b in hys.keys():
                d_H[b].append(hys[b])
            for b in res['MI'].keys():
                d_MI[b].append(res['MI'][b])
            for b in res['PAC'].keys():
                d_PAC[b].append(res['PAC'][b])

            for b in res['spikes'].keys():
                mrate = np.mean(res['spikes'][b].sum(0) / float(t))
                d_rate[b].append(mrate)

        # -- Save
        # H
        df_H = pd.DataFrame(d_H)
        df_H.to_csv(basepath + "_H.csv", index=False)

        sum_H = df_H.describe(percentiles=[.05, .25, .75, .95]).T
        sum_H.to_csv(basepath + "_H_summary.csv")

        # MI
        df_MI = pd.DataFrame(d_MI)
        df_MI.to_csv(basepath + "_MI.csv", index=False)

        sum_MI = df_MI.describe(percentiles=[.05, .25, .75, .95]).T
        sum_MI.to_csv(basepath + "_MI_summary.csv")

        # PAC
        df_PAC = pd.DataFrame(d_PAC)
        df_PAC.to_csv(basepath + "_PAC.csv", index=False)

        sum_PAC = df_PAC.describe(percentiles=[.05, .25, .75, .95]).T
        sum_PAC.to_csv(basepath + "_PAC_summary.csv")

        # rate
        df_rate = pd.DataFrame(d_rate)
        df_rate.to_csv(basepath + "_rate.csv", index=False)

        sum_rate = df_rate.describe(percentiles=[.05, .25, .75, .95]).T
        sum_rate.to_csv(basepath + "_rate_summary.csv")
