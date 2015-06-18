#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""PAC and information and alternate bernoulli firing scheme."""
import numpy as np
import pyentropy as en
import matplotlib.pyplot as plt; plt.ion()

from pacological import pac
from noisy import lfp
from neurosrc.spectral.pac import scpac
# from brian import correlogram


def main(n, t, Iosc, f, Istim, Sstim, dt, excitability):
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
        d_spikes[k + "_b"] = modspikes.bernoulli(
            d_bias[k], excitability=excitability
        )

    d_spikes['gain_bp'] = modspikes.poisson_bernoulli(
        d_bias['stim'], d_bias['osc'], excitability=excitability
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
        'PAC' : d_pacs,
        'spikes' : d_spikes
    }


if __name__ == "__main__":
    import sys
    import pandas as pd
    import os
    from itertools import product
    from collections import defaultdict

    path = sys.argv[1]

    # -- USER SETTINGS --------------------------------------------------------
    n = 500
    t = 3
    dt = 0.001
    f = 10
    Sstim = .05

    Ioscs = [5, 10, 15, 20, 25, 30]
    Istims = [5, 10, 15, 20, 25, 30]
    excitabilites = [0.2, 0.1, 0.05]

    params = product(Ioscs, Istims, excitabilites)

    iterations = range(20)
    for Iosc, Istim, ex in params:
        # Create basename for the data
        basename = "Iosc-{0}_Istim-{1}_ex-{2}".format(
                Iosc, Istim, ex)
        basepath = os.path.join(path, basename)

        # Tmp dicts for each param set
        d_H = defaultdict(list)
        d_MI = defaultdict(list)
        d_PAC = defaultdict(list)
        d_rate = defaultdict(list)

        # -- Run
        for i in iterations:
            res = main(n, t, Iosc, f, Istim, Sstim, dt, ex)

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
