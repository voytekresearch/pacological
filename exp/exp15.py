#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""PAC as selective amplification and information transmission."""
import numpy as np
import pyentropy as en
import matplotlib.pyplot as plt; plt.ion()

from pacological import pac
from noisy import lfp
from neurosrc.spectral.pac import scpac
# from brian import correlogram

def run(n, t, Iosc, f, Istim, Sstim, dt, k_spikes, excitability,
         pac_type='plv'):
    rate = 1.0 / dt

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
    d_bias['gain_silenced'] = (d_bias['osc'] * d_bias['stim']) - d_bias['osc']
    d_bias['summed'] = d_bias['osc'] + d_bias['stim']
    d_bias['silenced'] = d_bias['stim'] - d_bias['osc']

    # Simulate spiking
    # Create a fixed noiseless stimulus pattern, then....
    stim_sp = drivespikes.poisson(d_bias['stim'])

    # study how different modulation schemes interact
    # with it
    d_spikes = {}
    d_spikes['drive_p'] = stim_sp
    for k in d_bias.keys():
        d_spikes[k + "_p"] = modspikes.poisson(d_bias[k])

    if k_spikes > 0:
        d_spikes['gain_bp'] = modspikes.poisson_binary(
            d_bias['stim'], d_bias['osc'], k=k_spikes,
            excitability=excitability
        )

    # -- CREATE LFP --------------------------------------------------------------
    d_lfps = {}
    for k in d_spikes.keys():
        d_lfps[k] = lfp.create_synaptic_lfps(d_spikes[k])

    # -- I -----------------------------------------------------------------------
    to_calc = ('HX', 'HY', 'HXY')
    m = 8  # Per Ince's advice
    d_infos = {}
    for k in d_lfps.keys():
        d_infos[k] = en.DiscreteSystem(
            en.quantise(d_lfps['drive_p'], m)[0],
            (1, m),
            en.quantise(d_lfps[k], m)[0],
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
    n = 250
    t = 5
    dt = 0.001
    f = 10
    Sstim = .05

    # This ratio of k to excitability gives mean rates
    # equivilant to Poisson
    k_base = 0
    excitability_base = 0.0001
    bin_multipliers = [1, ]

    # Drives and iteration counter
    Ioscs = range(2, 32, 2)
    Istims = range(2, 32, 2)
    iterations = range(200)

    params = product(Ioscs, Istims, bin_multipliers)
    for Iosc, Istim, b_mult in params:
        # Create basename for the data
        basename = "Iosc-{0}_Istim-{1}_k{2}".format(
                Iosc, Istim, b_mult * k_base)
        print(basename)
        basepath = os.path.join(path, basename)

        # Tmp dicts for each param set
        d_H = defaultdict(list)
        d_MI = defaultdict(list)
        d_PAC = defaultdict(list)
        d_rate = defaultdict(list)

        # -- Run
        k = k_base * b_mult
        excitability = excitability_base / b_mult
        for i in iterations:
            print(i)
            res = run(n, t, Iosc, f, Istim, Sstim * Istim, dt, k, excitability)

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
