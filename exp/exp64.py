#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""PAC as selective amplification and information transmission."""
import sys
import pandas as pd
import os
import numpy as np
import pyentropy as en
from fakespikes import neurons, rates
from pacpy.pac import plv as pacfn
from joblib import Parallel, delayed
from itertools import product
from collections import defaultdict
from scipy.stats.mstats import zscore


def _run(n, n_b, t, Iosc, f, g, Istim, Sstim, Ipri,
        dt, back_type, stim_seed=None):
    # rate = 1.0 / dt

    # -- SIM ---------------------------------------------------------------------
    # Init spikers
    backspikes = neurons.Spikes(n_b, t, dt=dt)
    pacspikes = neurons.Spikes(n, t, dt=dt, private_stdev=Ipri)
    drivespikes = neurons.Spikes(n, t, dt=dt, private_stdev=Ipri)
    times = pacspikes.times  # brevity

    # --
    # Create biases
    d_bias = {}
    if back_type == 'constant':
        d_bias['back'] = rates.constant(times, 2)
    elif back_type == 'stim':
        d_bias['back'] = rates.stim(times, Istim, Sstim, seed=stim_seed)
    else:
        raise ValueError("pac_type not understood")

    # Drive and osc
    d_bias['osc'] = rates.osc(times, Iosc, f)
    d_bias['stim'] = rates.stim(times, Istim, Sstim, seed=stim_seed)

    # PAC math
    d_bias['gain'] = d_bias['stim'] * (g * d_bias['osc'])
    d_bias['summed'] = d_bias['stim'] + (g * d_bias['osc'])
    d_bias['silenced'] = d_bias['stim'] - (g * d_bias['osc'])

    # --
    # Simulate spiking
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

    # -- LFP ------------------------------------------------------------------
    d_lfps = {}
    for k in d_spikes.keys():
        d_lfps[k] = zscore(d_spikes[k].sum(1).astype(float))

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

    # -- PAC ------------------------------------------------------------------
    low_f = (f-2, f+2)
    high_f = (80, 250)

    d_pacs = {}
    for k in d_lfps.keys():
        d_pacs[k] = pacfn(d_lfps[k], d_lfps[k], low_f, high_f)

    return {
        'MI' : d_mis,
        'H' : d_hs,
        'PAC' : d_pacs,
        'spikes' : d_spikes,
        'times' : times
    }


def exp(Istim, g, N, pn, n_trials=20):
    path = sys.argv[1]

    # -- USER SETTINGS --------------------------------------------------------
    t = 5
    dt = 0.001
    f = 6
    # back_type = 'constant'
    back_type = 'stim'

    # Drives and iteration counter
    Iosc = 2
    Ipri = 0
    # Iback = 2
    # Ipub = 1

    Sstim = .01 * Istim
    n = int(pn * N)
    n_b = int((1 - pn) * N)
    if n_b < 2:
        n_b = 2

    # Create basename for the data
    basename = "Istim-{0}_g-{1}_N-{2}_pn-{3}_".format(Istim, g, N, pn)
    print(basename)
    basepath = os.path.join(path, basename)

    # Tmp dicts for each param set
    d_H = defaultdict(list)
    d_MI = defaultdict(list)
    d_PAC = defaultdict(list)
    d_rate = defaultdict(list)

    # -- Run
    iterations = range(n_trials)
    for i in iterations:
        print(i)
        res = _run(n, n_b, t, Iosc, f, g, Istim, Sstim, Ipri,
                    dt, back_type, stim_seed=i)

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


if __name__ == "__main__":
    Istims = range(2, 32, 4)
    gs = range(1, 9)
    Ns = range(100, 600, 100)
    pns = [0.25, 0.5, 0.75, 1]

    params = product(Istims, gs, Ns, pns)
    Parallel(n_jobs=12)(
        delayed(exp)(Istim, g, N, pn) for Istim, g, N, pn in params
    )
