#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""PAC as selective amplification and information transmission."""
import numpy as np
import matplotlib.pyplot as plt; plt.ion()


if __name__ == "__main__":
    from pacological.exp.exp6 import run
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
    pac_type = 'mi'
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
            res = run(n, t, Iosc, f, Istim, Sstim * Istim, dt, k, excitability,
                      pac_type=pac_type)

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
