#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""PAC as selective amplification and information transmission."""
import numpy as np
import matplotlib.pyplot as plt; plt.ion()


if __name__ == "__main__":
    from pacological.exp.exp6 import run
    from pacological import pac
    import sys
    import pandas as pd
    import os

    path = sys.argv[1]

    # -- USER SETTINGS --------------------------------------------------------
    n = 100
    t = 5
    dt = 0.001
    f = 10
    Sstim = .05

    # This ratio of k to excitability gives mean rates
    # equivilant to Poisson
    k_base = 1
    excitability_base = 0.0001
    bin_multipliers = range(2, 32, 2)

    # Drives and iteration counter
    Iosc = 5
    Istim = 5
    spikes, stims = {}, {}
    for b_mult in bin_multipliers:
        # -- Run
        k = k_base * b_mult
        excitability = excitability_base / b_mult

        res = run(n, t, Iosc, f, Istim, Sstim * Istim, dt, k, excitability)

        spikes[k] = res['spikes']['gain_bp']
        stims[k] = res['spikes']['stim_p']

