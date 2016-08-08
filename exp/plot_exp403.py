import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import os, sys
import numpy as np
from pykdf.kdf import load_kdf


def plot(name, pp):
    res = load_kdf(name)
    ys = res['ys']
    times = res['times']
    idx_r = res['idx_R']
    idx_h = res['idx_H']
    idx_in = res['idx_IN']

    plt.figure(figsize=(10, 10))

    tit = "d:{:10f}, w_e:{:10f}\nw_ie:{:10f}, w_ei:{:10f}\nw_ee:{:10f}, w_ii:{:10f}".format(
        float(res['d']), float(res['w_e']), float(res['w_ie']),
        float(res['w_ei']), float(res['w_ee']), float(res['w_ii']))

    n_plot = 3
    plot_n = 1
    plt.subplot(n_plot, 1, plot_n)
    plt.title(tit)
    plt.plot(times, ys[:, idx_in], label='IN', color='k')
    plt.legend()
    plt.ylabel("IN")
    plot_n += 1

    plt.subplot(n_plot, 1, plot_n)
    plt.plot(times, ys[:, idx_r[0]], color='k', label='E')
    plt.plot(times, ys[:, idx_r[1]], color='r', label='I')
    plt.legend()
    plt.ylabel("R")
    plot_n += 1

    plt.subplot(n_plot, 1, plot_n)
    plt.plot(times, ys[:, idx_h[0]], label='EE')
    plt.plot(times, ys[:, idx_h[1]], label='EI')
    plt.plot(times, ys[:, idx_h[2]], label='IE')
    plt.plot(times, ys[:, idx_h[3]], label='II')
    plt.ylabel("G")
    plt.legend()
    plt.xlabel("Time (s)")

    pp.savefig()
    plt.close()


save_path = "/home/ejp/src/pacological/data/exp403"
with PdfPages(os.path.join(save_path, 'plots_403.pdf')) as pdf:
    names = range(0, 1499)
    for i in names:
        name = os.path.join(save_path, "{}.hdf5".format(i))
        plot(name, pdf)
