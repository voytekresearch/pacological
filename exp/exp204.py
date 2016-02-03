"""Compare oscillation onset (constant rate) to a halving the rate."""
import os
import sys
from pacological.lif import gain, exp
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from convenience.numpy import save_hdfz


path = sys.argv[1]

# --
f = 10
t = 1

n_trial = 30
Is = range(0, 26, 2)

rates = np.zeros((len(Is), 3))
gs = np.zeros((len(Is), 3))
vs = np.zeros((len(Is), 3))
sds = np.zeros((len(Is), 3))
for i in range(len(Is)):
    I = Is[i]

    rtmp1, vtmp1, sdtmp1, gtmp1 = [], [], [], []
    rtmp2, vtmp2, sdtmp2, gtmp2 = [], [], [], []
    rtmp3, vtmp3, sdtmp3, gtmp3 = [], [], [], []

    for trial in range(n_trial):
        res1 = exp(t, I, 1, f=0)
        res2 = exp(t, I, 0.5, f=0)
        res3 = exp(t, I, 1, f=f)

        # --
        spikes1 = res1['spikes']
        traces1 = res1['traces']

        spikes2 = res2['spikes']
        traces2 = res2['traces']

        spikes3 = res3['spikes']
        traces3 = res3['traces']

        gtmp1.append(np.mean(traces1.g_e_[0] - traces1.g_i_[0]))
        vtmp1.append(np.mean(traces1.v_[0]))
        sdtmp1.append(np.std(traces1.v_[0]))
        rtmp1.append(np.mean(spikes1.t_[:].shape[0] / t))

        gtmp2.append(np.mean(traces2.g_e_[0] - traces2.g_i_[0]))
        vtmp2.append(np.mean(traces2.v_[0]))
        sdtmp2.append(np.std(traces2.v_[0]))
        rtmp2.append(np.mean(spikes2.t_[:].shape[0] / t))

        gtmp3.append(np.mean(traces3.g_e_[0] - traces3.g_i_[0]))
        vtmp3.append(np.mean(traces3.v_[0]))
        sdtmp3.append(np.std(traces3.v_[0]))
        rtmp3.append(np.mean(spikes3.t_[:].shape[0] / t))

    rates[i, 0] = np.mean(rtmp1)
    rates[i, 1] = np.mean(rtmp2)
    rates[i, 2] = np.mean(rtmp3)

    gs[i, 0] = np.mean(gtmp1)
    gs[i, 1] = np.mean(gtmp2)
    gs[i, 2] = np.mean(gtmp3)

    vs[i, 0] = np.mean(vtmp1)
    vs[i, 1] = np.mean(vtmp2)
    vs[i, 2] = np.mean(vtmp3)

    sds[i, 0] = np.mean(sdtmp1)
    sds[i, 1] = np.mean(sdtmp2)
    sds[i, 2] = np.mean(sdtmp3)

    print(I)

save_hdfz(os.path.join(path, 'versus'), Is=Is, rates=rates, gs=gs,
          vs=vs, vs_sd=sds, f=f)
