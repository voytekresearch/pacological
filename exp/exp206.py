"""How does gain scale with oscillation f"""
import os
import sys
from pacological.lif import gain, exp
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from convenience.numpy import save_hdfz

path = sys.argv[1]

# --
f = 0
t = 1

n_trial = 20
Is = range(0, 25, 5)
xs = [1, 2, 3]

rates1 = np.zeros((len(Is), len(xs)))
gs1 = np.zeros((len(Is), len(xs)))
vs1 = np.zeros((len(Is), len(xs)))

rates2 = np.zeros((len(Is), len(xs)))
gs2 = np.zeros((len(Is), len(xs)))
vs2 = np.zeros((len(Is), len(xs)))

for i in range(len(Is)):
    I = Is[i]
    for j in range(len(xs)):
        x = xs[j]
        
        xfactor1 = x
        xfactor2 = (1.6 * x, 0.4 * x)

        rtmp1, vtmp1, gtmp1 = [], [], []
        rtmp2, vtmp2, gtmp2 = [], [], []
        
        for trial in range(n_trial):
            res1 = exp(t, I, xfactor1, f=f)
            res2 = exp(t, I, xfactor2, f=f)

            # --
            spikes1 = res1['spikes']
            traces1 = res1['traces']
            spikes2 = res2['spikes']
            traces2 = res2['traces']

            gtmp1.append(np.mean(traces1.g_e_[0] - traces1.g_i_[0]))
            vtmp1.append(np.mean(traces1.v_[0]))
            rtmp1.append(np.mean(spikes1.t_[:].shape[0] / t))

            gtmp2.append(np.mean(traces2.g_e_[0] - traces2.g_i_[0]))
            vtmp2.append(np.mean(traces2.v_[0]))
            rtmp2.append(np.mean(spikes2.t_[:].shape[0] / t))

        rates1[i, j] = np.mean(rtmp1)
        gs1[i, j] = np.mean(gtmp1)
        vs1[i, j] = np.mean(vtmp1)

        rates2[i, j] = np.mean(rtmp2)
        gs2[i, j] = np.mean(gtmp2)
        vs2[i, j] = np.mean(vtmp2)
    
    print(I)
    
save_hdfz(os.path.join(path, 'bal'), rates1=rates1, gs1=gs1, vs1=vs1, 
        rates2=rates2, gs2=gs2, vs2=vs2, xs=xs, f=f)

