from pathological import pac
import seaborn as sns
import matplotlib.pyplot as plt

spikes = pac.Spikes(10, 1)
times = spikes.times

osc = pac.osc(times, 30, 10)
stim = pac.stim(times, 30, 1)
gain = pac.gain_pac(times, 30, 10, 30, 1)

osc_sp = spikes.poisson(osc)
stim_sp = spikes.poisson(stim)
gain_sp = spikes.poisson(gain)

