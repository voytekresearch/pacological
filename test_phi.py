from pacological.bluemass import phi
import numpy as np


# Params
L = 1
gb_0 = 1e-7
gb_syn = 9.995e-09
sigma = 1.166e-06

I_e = 2e-9
I_i = 1e-9
Is = np.arange(I_i / 2, I_i * 2, 100)

rates = phi(Is, I_e, gb_0, gb_syn, sigma)
