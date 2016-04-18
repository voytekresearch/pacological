import numpy as np


def perturb_params(BMparams, attr, sd=None, prng=None, seed=None):
    if prng is None:
        prng = np.random.RandomState(seed)

    X = getattr(BMparams, attr)

    try:
        SD = getattr(BMparams, sd)
    except TypeError:
        if sd is None:
            SD = X * 0.10
        elif sd.isdigit():
            STD = std
        else:
            print("std must be none, a number, or an attr of BMparams")
    # import ipdb; ipdb.set_trace()
    
    SD[SD == 0] = 1e-30
    SD = prng.normal(0, scale=SD, size=X.shape)
    assert X.shape == SD.shape, "X and SD are not the same shape."

    setattr(BMparams, attr, X + SD)

    return BMparams

        
class BMparams(object):
    # Fast effective membrane conductance; try 20 too?
    tau_m = 2e-3  
    Vth = -55e-3  # volts
    Vampa = 0 
    Vgaba = -80e-3

    def __init__(self, pops, conns, stim_i, back_i, 
            rbe=135e2, rbi=135e2, wbe=400e-9, wbi=1600e-9, 
            tau_e=5e-3, tau_i=10e-3, sigma=1,
            I_e=5e-12, I_i=3e-12, re0=8.0, ri0=12.0):
        
        # -- Args
        self.pops = pops
        self.conns = conns
        self.stim_i = stim_i # Processed to idx later
        self.back_i = back_i
        self.rbe = rbe 
        self.rbi = rbi 
        self.wbe = wbe 
        self.wbi = wbi 
        self.tau_e = tau_e 
        self.tau_i = tau_i 
        self.sigma = sigma
        self.I_e = I_e
        self.I_i = I_i
        self.re0 = re0
        self.ri0 = ri0

        # -- Derived
        n_pop = len(pops)
        self.n_pop = n_pop

        self.names = [n for n, _ in pops]

        self.codes = {}
        for i, p in enumerate(self.pops):
            self.codes[p[0]] = i

        self.stim_i = [self.codes[s] for s in self.stim_i]
        self.back_i = [self.codes[s] for s in self.back_i]
        
        # Define background matrice
        self.Kbe = np.ones(n_pop) * self.tau_e 
        self.Kbi = np.ones(n_pop) * self.tau_i
        self.Cbe = np.ones(n_pop) 
        self.Cbi = np.ones(n_pop) 
        self.Wbe = np.ones(n_pop) * self.wbe 
        self.Wbi = np.ones(n_pop) * self.wbi
        
        # Init the connectivity matrix
        self.Z = np.zeros((n_pop, n_pop), dtype=np.int)

        # Init the weight matrices
        self.W = np.zeros((n_pop, n_pop))
        self.Wstd = np.zeros((n_pop, n_pop))

        # Init the connection type E (1) or I (-1)
        self.T = np.zeros((n_pop, n_pop), dtype=np.int)

        # Init time constant matrix
        self.K = np.zeros((n_pop, n_pop))
        self.Kstd = np.zeros((n_pop, n_pop))
        
        # Init the number of synapses per connection
        self.CS = np.zeros((n_pop, n_pop), dtype=np.int)
        self.CSstd = np.zeros((n_pop, n_pop), dtype=np.int)
        
        # Init the TOTAL number of synapses
        self.C = np.zeros((n_pop, n_pop), dtype=np.int)
        self.Cstd = np.zeros((n_pop, n_pop), dtype=np.int)

        for p1, p2, pr in conns:
            i = self.codes[p1]
            j = self.codes[p2]
            
            self.Z[i, j] = 1
            
            self.W[i, j] = pr['w']
            self.Wstd[i, j] = pr['w_std']
            
            self.K[i, j] = pr['tau_decay']
            self.Kstd[i, j] = pr['tau_decay_std']

            self.C[i, j] = int(pops[i][1]['n'] * pr['p']  * pr['c'])
            self.Cstd[i, j] = int(pops[i][1]['n'] * pr['p'] * pr['c_std'])

            self.CS[i, j] = int(pr['c'])
            self.CSstd[i, j] = int(pr['c_std'])
            
            if self.pops[i][1]['type'] == 'E':
                self.T[i, j] = 1
            elif self.pops[i][1]['type'] == 'I':
                self.T[i, j] = -1
            else:
                raise ValueError("p1 not a known cell type")
        
        # Use T to define synaptic voltages
        self.V = np.zeros((n_pop, n_pop))
        self.V[self.T == 1] = self.Vampa - self.Vth
        self.V[self.T == -1] = self.Vgaba - self.Vth

        # Intial rates
        R0 = []
        for pop in pops:
            if pop[1]['type'] == 'E':
                R0.append(self.re0)
            elif pop[1]['type'] == 'I':
                R0.append(self.ri0)
            else:
                raise ValueError("type must be E or I")
        self.R0 = np.asarray(R0)

