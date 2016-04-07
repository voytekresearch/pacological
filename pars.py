import numpy as np


class BMparams(object):
    # Background synapses
    Nb = 10000
    Cb = np.array([Nb * 0.8 * 0.02, Nb * 0.2 * 0.02])
    Kb = np.array([5e-3, 10e-3])

    # Fast effective membrane conductance; try 20 too?
    tau_m = 2e-3  

    Vth = -55e-3  # volts
    Vampa = 0 
    Vgaba = -80e-3

    re0 = 8.0
    ri0 = 12.0

    # I_bias
    I_e = 400e-9
    I_i = 300e-9

    def __init__(self, pops, conns, stim_i, back_i, Rb, Wb, sigma=10):
        
        # -- Args
        self.pops = pops
        self.conns = conns
        self.stim_i = stim_i # Processed to idx later
        self.back_i = back_i
        self.Rb = np.asarray(Rb) 
        self.Wb = np.asarray(Wb)
        self.sigma = sigma

        # -- Derived
        n_pop = len(pops)
        self.n_pop = n_pop

        self.names = [n for n, _ in pops]

        self.codes = {}
        for i, p in enumerate(self.pops):
            self.codes[p[0]] = i

        self.stim_i = [self.codes[s] for s in self.stim_i]
        self.back_i = [self.codes[s] for s in self.back_i]

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

        # Init the number of synapses
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

