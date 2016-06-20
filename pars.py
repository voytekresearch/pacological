import numpy as np


def perturb_params(BMparams, attr, sd, idx=None, prng=None, seed=None):
    """Perturb BM parameters with Gaussian noise.

    Params
    ------
    BMparams : object
        A BMparams object
    attr : str
        Name of the parameter matrix to perturb
    sd : str, numeric, None
        Standard deviation of the noise. Either a name
        of a parameter matrix, a scalar.
    idx : array-like
        Only perturb the indexed elements of attr.
    prng : None, RandomState
        A random state object
    seed : numeric
        Seed value for RandomState (only used if prng is None)
    """

    if prng is None:
        prng = np.random.RandomState(seed)

    X = getattr(BMparams, attr)

    try:
        SD = getattr(BMparams, sd)
    except TypeError:
        if sd.isdigit():
            STD = std
        else:
            print("std must be number, or an attr of BMparams")

    SD[SD == 0] = 1e-30
    SD[idx] = prng.normal(0, scale=SD, size=X.shape)[idx]

    setattr(BMparams, attr, X + SD)

    return BMparams


class BMparams(object):
    # Try Fast effective membrane conductance 2 ms?
    tau_m = 20e-3  # seconds
    Vth = -55e-3  # volts
    Vampa = 0
    Vgaba = -80e-3

    def __init__(self,
                 pops,
                 conns,
                 backs,
                 inputs,
                 sigma=1,
                 t_back=20,
                 I_max=200e-3,
                 background_res=0,
                 neuron='lif'):

        # -- Args
        self.pops = pops
        self.conns = conns
        self.backs = backs
        self.inputs = inputs

        self.sigma = sigma
        self.I_max = I_max

        self.t_back = t_back
        self.background_res = background_res

        # -- Derived
        n_pop = len(pops)
        self.n_pop = n_pop

        # Init names and indices
        self.names = [n for n, _ in pops]
        self.codes = {}
        for i, p in enumerate(self.pops):
            self.codes[p[0]] = i

        # Init the external connectivity matrix
        self.Zi = np.zeros(n_pop)  # Connecitvity
        self.Wi = np.zeros(n_pop)  # Weights
        self.Ci = np.zeros(n_pop)  # Connection number
        self.Ki = np.zeros(n_pop)  # Tau
        self.Ti = np.zeros(n_pop)  # Cell type

        # Init the network connectivity matrix
        self.Z = np.zeros((n_pop, n_pop), dtype=np.int)  # Connectivity
        self.W = np.zeros((n_pop, n_pop))  # Weight
        self.Wstd = np.zeros((n_pop, n_pop))
        self.T = np.zeros((n_pop, n_pop), dtype=np.int)  # Cell type
        self.K = np.zeros((n_pop, n_pop))  # Tau
        self.Kstd = np.zeros((n_pop, n_pop))
        self.CS = np.zeros((n_pop, n_pop), dtype=np.int)  # Connection number
        self.CSstd = np.zeros((n_pop, n_pop), dtype=np.int)
        self.C = np.zeros((n_pop, n_pop), dtype=np.int)
        self.Cstd = np.zeros((n_pop, n_pop), dtype=np.int)
        self.V = np.zeros((n_pop, n_pop))

        # Populate network
        for p1, p2, pr in conns:
            i = self.codes[p1]
            j = self.codes[p2]

            self.Z[i, j] = 1

            self.W[i, j] = pr['w']
            self.Wstd[i, j] = pr['w_std']

            self.K[i, j] = pr['tau_decay']
            self.Kstd[i, j] = pr['tau_decay_std']

            self.C[i, j] = int(pops[i][1]['n'] * pr['p'] * pr['c'])
            self.Cstd[i, j] = int(pops[i][1]['n'] * pr['p'] * pr['c_std'])

            self.CS[i, j] = int(pr['c'])
            self.CSstd[i, j] = int(pr['c_std'])

            if self.pops[i][1]['type'] == 'E':
                self.T[i, j] = 1
            elif self.pops[i][1]['type'] == 'I':
                self.T[i, j] = -1
            else:
                raise ValueError("Unknown cell type in pop {}.".format(p1))

        # Use T to define synaptic voltages
        self.V[self.T == 1] = self.Vampa - self.Vth
        self.V[self.T == -1] = self.Vgaba - self.Vth

        # Populate inputs
        for p1, pr in inputs:
            i = self.codes[p1]

            self.Zi[i] = 1
            self.Wi[i] = pr['w']
            self.Ki[i] = pr['tau_decay']
            self.Ci[i] = int(pr['c'])

            if pr['type'] == 'E':
                self.Ti[i] = 1
            elif pr['type'] == 'I':
                self.Ti[i] = -1
            else:
                raise ValueError("Unknown cell type in input {}.".format(p1))
