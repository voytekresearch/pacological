import numpy as np
from pacological.pars import BMparams

# For n we are summing over all e-types
pops = [('L1_E', {'n': 800, 'type': 'E'}), ('L1_I', {'n': 200, 'type': 'I'})]

backs = [
    # L3
    ('L1_E', {'f': 0,
              'r_e': 135,
              'r_i': 135,
              'w_e': 4e-9,
              'w_i': 11.6e-8,
              'tau_e': 5e-3,
              'tau_i': 10e-3}),
    ('L1_I', {'f': 0,
              'r_e': 135,
              'r_i': 135,
              'w_e': 4e-9,
              'w_i': 1.6e-8,
              'tau_e': 5e-3,
              'tau_i': 10e-3}),
]

conns = [
    # Layer 1 -------------------------------------------------------
    # Internal
    ('L1_E', 'L1_E', {'tau_decay': 5e-3,
                      'tau_decay_std': 1e-3,
                      'w': 2e-9,
                      'w_std': .11e-9,
                      'c': 5.0,
                      'c_std': 0.1,
                      'p': 0.02}),
    ('L1_E', 'L1_I', {'tau_decay': 5e-3,
                      'tau_decay_std': 1e-3,
                      'w': 50e-9,
                      'w_std': .11e-9,
                      'c': 5.0,
                      'c_std': 0.1,
                      'p': 0.02}),
    ('L1_I', 'L1_E', {'tau_decay': 10e-3,
                      'tau_decay_std': 1e-3,
                      'w': 20e-9,
                      'w_std': .11e-9,
                      'c': 5.0,
                      'c_std': 0.1,
                      'p': 0.02}),
    ('L1_I', 'L1_I', {'tau_decay': 10e-3,
                      'tau_decay_std': 1e-3,
                      'w': 10e-9,
                      'w_std': .11e-9,
                      'c': 5.0,
                      'c_std': 0.1,
                      'p': 0.02})
]

pars = BMparams(pops, conns, backs, ['L1_E'], ['L1_E'])
