import numpy as np
from pacological.pars import BMparams

# For n we are summing over all e-types
pops = [('L1_E', {'n': 800,
                  'type': 'E',
                  'r_0': 8}), ('L1_I', {'n': 200,
                                        'type': 'I',
                                        'r_0': 12})]

inputs = [('L1_E', {'w': 400e-3, 'c': 1, 'tau_decay': 5e-3, 'type': 'E'})]

backs = [
    # L3
    ('L1_E', {'f': 10,
              'r_e': 135,
              'r_i': 135,
              'w_e': 4e-9,
              'w_i': 16e-9,
              'tau_e': 5e-3,
              'tau_i': 10e-3}),
    ('L1_I', {'f': 0,
              'r_e': 135,
              'r_i': 135,
              'w_e': 4e-9,
              'w_i': 16e-9,
              'tau_e': 5e-3,
              'tau_i': 10e-3}),
]

conns = [
    # Layer 1 -------------------------------------------------------
    # Internal
    ('L1_E', 'L1_E', {'tau_decay': 5e-3,
                      'tau_decay_std': 1e-3,
                      'w': 50e-3,
                      'w_std': .11e-9,
                      'c': 1.0,
                      'c_std': 0.1,
                      'p': 0.02}),
    ('L1_E', 'L1_I', {'tau_decay': 5e-3,
                      'tau_decay_std': 1e-3,
                      'w': 100e-3,
                      'w_std': .11e-9,
                      'c': 1.0,
                      'c_std': 0.1,
                      'p': 0.02}),
    ('L1_I', 'L1_E', {'tau_decay': 10e-3,
                      'tau_decay_std': 1e-3,
                      'w': 500e-3,
                      'w_std': .11e-9,
                      'c': 1.0,
                      'c_std': 0.1,
                      'p': 0.02}),
    ('L1_I', 'L1_I', {'tau_decay': 10e-3,
                      'tau_decay_std': 1e-3,
                      'w': 100e-3,
                      'w_std': .11e-9,
                      'c': 1.0,
                      'c_std': 0.1,
                      'p': 0.02})
]

pars = BMparams(pops, conns, backs, inputs, sigma=0, background_res=0)
