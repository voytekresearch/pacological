import numpy as np


# For n we are summing over all e-types
pops = [('L1_E', {'n': 1000,
                  'type': 'E',
                  'r_0': 8, 'bias' : 3e-3}), ('L1_I', {'n': 1000,
                                        'type': 'I',
                                        'r_0': 12, 'bias' : 30e-3})]

inputs = [('L1_E', {'w': 3e-3 ,
                    'c': 1,
                    'n': 500,
                    'p': 0.1,
                    'tau_decay': 5e-3,
                    'type': 'E'})]

backs = [
    # L3
    ('L1_E', {'f': 10,
              'n_bursts': None,
              'min_r': 30,
              'r_e': 135,
              'r_i': 135,
              'w_e': 4e-9,
              'w_i': 16e-9,
              'tau_e': 5e-3,
              'tau_i': 10e-3}),
    ('L1_I', {'f': 0,
              'n_bursts': None,
              'min_r': 30,
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
                      'w': 2e-3,
                      'w_std': .11e-9,
                      'c': 1.0,
                      'c_std': 0.1,
                      'p': 0.1}),
    ('L1_E', 'L1_I', {'tau_decay': 5e-3,
                      'tau_decay_std': 1e-3,
                      'w': 2e-3,
                      'w_std': .11e-9,
                      'c': 1.0,
                      'c_std': 0.1,
                      'p': 0.1}),
    ('L1_I', 'L1_E', {'tau_decay': 10e-3,
                      'tau_decay_std': 1e-3,
                      'w': 2e-3,
                      'w_std': .11e-9,
                      'c': 1.0,
                      'c_std': 0.1,
                      'p': 0.1}),
    ('L1_I', 'L1_I', {'tau_decay': 10e-3,
                      'tau_decay_std': 1e-3,
                      'w': 2e-3,
                      'w_std': .11e-9,
                      'c': 1.0,
                      'c_std': 0.1,
                      'p': 0.1})
]

