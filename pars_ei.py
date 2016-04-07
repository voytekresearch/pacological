import numpy as np
from pacological.pars import BMparams

# For n we are summing over all e-types
pops = [
    ('L1_E', {'n': 8000, 'type' : 'E'}),
    ('L1_I', {'n': 2000, 'type' : 'I'})
]

conns = [
    # Layer 1 -------------------------------------------------------
    # Internal
    (
        'L1_E', 'L1_E', 
        {   'tau_decay' : 5e-3, 
            'tau_decay_std' : 1e-3, 
            'w' : 2e-9, 
            'w_std' : .11e-9, 
            'c' : 1.0, 
            'c_std' : 0.1, 
            'p': 0.02
        }
    ),
    (
        'L1_E', 'L1_I', 
        {   'tau_decay' : 5e-3, 
            'tau_decay_std' : 1e-3, 
            'w' : 50e-9, 
            'w_std' : .11e-9, 
            'c' : 1.0, 
            'c_std' : 0.1, 
            'p': 0.02
        }
    ),
    (
        'L1_I', 'L1_E', 
        {   'tau_decay' : 10e-3, 
            'tau_decay_std' : 1e-3, 
            'w' : 20e-9, 
            'w_std' : .11e-9, 
            'c' : 1.0, 
            'c_std' : 0.1, 
            'p': 0.02
        }
    ),
    (
        'L1_I', 'L1_I', 
        {   'tau_decay' : 10e-3, 
            'tau_decay_std' : 1e-3, 
            'w' : 10e-9, 
            'w_std' : .11e-9, 
            'c' : 1.0, 
            'c_std' : 0.1, 
            'p': 0.02
        }
    )
]

pars = BMparams(pops, conns, ['L1_E'], ['L1_E'], 
        [135e1, 135e1], [400e-9, 1600e-9])

