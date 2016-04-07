import numpy as np
from pacological.pars import BMparams

# A key question: What is the column computing?
# A key question: where is the background oscillation coming from?
# A key question: where is the stimulus coming from?

# Q: What is the output layer?
# A: Layers 5 and 6.

# Q: What is the input layer?    
# Layers 4 and 6.

# Setup the 18 populations and give them ----------------------------
# integer codes for easy indexing.
# For n we are summing over all e-types

pops = [
    # L3
    ('L3_PC', {'n' : 5877, 'type' : 'E'}),
    ('L3_LBC', {'n' : 22 + 108 + 76 + 188 + 35 + 27, 'type' : 'I'}),
    ('L3_NBC', {'n' : 80+14+6+6+6+65+97, 'type' : 'I'}),
    # L4
    ('L4_PC', {'n' : 2674, 'type' : 'E'}),
    ('L4_SS', {'n' : 406, 'type' : 'E'}),  # in
    ('L4_SP', {'n' : 1098, 'type' : 'E'}),
    ('L4_LBC', {'n' : 22 + 31 + 46 + 9 + 14, 'type' : 'I'}),
    ('L4_NBC', {'n' : 36 + 45 + 10 +5, 'type' : 'I'}),
    # L5
    ('L5_TTPC1', {'n' : 2403, 'type' : 'E'}),  # out
    ('L5_TTPC2', {'n' : 2000, 'type' : 'E'}),  # out
    ('L5_LBC', {'n' : 13 + 37 + 25 + 49 + 37 + 12 +37, 'type' : 'I'}),
    ('L5_NBC', {'n' : 40 + 39 + 14 + 27 + 14 + 40 + 12 + 14, 'type' : 'I'}),
    # L6
    ('L6_BPC', {'n' : 3174, 'type' : 'E'}),  # TODO in where in 6?
    ('L6_IPC', {'n' : 3476, 'type' : 'E'}),
    ('L6_UTPC', {'n' : 1735, 'type' : 'E'}),
    ('L6_TCP_L1', {'n' : 1637, 'type' : 'E'}), 
    ('L6_TCP_L4', {'n' : 1440, 'type' : 'E'}),  
    ('L6_NBC', {'n' : 14 + 27 + 14 + 39 + 37 + 39 + 14 + 14, 'type' : 'I'}),
    ('L6_LBC', {'n' : 43 + 42 + 84 + 124 + 127 + 43, 'type' : 'I'})
]

# Define all conns as 3-tuples, (pop1, pop2, {params}) --------------
conns = [
    # Layer 3 -------------------------------------------------------
    # Internal
    (
        'L3_PC', 'L3_PC', 
        {   'tau_decay' : 47e-3, 
            'tau_decay_std' : 5.2e-3, 
            'w' : 0.3e-9, 
            'w_std' : .11e-9, 
            'c' : 2.8, 
            'c_std' : 1.2, 
            'p': 0.056
        }
    ),
    (
        'L3_PC', 'L3_LBC', 
        {   'tau_decay' : 24e-3, 
            'tau_decay_std' : 16e-3, 
            'w' : 0.3e-9, 
            'w_std' : .1e-9, 
            'c' : 8.1, 
            'c_std' : 3.1, 
            'p': 0.051
        }
    ),  
    (
        'L3_LBC', 'L3_PC', 
        {   'tau_decay' : 48e-3, 
            'tau_decay_std' : 13e-3, 
            'w' : 1.1e-9, 
            'w_std' : 1.4e-9, 
            'c' : 17, 
            'c_std' :6.5 , 
            'p': 0.05 
        }
    ),  
    (
        'L3_LBC', 'L3_LBC',                           
        {   'tau_decay' : 73e-3, 
            'tau_decay_std' : 54e-3, 
            'w' : 0.33e-9, 
            'w_std' : 0.15e-9, 
            'c' : 15, 
            'c_std' : 7, 
            'p': 0.051 
        }
    ),  
    (
        'L3_PC', 'L3_NBC', 
        {   'tau_decay' : 17e-3, 
            'tau_decay_std' : 12e-3, 
            'w' : 0.31e-9, 
            'w_std' : 0.11e-9, 
            'c' : 3.8, 
            'c_std' : 1.6, 
            'p': 0.066
        }
    ),  
    (
        'L3_NBC', 'L3_PC', 
        {   'tau_decay' : 48e-3, 
            'tau_decay_std' : 14e-3, 
            'w' : 1.4e-9, 
            'w_std' : 1.6e-9, 
            'c' : 17, 
            'c_std' : 5.2, 
            'p': 0.10 
        }
    ),  
    (
        'L3_NBC', 'L3_NBC', 
        {   'tau_decay' : 83e-3, 
            'tau_decay_std' : 59e-3, 
            'w' : 0.31e-9, 
            'w_std' : 0.11e-9, 
            'c' : 16, 
            'c_std' : 6.3, 
            'p': 0.094 
        }
    ),  
    # External E -> E and E -> I
    # to L4
    (
        'L3_PC', 'L4_PC', 
        {   'tau_decay' : 52e-3, 
            'tau_decay_std' : 5.3e-3, 
            'w' : 0.31e-9, 
            'w_std' : .11e-9, 
            'c' : 2.6, 
            'c_std' : 1.1, 
            'p': 0.066
        }
    ),
    (
        'L3_PC', 'L4_SS', 
        {   'tau_decay' : 55e-3, 
            'tau_decay_std' : 2.7e-3, 
            'w' : 0.31e-9, 
            'w_std' : .11e-9, 
            'c' : 2.5, 
            'c_std' : 0.94, 
            'p': 0.053
        }
    ),
    (
        'L3_PC', 'L4_LBC', 
        {   'tau_decay' : 22e-3, 
            'tau_decay_std' : 12e-3, 
            'w' : 0.3e-9, 
            'w_std' : .11e-9, 
            'c' : 6.9, 
            'c_std' : 2.5, 
            'p': 0.029
        }
    ),
    (
        'L3_PC', 'L4_NBC', 
        {   'tau_decay' : 17e-3, 
            'tau_decay_std' : 13e-3, 
            'w' : 0.3e-9, 
            'w_std' : .11e-9, 
            'c' : 6.7, 
            'c_std' : 2.6, 
            'p': 0.031
        }
    ),
    # to L5
    (
        'L3_PC', 'L5_TTPC1',   
        {   'tau_decay' : 71e-3, 
            'tau_decay_std' : 28e-3, 
            'w' : 0.29e-9, 
            'w_std' : .11e-9, 
            'c' : 3.8, 
            'c_std' : 1.6, 
            'p': 0.078
        }
    ),
    (
        'L3_PC', 'L5_TTPC2',   
        {   'tau_decay' : 58e-3, 
            'tau_decay_std' : 14e-3, 
            'w' : 0.3e-9, 
            'w_std' : .1e-9, 
            'c' : 3.8, 
            'c_std' : 1.6, 
            'p': 0.082
        }
    ),
    (
        'L3_PC', 'L5_LBC', 
        {   'tau_decay' : 16e-3, 
            'tau_decay_std' : 14e-3, 
            'w' : 0.29e-9, 
            'w_std' : .11e-9, 
            'c' : 6.5, 
            'c_std' : 2.5, 
            'p': 0.016
        }
    ),
    (
        'L3_PC', 'L5_NBC', 
        {   'tau_decay' : 7.9e-3, 
            'tau_decay_std' : 5.1e-3, 
            'w' : 0.3e-9, 
            'w_std' : .1e-9, 
            'c' : 5.8, 
            'c_std' : 2.5, 
            'p': 0.014
        }
    ),
    # to L5 I - > 
    # See Fgiure s^ in Markram for justification
    (
        'L3_LBC', 'L5_TTPC1', 
        {   'tau_decay' : 130e-3, 
            'tau_decay_std' : 48e-3, 
            'w' : 1.2e-9, 
            'w_std' : 1.5e-9, 
            'c' : 16.5, 
            'c_std' : 6.1, 
            'p': 0.033
        }
    ),
    (
        'L3_LBC', 'L5_TTPC2', 
        {   'tau_decay' : 120e-3, 
            'tau_decay_std' : 42e-3, 
            'w' : 1.1e-9, 
            'w_std' : 1.4e-9, 
            'c' : 16, 
            'c_std' : 6.5, 
            'p': 0.033
        }
    ),
    (
        'L3_NBC', 'L5_TTPC1', 
        {   'tau_decay' : 140e-3, 
            'tau_decay_std' : 47e-3, 
            'w' : 1.2e-9, 
            'w_std' : 1.5e-9, 
            'c' : 13, 
            'c_std' : 4.8, 
            'p': 0.043
        }
    ),
    (
        'L3_NBC', 'L5_TTPC2', 
        {   'tau_decay' : 120e-3, 
            'tau_decay_std' : 42e-3, 
            'w' : 1.4e-9, 
            'w_std' : 1.6e-9, 
            'c' : 14, 
            'c_std' : 5, 
            'p': 0.04
        }
    ),
    # to L6
    (
        'L3_PC', 'L6_BPC',
        {   'tau_decay' : 56e-3, 
            'tau_decay_std' : 19e-3, 
            'w' : 0.3e-9, 
            'w_std' : 0.1e-9, 
            'c' : 2.4, 
            'c_std' : .92, 
            'p': 0.014
        }
    ),
    (
        'L3_PC', 'L6_IPC',
        {   'tau_decay' : 45e-3, 
            'tau_decay_std' : 6.5e-3, 
            'w' : 0.3e-9, 
            'w_std' : 0.1e-9, 
            'c' : 2.3, 
            'c_std' : 0.81, 
            'p': 0.0028
        }
    ),

    (
        'L3_PC', 'L6_UTPC',
        {   'tau_decay' : 70e-3, 
            'tau_decay_std' : 30e-3, 
            'w' : 0.3e-9, 
            'w_std' : 0.1e-9, 
            'c' : 2.5, 
            'c_std' : 0.97, 
            'p': 0.002
        }
    ),
    (
        'L3_PC', 'L6_TCP_L1',
        {   'tau_decay' : 57e-3, 
            'tau_decay_std' : 17e-3, 
            'w' : 0.3e-9, 
            'w_std' : 0.1e-9, 
            'c' : 2.4, 
            'c_std' : 0.96, 
            'p': 0.016
        }
    ),
    (
        'L3_PC', 'L6_TCP_L4',
        {   'tau_decay' : 57e-3, 
            'tau_decay_std' : 12e-3, 
            'w' : 0.3e-9, 
            'w_std' : 0.1e-9, 
            'c' : 2.4, 
            'c_std' : 1, 
            'p': 0.021
        }
    ),
    (
        'L3_PC', 'L6_NBC',
        {   'tau_decay' : 8.2e-3, 
            'tau_decay_std' : 4.8e-3, 
            'w' : 0.3e-9, 
            'w_std' : 0.1e-9, 
            'c' : 4.5, 
            'c_std' : 2, 
            'p': 0.0022
        }
    ),
    (
        'L3_PC', 'L6_LBC',
        {   'tau_decay' : 16e-3, 
            'tau_decay_std' : 26e-3, 
            'w' : 0.3e-9, 
            'w_std' : 0.1e-9, 
            'c' : 4.9, 
            'c_std' : 2.2, 
            'p': 0.0026
        }
    ),
    # Layer 4 ---------------------------------------------------------
    # Internal
    # PCs
    (
        'L4_PC', 'L4_PC', 
        {   'tau_decay' : 61e-3, 
            'tau_decay_std' : 1.9e-3, 
            'w' : 0.29e-9, 
            'w_std' : .11e-9, 
            'c' : 3.3, 
            'c_std' : 1.4, 
            'p': 0.076
        }
    ),
    (
        'L4_PC', 'L4_SS', 
        {   'tau_decay' : 67e-3, 
            'tau_decay_std' : 0.96e-3, 
            'w' : 0.29e-9, 
            'w_std' : .11e-9, 
            'c' : 3.2, 
            'c_std' : 1.3, 
            'p': 0.065
        }
    ),
    (
        'L4_PC', 'L4_SP', 
        {   'tau_decay' : 62e-3, 
            'tau_decay_std' : 1.6e-3, 
            'w' : 0.29e-9, 
            'w_std' : .11e-9, 
            'c' : 3.5, 
            'c_std' : 1.4, 
            'p': 0.064
        }
    ),
    (
        'L4_PC', 'L4_LBC', 
        {   'tau_decay' : 25e-3, 
            'tau_decay_std' : 9.7e-3, 
            'w' : 0.29e-9, 
            'w_std' : .11e-9, 
            'c' : 7.9, 
            'c_std' : 3, 
            'p': 0.042
        }
    ),
    # LBCs
    (
        'L4_LBC', 'L4_PC', 
        {   'tau_decay' : 61e-3, 
            'tau_decay_std' : 19e-3, 
            'w' : 0.89e-9, 
            'w_std' : 1.3e-9, 
            'c' : 16, 
            'c_std' : 6.2, 
            'p': 0.063
        }
    ),
    (
        'L4_LBC', 'L4_LBC', 
        {   'tau_decay' : 160e-3, 
            'tau_decay_std' : 98e-3, 
            'w' : 0.33e-9, 
            'w_std' : 0.15e-9, 
            'c' : 14, 
            'c_std' : 6, 
            'p': 0.062
        }
    ),
    (
        'L4_LBC', 'L4_SS', 
        {   'tau_decay' : 62e-3, 
            'tau_decay_std' : 19e-3, 
            'w' : 1.1e-9, 
            'w_std' : 1.4e-9, 
            'c' : 16, 
            'c_std' : 5.7, 
            'p': 0.084
        }
    ),
    (
        'L4_LBC', 'L4_SP', 
        {   'tau_decay' : 52e-3, 
            'tau_decay_std' : 14e-3, 
            'w' : 1.2e-9, 
            'w_std' : 1.5e-9, 
            'c' : 16, 
            'c_std' : 6.2, 
            'p': 0.074
        }
    ),
    # NBCs
    (
        'L4_NBC', 'L4_NBC', 
        {   'tau_decay' : 110e-3, 
            'tau_decay_std' : 63e-3, 
            'w' : 0.34e-9, 
            'w_std' : 0.15e-9, 
            'c' : 19, 
            'c_std' : 7.5, 
            'p': 0.067
        }
    ),
    (
        'L4_NBC', 'L4_SS', 
        {   'tau_decay' : 59e-3, 
            'tau_decay_std' : 15e-3, 
            'w' : 0.64e-9, 
            'w_std' : 1.1e-9, 
            'c' : 19, 
            'c_std' : 7.4, 
            'p': 0.074
        }
    ),
    (
        'L4_NBC', 'L4_SP', 
        {   'tau_decay' : 50e-3, 
            'tau_decay_std' : 11e-3, 
            'w' : 0.53e-9, 
            'w_std' : 0.92e-9, 
            'c' : 21, 
            'c_std' : 8.2, 
            'p': 0.067
        }
    ),
    # SS - internal
    (
        'L4_SS', 'L4_SS', 
        {   'tau_decay' : 67e-3, 
            'tau_decay_std' : 11e-3, 
            'w' : 0.31e-9, 
            'w_std' : 0.1e-9, 
            'c' : 2.9, 
            'c_std' : 1.1, 
            'p': 0.051
        }
    ),
    (
        'L4_SS', 'L4_PC', 
        {   'tau_decay' : 60e-3, 
            'tau_decay_std' : 2.6e-3, 
            'w' : 0.3e-9, 
            'w_std' : 0.1e-9, 
            'c' : 3, 
            'c_std' : 1.2, 
            'p': 0.062
        }
    ),
    (
        'L4_SS', 'L4_LBC', 
        {   'tau_decay' : 23e-3, 
            'tau_decay_std' : 11e-3, 
            'w' : 0.3e-9, 
            'w_std' : 0.1e-9, 
            'c' : 8, 
            'c_std' : 4, 
            'p': 0.026
        }
    ),
    (
        'L4_SS', 'L4_NBC', 
        {   'tau_decay' : 17e-3, 
            'tau_decay_std' : 12e-3, 
            'w' : 0.3e-9, 
            'w_std' : 0.11e-9, 
            'c' : 8.1, 
            'c_std' : 3.1, 
            'p': 0.03
        }
    ),
    # SS - external
    # to L3
    (
        'L4_SS', 'L3_PC', 
        {   'tau_decay' : 50e-3, 
            'tau_decay_std' : 6e-3, 
            'w' : 0.33e-9, 
            'w_std' : 0.16e-9, 
            'c' : 4.8, 
            'c_std' : 1.6, 
            'p': 0.0096
        }
    ),
    (
        'L4_SS', 'L3_LBC', 
        {   'tau_decay' : 22e-3, 
            'tau_decay_std' : 18e-3, 
            'w' : 0.33e-9, 
            'w_std' : 0.16e-9, 
            'c' : 6.1, 
            'c_std' : 1.9, 
            'p': 0.0088
        }
    ),
    # to L5
    (
        'L4_SS', 'L5_TTPC1', 
        {   'tau_decay' : 59e-3, 
            'tau_decay_std' : 13e-3, 
            'w' : 0.31e-9, 
            'w_std' : 0.1e-9, 
            'c' : 4.2, 
            'c_std' : 1.6, 
            'p': 0.084
        }
    ),
    (
        'L4_SS', 'L5_TTPC2',  
        {   'tau_decay' : 56e-3, 
            'tau_decay_std' : 14e-3, 
            'w' : 0.31e-9, 
            'w_std' : 0.1e-9, 
            'c' : 4.3, 
            'c_std' : 1.7, 
            'p': 0.085
        }
    ),
    (
        'L4_SS', 'L5_LBC',
        {   'tau_decay' : 21e-3, 
            'tau_decay_std' : 28e-3, 
            'w' : 0.3e-9, 
            'w_std' : 0.11e-9, 
            'c' : 7.8, 
            'c_std' : 2.9, 
            'p': 0.019
        }
    ),
    (
        'L4_SS', 'L5_NBC',  
        {   'tau_decay' : 8.2e-3, 
            'tau_decay_std' : 5.7e-3, 
            'w' : 0.3e-9, 
            'w_std' : 0.11e-9, 
            'c' : 7.1, 
            'c_std' : 3.1, 
            'p': 0.019
        }
    ),
    (
        'L4_LBC', 'L5_TTPC1',  
        {   'tau_decay' : 95e-3, 
            'tau_decay_std' : 36e-3, 
            'w' : 0.98e-9, 
            'w_std' : 1.3e-9, 
            'c' : 17, 
            'c_std' : 7.2, 
            'p': 0.056
        }
    ),
    # to L5 I -> 
    (
        'L4_LBC', 'L5_TTPC1',  
        {   'tau_decay' : 95e-3, 
            'tau_decay_std' : 36e-3, 
            'w' : 0.98e-9, 
            'w_std' : 1.3e-9, 
            'c' : 17, 
            'c_std' : 7.2, 
            'p': 0.056
        }
    ),
   (
        'L4_LBC', 'L5_TTPC2',  
        {   'tau_decay' : 88e-3, 
            'tau_decay_std' : 34e-3, 
            'w' : 0.9e-9, 
            'w_std' : 1.3e-9, 
            'c' : 17, 
            'c_std' : 7.2, 
            'p': 0.058
        }
    ),
    (
         'L4_NBC', 'L5_TTPC1',  
         {   'tau_decay' : 97e-3, 
             'tau_decay_std' : 37e-3, 
             'w' : 0.43e-9, 
             'w_std' : 0.7e-9, 
             'c' : 22, 
             'c_std' : 9.7, 
             'p': 0.045
         }
     ),
    (
         'L4_NBC', 'L5_TTPC2',  
         {   'tau_decay' : 90e-3, 
             'tau_decay_std' : 36e-3, 
             'w' : 1.1e-9, 
             'w_std' : 1.4e-9, 
             'c' : 23, 
             'c_std' : 112, 
             'p': 0.043
         }
     ),
    # to L6
    (
        'L4_PC', 'L6_BPC',
        {   'tau_decay' : 59e-3, 
            'tau_decay_std' : 24e-3, 
            'w' : 0.3e-9, 
            'w_std' : 0.1e-9, 
            'c' : 2.5, 
            'c_std' : 0.95, 
            'p': 0.032
        }
    ),
    (
        'L4_PC', 'L6_IPC',
        {   'tau_decay' : 46e-3, 
            'tau_decay_std' : 7e-3, 
            'w' : 0.3e-9, 
            'w_std' : 0.1e-9, 
            'c' : 2.4, 
            'c_std' : .82, 
            'p': 0.0081
        }
    ),
    (
        'L4_PC', 'L6_UTPC',
        {   'tau_decay' : 56e-3, 
            'tau_decay_std' : 12e-3, 
            'w' : 0.3e-9, 
            'w_std' : 0.1e-9, 
            'c' : 2.5, 
            'c_std' : 1, 
            'p': 0.031
        }
    ),
    (
        'L4_PC', 'L6_TCP_L1',
        {   'tau_decay' : 56e-3, 
            'tau_decay_std' : 12e-3, 
            'w' : 0.31e-9, 
            'w_std' : 0.1e-9, 
            'c' : 2.5, 
            'c_std' : 1, 
            'p': 0.031
        }
    ),
    (
        'L4_PC', 'L6_TCP_L4',
        {   'tau_decay' : 57e-3, 
            'tau_decay_std' : 11e-3, 
            'w' : 0.3e-9, 
            'w_std' : 0.1e-9, 
            'c' : 2.5, 
            'c_std' : 1.1, 
            'p': 0.044
        }
    ),
     (
        'L4_PC', 'L6_UTPC',
        {   'tau_decay' : 67e-3, 
            'tau_decay_std' : 32e-3, 
            'w' : 0.3e-9, 
            'w_std' : 0.1e-9, 
            'c' : 2.5, 
            'c_std' : 0.98, 
            'p': 0.004
        }
    ),
    (
        'L4_PC', 'L6_LBC',
        {   'tau_decay' : 17e-3, 
            'tau_decay_std' : 24e-3, 
            'w' : 0.3e-9, 
            'w_std' : 0.1e-9, 
            'c' : 5.5, 
            'c_std' : 2.2, 
            'p': 0.0063
        }
    ),
    (
        'L4_PC', 'L6_NBC',
        {   'tau_decay' : 7.7e-3, 
            'tau_decay_std' : 5e-3, 
            'w' : 0.3e-9, 
            'w_std' : 0.1e-9, 
            'c' : 5.4, 
            'c_std' : 2.4, 
            'p': 0.012
        }
    ),
    # Because they seem to make up a relatvily small fraction of cross-layer 
    # projections here L4_SP and L4_SS -> l^_* are not included. 
    # 
    # Layer 5 ---------------------------------------------------------
    # Internal
    (
        'L5_TTPC1', 'L5_TTPC1',  
        {   'tau_decay' : 58e-3, 
            'tau_decay_std' : 4.9e-3, 
            'w' : 0.31e-9, 
            'w_std' : 0.11e-9, 
            'c' : 6.2, 
            'c_std' : 2.6, 
            'p': 0.063
        }
    ),   
    (
        'L5_TTPC1', 'L5_TTPC2',    
        {   'tau_decay' : 55e-3, 
            'tau_decay_std' : 4.2e-3, 
            'w' : 0.31e-9, 
            'w_std' : 0.11e-9, 
            'c' : 6.1, 
            'c_std' : 2.6, 
            'p': 0.069
        }
    ), 
    (
        'L5_TTPC2', 'L5_TTPC2',   
        {   'tau_decay' : 55e-3, 
            'tau_decay_std' : 4.1e-3, 
            'w' : 0.31e-9, 
            'w_std' : 0.11e-9, 
            'c' : 6, 
            'c_std' : 2.4, 
            'p': 0.088
        }
    ), 
    (
        'L5_TTPC2', 'L5_TTPC1',  
        {   'tau_decay' : 58e-3, 
            'tau_decay_std' : 4.8e-3, 
            'w' : 0.31e-9, 
            'w_std' : 0.11e-9, 
            'c' : 6, 
            'c_std' : 2.4, 
            'p': 0.082
        }
    ), 
    # LBC
    (
        'L5_TTPC1', 'L5_LBC',  
        {   'tau_decay' : 19e-3, 
            'tau_decay_std' : 25e-3, 
            'w' : 0.3e-9, 
            'w_std' : 0.11e-9, 
            'c' : 8.2, 
            'c_std' : 3.6, 
            'p': 0.032
        }
    ), 
    (
        'L5_LBC', 'L5_TTPC1',  # doubled c to account for L5_TTPC2 
        {   'tau_decay' : 96e-3, 
            'tau_decay_std' : 37e-3, 
            'w' : 0.92e-9, 
            'w_std' : 1.3e-9, 
            'c' : 22, 
            'c_std' : 10, 
            'p': 0.064
        }
    ),      
    (
        'L5_LBC', 'L5_LBC',    
        {   'tau_decay' : 150e-3, 
            'tau_decay_std' : 100e-3, 
            'w' : 0.34e-9, 
            'w_std' : .15e-9, 
            'c' : 15, 
            'c_std' : 6.3, 
            'p': 0.047
        }
    ), 
    (
        'L5_TTPC2', 'L5_LBC',  
        {   'tau_decay' : 16e-3, 
            'tau_decay_std' : 13e-3, 
            'w' : 0.3e-9, 
            'w_std' : 0.11e-9, 
            'c' : 8.2, 
            'c_std' : 3.6, 
            'p': 0.042
        }
    ), 
    (
        'L5_LBC', 'L5_TTPC2', 
        {   'tau_decay' : 81e-3, 
            'tau_decay_std' : 31e-3, 
            'w' : 0.98e-9, 
            'w_std' : 1.3e-9, 
            'c' : 23, 
            'c_std' : 12, 
            'p': 0.078
        }
    ),      
    # NBC
    # (has no NBC-NBC recurrance!)
    (
        'L5_TTPC1', 'L5_NBC',  
        {   'tau_decay' : 8.1e-3, 
            'tau_decay_std' : 5.1e-3, 
            'w' : 0.3e-9, 
            'w_std' : 0.11e-9, 
            'c' : 3.7, 
            'c_std' : 1.5, 
            'p': 0.062
        }
    ), 
    (
        'L5_NBC', 'L5_TTPC1',  
        {   'tau_decay' : 95e-3, 
            'tau_decay_std' : 36e-3, 
            'w' : 1.7e-9, 
            'w_std' : 1.7e-9, 
            'c' : 24, 
            'c_std' : 12, 
            'p': 0.079
        }
    ),      
    (
        'L5_TTPC2', 'L5_NBC',  
        {   'tau_decay' : 9.8e-3, 
            'tau_decay_std' : 6.7e-3, 
            'w' : 0.3e-9, 
            'w_std' : 0.11e-9, 
            'c' : 3.7, 
            'c_std' : 1.5, 
            'p': 0.075
        }
    ), 
    (
        'L5_NBC', 'L5_TTPC2', 
        {   'tau_decay' : 81e-3, 
            'tau_decay_std' : 31e-3, 
            'w' : 0.98e-9, 
            'w_std' : 1.3e-9, 
            'c' : 25, 
            'c_std' : 12, 
            'p': 0.078
        }
    ),      
    # External
    # to L3
    (
        'L5_TTPC1', 'L3_PC',
        {   'tau_decay' : 46e-3, 
            'tau_decay_std' : 4.8e-3, 
            'w' : 0.3e-9, 
            'w_std' : 0.11e-9, 
            'c' : 2.3, 
            'c_std' : 0.74, 
            'p': 0.00073
        }
    ), 
    (
        'L5_TTPC1', 'L3_LBC',
        {   'tau_decay' : 21e-3, 
            'tau_decay_std' : 16e-3, 
            'w' : 0.3e-9, 
            'w_std' : 0.11e-9, 
            'c' : 5.3, 
            'c_std' : 1.9, 
            'p': 0.0003
        }
    ), 
    (
        'L5_TTPC1', 'L3_NBC',
        {   'tau_decay' : 21e-3, 
            'tau_decay_std' : 16e-3, 
            'w' : 0.3e-9, 
            'w_std' : 0.11e-9, 
            'c' : 5.3, 
            'c_std' : 1.9, 
            'p': 0.0003
        }
    ), 
    # to L4 
    # here
    (
        'L5_TTPC1', 'L4_PC',
        {   'tau_decay' : 51e-3, 
            'tau_decay_std' : 5.4e-3, 
            'w' : 0.29e-9, 
            'w_std' : 0.11e-9, 
            'c' : 2.5, 
            'c_std' : 0.89, 
            'p': 0.011
        }
    ), 
    (
        'L5_TTPC1', 'L4_SS',
        {   'tau_decay' : 55e-3, 
            'tau_decay_std' : 1.8e-3, 
            'w' : 0.3e-9, 
            'w_std' : 0.11e-9, 
            'c' : 2.4, 
            'c_std' : 0.79, 
            'p': 0.0094
        }
    ), 
    (
        'L5_TTPC1', 'L4_SP',
        {   'tau_decay' : 51e-3, 
            'tau_decay_std' : 3.4e-3, 
            'w' : 0.3e-9, 
            'w_std' : 0.11e-9, 
            'c' : 2.4, 
            'c_std' : 0.88, 
            'p': 0.0072
        }
    ), 
    (
        'L5_TTPC1', 'L4_LBC',
        {   'tau_decay' : 25e-3, 
            'tau_decay_std' : 12e-3, 
            'w' : 0.38e-9, 
            'w_std' : 0.21e-9, 
            'c' : 6.1, 
            'c_std' : 2.1, 
            'p': 0.0069
        }
    ),
    (
        'L5_TTPC1', 'L4_NBC',
        {   'tau_decay' : 18e-3, 
            'tau_decay_std' : 14e-3, 
            'w' : 0.38e-9, 
            'w_std' : 0.21e-9, 
            'c' : 5.9, 
            'c_std' : 2.1, 
            'p': 0.0045
        }
    ),
    (
        'L5_TTPC2', 'L4_PC',
        {   'tau_decay' : 51e-3, 
            'tau_decay_std' : 5.6e-3, 
            'w' : 0.3e-9, 
            'w_std' : 0.11e-9, 
            'c' : 2.6, 
            'c_std' : 1, 
            'p': 0.015
        }
    ), 
    (
        'L5_TTPC2', 'L4_SS',
        {   'tau_decay' : 55e-3, 
            'tau_decay_std' : 2.3e-3, 
            'w' : 0.3e-9, 
            'w_std' : 0.11e-9, 
            'c' : 2.4, 
            'c_std' : 0.89, 
            'p': 0.013
        }
    ), 
    (
        'L5_TTPC2', 'L4_SP',
        {   'tau_decay' : 51e-3, 
            'tau_decay_std' : 3.8e-3, 
            'w' : 0.3e-9, 
            'w_std' : 0.11e-9, 
            'c' : 2.6, 
            'c_std' : 0.97, 
            'p': 0.011
        }
    ), 
    (
        'L5_TTPC2', 'L4_LBC',
        {   'tau_decay' : 23e-3, 
            'tau_decay_std' : 12e-3, 
            'w' : 0.3e-9, 
            'w_std' : 0.1e-9, 
            'c' : 6.9, 
            'c_std' : 2.5, 
            'p': 0.022
        }
    ),
    (
        'L5_TTPC2', 'L4_NBC',
        {   'tau_decay' : 18e-3, 
            'tau_decay_std' : 13e-3, 
            'w' : 0.3e-9, 
            'w_std' : 0.1e-9, 
            'c' : 6.7, 
            'c_std' : 2.1, 
            'p': 0.0056
        }
    ),
    # to L6
    (
        'L5_TTPC1', 'L6_BPC',
        {   'tau_decay' : 51e-3, 
            'tau_decay_std' : 14e-3, 
            'w' : 0.3e-9, 
            'w_std' : 0.1e-9, 
            'c' : 3.4, 
            'c_std' : 1.5, 
            'p': 0.047
        }
    ),
    (
        'L5_TTPC1', 'L6_IPC',
        {   'tau_decay' : 44e-3, 
            'tau_decay_std' : 5.5e-3, 
            'w' : 0.3e-9, 
            'w_std' : 0.1e-9, 
            'c' : 3.5, 
            'c_std' : 1.5, 
            'p': 0.018
        }
    ),
    (
        'L5_TTPC1', 'L6_UTPC',
        {   'tau_decay' : 53e-3, 
            'tau_decay_std' : 13e-3, 
            'w' : 0.3e-9, 
            'w_std' : 0.1e-9, 
            'c' : 3.6, 
            'c_std' : 1.6, 
            'p': 0.053
        }
    ),
    (
        'L5_TTPC1', 'L6_TCP_L1',
        {   'tau_decay' : 46e-3, 
            'tau_decay_std' : 7.6e-3, 
            'w' : 0.3e-9, 
            'w_std' : 0.1e-9, 
            'c' : 3.6, 
            'c_std' : 1.6, 
            'p': 0.045
        }
    ),
    (
        'L5_TTPC1', 'L6_TCP_L4',
        {   'tau_decay' : 47e-3, 
            'tau_decay_std' : 9.7e-3, 
            'w' : 0.3e-9, 
            'w_std' : 0.1e-9, 
            'c' : 3.5, 
            'c_std' : 1.5, 
            'p': 0.047
        }
    ),
    (
        'L5_TTPC1', 'L6_LBC',
        {   'tau_decay' : 22e-3, 
            'tau_decay_std' : 31e-3, 
            'w' : 0.3e-9, 
            'w_std' : 0.1e-9, 
            'c' : 8.3, 
            'c_std' : 3.2, 
            'p': 0.015
        }
    ),
    (
        'L5_TTPC1', 'L6_NBC',
        {   'tau_decay' : 10e-3, 
            'tau_decay_std' : 6.7e-3, 
            'w' : 0.3e-9, 
            'w_std' : 0.1e-9, 
            'c' : 8.3, 
            'c_std' : 3, 
            'p': 0.013
        }
    ),
    (
        'L5_TTPC2', 'L6_BPC',
        {   'tau_decay' : 51e-3, 
            'tau_decay_std' : 18e-3, 
            'w' : 0.3e-9, 
            'w_std' : 0.1e-9, 
            'c' : 3, 
            'c_std' : 1.3, 
            'p': 0.046
        }
    ),
    (
        'L5_TTPC2', 'L6_IPC',
        {   'tau_decay' : 45e-3, 
            'tau_decay_std' : 5.8e-3, 
            'w' : 0.3e-9, 
            'w_std' : 0.1e-9, 
            'c' : 3.3, 
            'c_std' : 1.5, 
            'p': 0.016
        }
    ),
    (
        'L5_TTPC2', 'L6_UTPC',
        {   'tau_decay' : 54e-3, 
            'tau_decay_std' : 15e-3, 
            'w' : 0.3e-9, 
            'w_std' : 0.1e-9, 
            'c' : 3.2, 
            'c_std' : 1.5, 
            'p': 0.052
        }
    ),
    (
        'L5_TTPC2', 'L6_TCP_L1',
        {   'tau_decay' : 46e-3, 
            'tau_decay_std' : 6.8e-3, 
            'w' : 0.3e-9, 
            'w_std' : 0.1e-9, 
            'c' : 3.3, 
            'c_std' : 1.5, 
            'p': 0.044
        }
    ),
    (
        'L5_TTPC2', 'L6_TCP_L4',
        {   'tau_decay' : 49e-3, 
            'tau_decay_std' : 11e-3, 
            'w' : 0.3e-9, 
            'w_std' : 0.1e-9, 
            'c' : 3.2, 
            'c_std' : 1.4, 
            'p': 0.048
        }
    ),                                   
    (
        'L5_TTPC2', 'L6_LBC',
        {   'tau_decay' : 23e-3, 
            'tau_decay_std' : 29e-3, 
            'w' : 0.3e-9, 
            'w_std' : 0.1e-9, 
            'c' : 7.9, 
            'c_std' : 3.1, 
            'p': 0.013
        }
    ),
    (
        'L5_TTPC2', 'L6_NBC',
        {   'tau_decay' : 10e-3, 
            'tau_decay_std' : 6.3e-3, 
            'w' : 0.3e-9, 
            'w_std' : 0.1e-9, 
            'c' : 8.1, 
            'c_std' : 3, 
            'p': 0.012
        }
    ),       
    # Layer 6 ---------------------------------------------------------
    # Internal
    (
        'L6_BPC', 'L6_BPC',
        {   'tau_decay' : 48e-3, 
            'tau_decay_std' : 11e-3, 
            'w' : 0.3e-9, 
            'w_std' : 0.1e-9, 
            'c' : 2.9, 
            'c_std' : 1.2, 
            'p': 0.1
        }
    ),
    (
        'L6_BPC', 'L6_IPC',
        {   'tau_decay' : 44e-3, 
            'tau_decay_std' : 6e-3, 
            'w' : 0.3e-9, 
            'w_std' : 0.1e-9, 
            'c' : 2.7, 
            'c_std' : 1.1, 
            'p': 0.11
        }
    ),
    (
        'L6_BPC', 'L6_TCP_L1',
        {   'tau_decay' : 44e-3, 
            'tau_decay_std' : 4.9e-3, 
            'w' : 0.3e-9, 
            'w_std' : 0.1e-9, 
            'c' : 3.2, 
            'c_std' : 1.5, 
            'p': 0.072
        }
    ),
    (
        'L6_BPC', 'L6_TCP_L4',
        {   'tau_decay' : 47e-3, 
            'tau_decay_std' : 8.7e-3, 
            'w' : 0.3e-9, 
            'w_std' : 0.1e-9, 
            'c' : 3, 
            'c_std' : 1.3, 
            'p': 0.11
        }
    ),
    (
        'L6_BPC', 'L6_UTPC',
        {   'tau_decay' : 47e-3, 
            'tau_decay_std' : 8.7e-3, 
            'w' : 0.3e-9, 
            'w_std' : 0.1e-9, 
            'c' : 3, 
            'c_std' : 1.3, 
            'p': 0.11
        }
    ),
    (
        'L6_BPC', 'L6_LBC',
        {   'tau_decay' : 16e-3, 
            'tau_decay_std' : 20e-3, 
            'w' : 0.3e-9, 
            'w_std' : 0.1e-9, 
            'c' : 7.2, 
            'c_std' : 2.8, 
            'p': 0.054
        }
    ),
    (
        'L6_BPC', 'L6_NBC',
        {   'tau_decay' : 8.6e-3, 
            'tau_decay_std' : 5.8e-3, 
            'w' : 0.3e-9, 
            'w_std' : 0.1e-9, 
            'c' : 6.8, 
            'c_std' : 2.5, 
            'p': 0.062
        }
    ),
    (
        'L6_IPC', 'L6_IPC',
        {   'tau_decay' : 44e-3, 
            'tau_decay_std' : 6.9e-3, 
            'w' : 0.3e-9, 
            'w_std' : 0.1e-9, 
            'c' : 2.8, 
            'c_std' : 1.2, 
            'p': 0.089
        }
    ),
    (
        'L6_IPC', 'L6_BPC',
        {   'tau_decay' : 40e-3, 
            'tau_decay_std' : 9.2e-3, 
            'w' : 0.3e-9, 
            'w_std' : 0.1e-9, 
            'c' : 2.9, 
            'c_std' : 1.3, 
            'p': 0.088
        }
    ),
    (
        'L6_IPC', 'L6_TCP_L4',
        {   'tau_decay' : 33e-3, 
            'tau_decay_std' : 9.5e-3, 
            'w' : 0.3e-9, 
            'w_std' : 0.1e-9, 
            'c' : 3, 
            'c_std' : 1.3, 
            'p': 0.075
        }
    ),
    (
        'L6_IPC', 'L6_TCP_L1',
        {   'tau_decay' : 35e-3, 
            'tau_decay_std' : 8.9e-3, 
            'w' : 0.3e-9, 
            'w_std' : 0.1e-9, 
            'c' : 3.3, 
            'c_std' : 1.4, 
            'p': 0.073
        }
    ),
    (
        'L6_IPC', 'L6_UTPC',
        {   'tau_decay' : 46e-3, 
            'tau_decay_std' : 5.3e-3, 
            'w' : 0.3e-9, 
            'w_std' : 0.1e-9, 
            'c' : 3, 
            'c_std' : 1.3, 
            'p': 0.097
        }
    ),
    (
        'L6_IPC', 'L6_LBC',
        {   'tau_decay' : 53e-3, 
            'tau_decay_std' : 39e-3, 
            'w' : 0.3e-9, 
            'w_std' : 0.1e-9, 
            'c' : 7.4, 
            'c_std' : 3, 
            'p': 0.052
        }
    ),
    (
        'L6_IPC', 'L6_NBC',
        {   'tau_decay' : 79e-3, 
            'tau_decay_std' : 36e-3, 
            'w' : 0.3e-9, 
            'w_std' : 0.1e-9, 
            'c' : 7, 
            'c_std' : 2.7, 
            'p': 0.059
        }
    ),
    (
        'L6_TCP_L1', 'L6_TCP_L1',
        {   'tau_decay' : 33e-3, 
            'tau_decay_std' : 8.7e-3, 
            'w' : 0.3e-9, 
            'w_std' : 0.1e-9, 
            'c' : 4.6, 
            'c_std' : 2.2, 
            'p': 0.071
        }
    ),
    (
        'L6_TCP_L1', 'L6_IPC',
        {   'tau_decay' : 45e-3, 
            'tau_decay_std' : 6.8e-3, 
            'w' : 0.3e-9, 
            'w_std' : 0.1e-9, 
            'c' : 3, 
            'c_std' : 1.3, 
            'p': 0.11
        }
    ),
    (
        'L6_TCP_L1', 'L6_TCP_L4',
        {   'tau_decay' : 36e-3, 
            'tau_decay_std' : 11e-3, 
            'w' : 0.3e-9, 
            'w_std' : 0.1e-9, 
            'c' : 4.6, 
            'c_std' : 2.1, 
            'p': 0.07
        }
    ),
    (
        'L6_TCP_L1', 'L6_UTPC',
        {   'tau_decay' : 45e-3, 
            'tau_decay_std' : 5.9e-3, 
            'w' : 0.3e-9, 
            'w_std' : 0.1e-9, 
            'c' : 3.1, 
            'c_std' : 1.4, 
            'p': 0.13
        }
    ),
    (
        'L6_TCP_L4', 'L6_TCP_L4',
        {   'tau_decay' : 36e-3, 
            'tau_decay_std' : 11e-3, 
            'w' : 0.3e-9, 
            'w_std' : 0.1e-9, 
            'c' : 4.5, 
            'c_std' : 2.1, 
            'p': 0.055
        }
    ),
    (
        'L6_TCP_L4', 'L6_IPC',
        {   'tau_decay' : 45e-3, 
            'tau_decay_std' : 6.2e-3, 
            'w' : 0.3e-9, 
            'w_std' : 0.1e-9, 
            'c' : 2.8, 
            'c_std' : 1.2, 
            'p': 0.11
        }
    ),
    (
        'L6_TCP_L4', 'L6_TCP_L1',
        {   'tau_decay' : 35e-3, 
            'tau_decay_std' : 8.9e-3, 
            'w' : 0.3e-9, 
            'w_std' : 0.1e-9, 
            'c' : 4.6, 
            'c_std' : 2.1, 
            'p': 0.055
        }
    ),
    (
        'L6_TCP_L4', 'L6_UTPC',
        {   'tau_decay' : 47e-3, 
            'tau_decay_std' : 6.4e-3, 
            'w' : 0.3e-9, 
            'w_std' : 0.1e-9, 
            'c' : 3, 
            'c_std' : 1.3, 
            'p': 0.11
        }
    ),
    (
        'L6_UTPC', 'L6_UTPC',
        {   'tau_decay' : 46e-3, 
            'tau_decay_std' : 5.1e-3, 
            'w' : 0.3e-9, 
            'w_std' : 0.1e-9, 
            'c' : 3.3, 
            'c_std' : 1.5, 
            'p': 0.13
        }
    ),
    (
        'L6_UTPC', 'L6_BPC',
        {   'tau_decay' : 46e-3, 
            'tau_decay_std' : 6.7e-3, 
            'w' : 0.3e-9, 
            'w_std' : .1e-9, 
            'c' : 3.2, 
            'c_std' : 1.4, 
            'p': 0.13
        }
    ),
    (
        'L6_UTPC', 'L6_IPC',
        {   'tau_decay' : 44e-3, 
            'tau_decay_std' : 5.9e-3, 
            'w' : 0.3e-9, 
            'w_std' : 0.1e-9, 
            'c' : 3, 
            'c_std' : 1.3, 
            'p': 0.13
        }
    ),
    (
        'L6_UTPC', 'L6_TCP_L1',
        {   'tau_decay' : 43e-3, 
            'tau_decay_std' : 5.3e-3, 
            'w' : 0.3e-9, 
            'w_std' : 0.1e-9, 
            'c' : 3.7, 
            'c_std' : 1.6, 
            'p' : 0.091
        }
    ),
    (
        'L6_UTPC', 'L6_TCP_L4',
        {   'tau_decay' : 42e-3, 
            'tau_decay_std' : 10e-3, 
            'w' : 0.3e-9, 
            'w_std' : 0.1e-9, 
            'c' : 3.3, 
            'c_std' : 1.4, 
            'p': 0.10
        }
    ),
    (
        'L6_UTPC', 'L6_LBC',
        {   'tau_decay' : 18e-3, 
            'tau_decay_std' : 26e-3, 
            'w' : 0.3e-9, 
            'w_std' : 0.1e-9, 
            'c' : 7.9, 
            'c_std' : 3.1, 
            'p': 0.071
        }
    ),
    (
        'L6_UTPC', 'L6_NBC',
        {   'tau_decay' : 9.9e-3, 
            'tau_decay_std' : 6.2e-3, 
            'w' : 0.3e-9,
            'w_std' : 0.1e-9, 
            'c' : 7.6, 
            'c_std' : 2.7, 
            'p': 0.078
        }
    ),
    # L6_LBC
    (
        'L6_LBC', 'L6_LBC',
        {   'tau_decay' : 36e-3, 
            'tau_decay_std' : 16e-3, 
            'w' : 0.3e-9, 
            'w_std' : 0.14e-9, 
            'c' : 13, 
            'c_std' : 4.6, 
            'p': 0.056
        }
    ),
    (
        'L6_LBC', 'L6_BPC',
        {   'tau_decay' : 51e-3, 
            'tau_decay_std' : 16e-3, 
            'w' : 0.3e-9,
            'w_std' : 0.1e-9, 
            'c' : 13, 
            'c_std' : 4.8, 
            'p': 0.079
        }
    ),
    (
        'L6_LBC', 'L6_IPC',
        {   'tau_decay' : 50e-3, 
            'tau_decay_std' : 16e-3, 
            'w' : 0.27e-9,
            'w_std' : 0.13e-9, 
            'c' : 13, 
            'c_std' : 4.6, 
            'p': 0.057
        }
    ),
    (
        'L6_LBC', 'L6_TCP_L1',
        {   'tau_decay' : 48e-3, 
            'tau_decay_std' : 14e-3, 
            'w' : 0.27e-9, 
            'w_std' : .1e-9, 
            'c' : 14, 
            'c_std' : 5.3, 
            'p': 0.074
        }
    ),
    (
        'L6_LBC', 'L6_TCP_L4',
        {   'tau_decay' : 55e-3, 
            'tau_decay_std' : 28e-3, 
            'w' : .26e-9,
            'w_std' : .13e-9, 
            'c' : 13, 
            'c_std' : 4.6, 
            'p': 0.076
        }
    ),
    (
        'L6_LBC', 'L6_UTPC',
        {   'tau_decay' : 55e-3, 
            'tau_decay_std' : 22e-3, 
            'w' : 0.27e-9, 
            'w_std' : .13e-9, 
            'c' : 14, 
            'c_std' : 5.3, 
            'p': 0.075
        }
    ),
    # L6_NBC
    (
        'L6_NBC', 'L6_IPC',
        {   'tau_decay' : 51e-3, 
            'tau_decay_std' : 16e-3, 
            'w' : 1.4e-9, 
            'w_std' : 1.6e-9, 
            'c' : 15, 
            'c_std' : 6, 
            'p': 0.065
        }
    ),
    (
        'L6_NBC', 'L6_TCP_L1',
        {   'tau_decay' : 51e-3, 
            'tau_decay_std' : 22e-3, 
            'w' : 1.8e-9, 
            'w_std' : 1.7e-9, 
            'c' : 17, 
            'c_std' : 7.1, 
            'p': 0.079
        }
    ),
    (
        'L6_NBC', 'L6_TCP_L4',
        {   'tau_decay' : 54e-3, 
            'tau_decay_std' : 25e-3, 
            'w' : 1.9e-9, 
            'w_std' : 1.7e-9, 
            'c' : 16, 
            'c_std' : 6.4, 
            'p': 0.078
        }
    ),
    (
        'L6_NBC', 'L6_UTPC',
        {   'tau_decay' : 53e-3, 
            'tau_decay_std' : 21e-3, 
            'w' : 1.8e-9, 
            'w_std' : 1.7e-9, 
            'c' : 16, 
            'c_std' : 7, 
            'p': 0.075
        }
    ),
    # External
    # to L5, I =>
    (
        'L6_LBC', 'L5_TTPC1',
        {   'tau_decay' : 100e-3, 
            'tau_decay_std' : 39e-3, 
            'w' : 0.27e-9, 
            'w_std' : 0.13e-9, 
            'c' : 18.8, 
            'c_std' : 8.8, 
            'p': 0.017
        }
    ),
    (
        'L6_LBC', 'L5_TTPC2',
        {   'tau_decay' : 90e-3, 
            'tau_decay_std' : 34e-3, 
            'w' : 0.27e-9, 
            'w_std' : 0.13e-9, 
            'c' : 16, 
            'c_std' : 7.1, 
            'p': 0.018
        }
    ),
    # There are no L6_NBC to L5_TTPC1 connections
    (
        'L6_NBC', 'L5_TTPC2',
        {   'tau_decay' : 92e-3, 
            'tau_decay_std' : 35e-3, 
            'w' : 2.0e-9, 
            'w_std' : 1.7e-9, 
            'c' : 22, 
            'c_std' : 10, 
            'p': 0.0079
        }
    ),
]   

pars = BMparams(pops, conns, 
        ['L4_SS', 'L4_PC'], 
        ['L6_UTPC', 'L6_IPC', 'L6_TCP_L1', 'L6_TCP_L4'],
        [135e1, 135e1], [400e-9, 1600e-9])
