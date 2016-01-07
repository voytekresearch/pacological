#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------
from brian2 import *
import argparse

parser = argparse.ArgumentParser(
    description="A wilson cowan model.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument(
    "name",
    help="Name of exp, used to save results as hdf5."
)
parser.add_argument(
    "-t", "--time",
    help="Simulation run time (s)",
    default=2,
    type=float
)
parser.add_argument(
    "--re",
    help="E firing rate",
    default=1.0,
    type=float
)
parser.add_argument(
    "--ri",
    help="I firing rate",
    default=0.5,
    type=float
)
parser.add_argument(
    "--kn",
    help="?",
    default=1.0,
    type=float
)
parser.add_argument(
    "--k",
    help="?",
    default=1.0,
    type=float
)
parser.add_argument(
    "--tau_e",
    help="E decay time (s)",
    default=0.005,
    type=float
)
parser.add_argument(
    "--tau_i",
    help="I decay time (s)",
    default=0.02,
    type=float
)
parser.add_argument(
    "--c1",
    help="Connectivity 1",
    default=15.0,
    type=float
)
parser.add_argument(
    "--c2",
    help="Connectivity 1",
    default=15.0,
    type=float
)
parser.add_argument(
    "--c3",
    help="Connectivity 1",
    default=15.0,
    type=float
)
parser.add_argument(
    "--c4",
    help="Connectivity 1",
    default=3.0,
    type=float
)

args = parser.parse_args()

# --
time = args.time * second
time_step = 1 * ms

# Parameters.
re = args.re
ri = args.ri
kn = args.k
k = args.kn

tau_e = args.tau_e * second
tau_i = args.tau_i * second

# Connectivity coefficients.
c1 = args.c1
c2 = args.c2
c3 = args.c3
c4 = args.c4

# Input P(t) and Q(t).
Q = 1
P = 2

# Model definition. Our model is a system of 2 non-linear equations.
# Pop S EI, at fixed driven by a diffusion
# Pop O EI, oscillatin near alpha
# Eqs = """
#         dE/dt = -E/tau_e + ((1 - re * E) * (1 / (1 + exp(-(k * c1 * E - k * c2 * I+ k* P - 2))) - 1/(1 + exp(2*1.0)))) / tau_e : 1
#         dI/dt = -I/tau_i + ((1 - ri * I) * (1 / (1 + exp(-2 * (kn * c3 * E - kn * c4 * I + kn * Q - 2.5))) - 1/(1 + exp(2*2.5)))) / tau_i : 1
#         P = 3*(2**-0.03) : 1
#     """

Eqs = """
        dE/dt = -E/tau_e + ((1 - re * E) * (1 / (1 + exp(-(k * c1 * E - k * c2 * I+ k* P - 2))) - 1/(1 + exp(2*1.0)))) / tau_e : 1
        dI/dt = -I/tau_i + ((1 - ri * I) * (1 / (1 + exp(-2 * (kn * c3 * E - kn * c4 * I + kn * Q - 2.5))) - 1/(1 + exp(2*2.5)))) / tau_i : 1
        P = 3*(2**-0.03) : 1
    """
P = NeuronGroup(1, model=Eqs)
P.E = 0
P.I = 0

# --
# Record
mon = StateMonitor(P, ('E', 'I'), record=True)

# --
# Run
defaultclock.dt = time_step
run(time, report='text')
