demo:
	-mkdir data/demo
	-rm data/demo/*
	nice -19 python exp/demo.py data/demo

# I and O rates
exp1:
	-mkdir data/exp1
	-rm data/exp1/*
	nice -19 python exp/exp1.py data/exp1/

# High and low rate (fixed)
# explore ks
exp2:
	-mkdir data/exp2
	-rm data/exp2/*
	nice -19 python exp/exp2.py data/exp2/

# High and low rate
# explore Fs
exp3:
	-mkdir data/exp3
	-rm data/exp3/*
	nice -19 python exp/exp3.py data/exp3/

# Explore relation between k and excitability
# for fixed abs rate and f
exp4:
	-mkdir data/exp4
	-rm data/exp4/*
	nice -19 python exp/exp4.py data/exp4/

# Explore relation between excitability and firing rate
# for alternate synch firing scheme
exp5:
	-mkdir data/exp5
	-rm data/exp5/*
	nice -19 python exp/exp5.py data/exp5/


# -------------------------------------------------------------------
#  Dropped synchrony experiments. Revist later. Focus on gain/ge/gi
#  model.
rate_models: exp6 exp7 exp8 exp9 exp10 exp11 exp12 exp13 exp14 exp15

# ========================================================================
# Rate
# Baseline experiement is 6, everything below is a modification of exp6
# N = 5000
exp6:
	-mkdir data/exp6
	-rm data/exp6/*
	nice -19 python exp/exp6.py data/exp6/

# N = 100
exp7:
	-mkdir data/exp7
	-rm data/exp7/*
	nice -19 python exp/exp7.py data/exp7/

# N = 250
exp8:
	-mkdir data/exp8
	-rm data/exp8/*
	nice -19 python exp/exp8.py data/exp8/

# N = 500
exp9:
	-mkdir data/exp9
	-rm data/exp9/*
	nice -19 python exp/exp9.py data/exp9/

# N = 750
exp10:
	-mkdir data/exp10
	-rm data/exp10/*
	nice -19 python exp/exp10.py data/exp10/

# N = 1000 
exp11:
	-mkdir data/exp11
	-rm data/exp11/*
	nice -19 python exp/exp11.py data/exp11/

# Try MI for PAC, N = 500
exp12:
	-mkdir data/exp12
	-rm data/exp12/*
	nice -19 python exp/exp12.py data/exp12/

# Try 6 Hz, N = 250
exp13:
	-mkdir data/exp13
	-rm data/exp13/*
	nice -19 python exp/exp13.py data/exp13/

# Try 20 Hz, N = 250
exp14:
	-mkdir data/exp14
	-rm data/exp14/*
	nice -19 python exp/exp14.py data/exp14/

# TODO -- redo, deleted on accident. Renumber too.
# Identical to exp6 but LFP is used for MI/H in place of spikes
exp15:
	-mkdir data/exp15
	-rm data/exp15/*
	nice -19 python exp/exp15.py data/exp15/

#
# ========================================================================
# # Binary
# Set N to a 100, 500 
# Change k, fixing Istim and I osc to low/high

# Try poisson_binary where osc amp matters
# N = 100?

