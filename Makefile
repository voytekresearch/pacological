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

# Try MI for PAC, N = 250
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

# Identical to exp6 but LFP is used for MI/H in place of spikes
exp15:
	-mkdir data/exp15
	-rm data/exp15/*
	nice -19 python exp/exp15.py data/exp15/

# Does a big M (that is indersampled) effect the I
# Set m = 20, N = 100 
# Result: The seperation between
# Iosc levels was a bit better. 
exp16:
	-mkdir data/exp16
	-rm data/exp16/*
	nice -19 python exp/exp16.py data/exp16/

# ---
# Exp 15 looked good....
# So rerunning all major rate experiments at m = 20.  
# Also changing f to 6 hz
# TODO Redo synch too?
rate_models_m20: exp17 exp18 exp19 exp20 exp21

# N = 250
exp17:
	-mkdir data/exp17
	-rm data/exp17/*
	nice -19 python exp/exp17.py data/exp17/

# N = 500
exp18:
	-mkdir data/exp18
	-rm data/exp18/*
	nice -19 python exp/exp18.py data/exp18/

# N = 750
exp19:
	-mkdir data/exp19
	-rm data/exp19/*
	nice -19 python exp/exp19.py data/exp19/

# N = 1000
exp20:
	-mkdir data/exp20
	-rm data/exp20/*
	nice -19 python exp/exp20.py data/exp20/

# N = 250, rereun of exp 15 but with m = 20
exp21:
	-mkdir data/exp21
	-rm data/exp21/*
	nice -19 python exp/exp21.py data/exp21/

# ---
# In exp 12 MI was used instead of PLV, and it gave very different results.
# Here we try the other two metrics scott implemented. 
# Roermer arues that Canolty MI and PLV are, 'the closest to the data'
rate_models_pac: exp22 exp23

# GLM crashes with a matrix inversion error. Skipping fixiing this for now.
exp22:
	-mkdir data/exp22
	-rm data/exp22/*
	nice -19 python exp/exp22.py data/exp22/

exp23:
	-mkdir data/exp23
	-rm data/exp23/*
	nice -19 python exp/exp23.py data/exp23/

# ========================================================================
# # Binary
# Set N to a 100, 250, 500, 750, 1000
# Change k, fixing Istim and I osc to [5, 30]; [low, high]
synch_models: exp30 exp31 exp32 exp33 exp34 exp35 exp36

# N = 100
exp30:
	-mkdir data/exp30
	-rm data/exp30/*
	nice -19 python exp/exp30.py data/exp30/

# N = 250
exp31:
	-mkdir data/exp31
	-rm data/exp31/*
	nice -19 python exp/exp31.py data/exp31/

# N = 500
exp32:
	-mkdir data/exp32
	-rm data/exp32/*
	nice -19 python exp/exp32.py data/exp32/

# N = 750
exp33:
	-mkdir data/exp33
	-rm data/exp33/*
	nice -19 python exp/exp33.py data/exp33/

# N = 1000
exp34:
	-mkdir data/exp34
	-rm data/exp34/*
	nice -19 python exp/exp34.py data/exp34/

# Try poisson_binary where oscllation amplitude matters
# N = 100
exp35:
	-mkdir data/exp35
	-rm data/exp35/*
	nice -19 python exp/exp35.py data/exp35/

# N = 500
exp36:
	-mkdir data/exp36
	-rm data/exp36/*
	nice -19 python exp/exp36.py data/exp36/
