demo:
	-mkdir data/demo
	-rm data/demo/*
	nice -19 python exp/demo.py data/demo

demo2:
	-mkdir data/demo2
	-rm data/demo2/*
	nice -19 python exp/demo_pac2.py data/demo2

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

# --
# Assembly recruitment implementation, with improved noise model
# pn = 0.5
exp24:
	-mkdir data/exp24
	-rm data/exp24/*
	nice -19 python exp/exp24.py data/exp24/

# pn = 0.25
exp25:
	-mkdir data/exp25
	-rm data/exp25/*
	nice -19 python exp/exp25.py data/exp25/

# pn = 0.75
exp26:
	-mkdir data/exp26
	-rm data/exp26/*
	nice -19 python exp/exp26.py data/exp26/

# pn = 0.99
exp28:
	-mkdir data/exp28
	-rm data/exp28/*
	nice -19 python exp/exp28.py data/exp28/

# In exp24-6 MI was below stim_p. N is the cause, like in the intial sims?
# So set N = 500. If this fixes it, why is not intuitive. I though the 
# Background would have a much stronger effect. It doesn't seem to be doing 
# anything? Double check implementation?
#
# p = 0.25
exp29:
	-mkdir data/exp29
	-rm data/exp29/*
	nice -19 python exp/exp29.py data/exp29/

# p = 0.50
exp50:
	-mkdir data/exp50
	-rm data/exp50/*
	nice -19 python exp/exp50.py data/exp50/

# p = 0.75
exp51:
	-mkdir data/exp51
	-rm data/exp51/*
	nice -19 python exp/exp51.py data/exp51/

# pn = 0.99 (i.e. two neuron background), N = 250, constant bias off
# this should reproduce gain_p giving higher than stim MI
# Looked fine.
exp52:
	-mkdir data/exp52
	-rm data/exp52/*
	nice -19 python exp/exp52.py data/exp52/

# --
# Consider the background is stimulus and then a
# subpop gets PACed.
back_stim: exp53 exp54 exp55


exp53:
	-mkdir data/exp53
	-rm data/exp53/*
	nice -19 python exp/exp53.py data/exp53/

exp54:
	-mkdir data/exp54
	-rm data/exp54/*
	nice -19 python exp/exp54.py data/exp54/

exp55:
	-mkdir data/exp55
	-rm data/exp55/*
	nice -19 python exp/exp55.py data/exp55/


# -- 
#  Recast 24 with ge/gi instead of Iosc
#  A fairly course paramterization just to check things out.
exp27:
	-mkdir data/exp27
	-rm data/exp27/*
	nice -19 python exp/exp27.py data/exp27/


# --
#  Combining all the above into a set of 'final' runs
#  these models explore g, N, p_n, Istim, ep_back
#  back_type = constant
exp56:
	-mkdir data/exp56
	-rm data/exp56/*
	nice -19 python exp/exp56.py data/exp56/

# back_type = stim
exp57:
	-mkdir data/exp57
	-rm data/exp57/*
	nice -19 python exp/exp57.py data/exp57/

# back_type = stim
# Ipri = 0.5
exp58:
	-mkdir data/exp58
	-rm data/exp58/*
	nice -19 python exp/exp58.py data/exp58/

# back_type = stim
# m = 4
exp59:
	-mkdir data/exp59
	-rm data/exp59/*
	nice -19 python exp/exp59.py data/exp59/

# Try Ozkurt for 
# back_type = stim
exp60:
	-mkdir data/exp60
	-rm data/exp60/*
	nice -19 python exp/exp60.py data/exp60/


# Try f = 20 for beta gamma pac (exp57 base)
# back_type = stim
exp61:
	-mkdir data/exp61
	-rm data/exp61/*
	nice -19 python exp/exp61.py data/exp61/

# Try a mixture of stong gain (fix at 8) and add/sub
# PAC. May be a useful fig for the dicussion/supp
exp62:
	-mkdir data/exp62
	-rm data/exp62/*
	nice -19 python exp/exp62.py data/exp62/

# Uses LFP to do MI calculations. Repeat of exp57.
exp63:
	-mkdir data/exp63
	-rm data/exp63/*
	nice -19 python exp/exp63.py data/exp63/

# Replace exp based LFP with simple sum of 
# of spiking to make the LFP. Ensure the PLV/OZ
# pattern is not somehow an artfact of the
# exp convolution.
exp64:
	-mkdir data/exp64
	-rm data/exp64/*
	nice -19 python exp/exp64.py data/exp64/

# Try a mixture of stong gain (fix at 2) and add/sub
# PAC. May be a useful fig for the dicussion/supp
exp65:
	-mkdir data/exp65
	-rm data/exp65/*
	nice -19 python exp/exp65.py data/exp65/

# Same as 65 but gmult=4
exp66:
	-mkdir data/exp66
	-rm data/exp66/*
	nice -19 python exp/exp66.py data/exp66/

# Same as 65 but gmult is reset to g, but still
# subtracting from 'gain' instead of 'stim'.
exp67:
	-mkdir data/exp67
	-rm data/exp67/*
	nice -19 python exp/exp67.py data/exp67/

# Same as 67 but control with m = 30
exp68:
	-mkdir data/exp68
	-rm data/exp68/*
	nice -19 python exp/exp68.py data/exp68/

# Same as 67 but E+I injection is constant not sin
exp69:
	-mkdir data/exp69
	-rm data/exp69/*
	nice -19 python exp/exp69.py data/exp69/

# Variation of exp57, but the low part of PAC
# (OZ) is done with osc not the pac pop firing
exp70:
	-mkdir data/exp70
	-rm data/exp70/*
	nice -19 python exp/exp70.py data/exp70/

# Variation of 70, saving the spike variances
exp71:
	-mkdir data/exp71
	-rm data/exp71/*
	nice -19 python exp/exp71.py data/exp71/

# Variation of exp57 where coupling is via softmax(g)

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


# =========================================================================
# Neural mass experiments
# Jansen rit test 
# f8b7968c3189ba709f324d1a82bcb893d7dc72f7
exp80:
	-mkdir data/exp80
	-rm data/exp80/*
	nice -19 python exp/exp80.py data/exp80/

# Explore c and g (under fixed modest noise)
# 1b0624c3d8acb8caf237d554dece1fd71baaae97
exp81:
	-mkdir data/exp81
	-rm data/exp81/*
	nice -19 python exp/exp81.py data/exp81/

# Try my XJW-based model instead of Jansen Rit
# this first go had Ji_e/Ji_i at 0. 
# and had no gain effect.
exp82:
	-mkdir data/exp82
	-rm data/exp82/*
	nice -19 python exp/exp82.py data/exp82/

# clone of 82 but exploring Ji_e/_i
# (just in case; almost certainly isn't
# going to help)
exp83:
	-mkdir data/exp83
	-rm data/exp83/*
	nice -19 python exp/exp83.py data/exp83/


# =========================================================================
# LIF simulations =========================================================
# =========================================================================
# The classic model
# exp200:
# 	-mkdir data/exp200
# 	-rm data/exp200/*
# 	nice -19 python exp/exp200.py data/exp200/
#
# # Classic but oscillating
# exp201:
# 	-mkdir data/exp201
# 	-rm data/exp201/*
# 	nice -19 python exp/exp201.py data/exp201/
#
# # oscillating favoring re
# exp202:
# 	-mkdir data/exp202
# 	-rm data/exp202/*
# 	nice -19 python exp/exp202.py data/exp202/
#
# # oscillating favoring ri
# exp203:
# 	-mkdir data/exp203
# 	-rm data/exp203/*
# 	nice -19 python exp/exp203.py data/exp203/
#
# # Compare 1X, 0.5X and oscllation
# exp204:
# 	-mkdir data/exp204
# 	-rm data/exp204/*
# 	nice -19 python exp/exp204.py data/exp204/
#
# # Look at effect of oscllation f, from 1-60 Hz
# # (uses 6 cores)
# exp205:
# 	-mkdir data/exp205
# 	-rm data/exp205/*
# 	nice -19 python exp/exp205.py data/exp205/
#
# # Play with rate balance (1.6*re, 0.4*ri)
# exp206:
# 	-mkdir data/exp206
# 	-rm data/exp206/*
# 	nice -19 python exp/exp206.py data/exp206/
#
# # Partial rereun of 205 to explore weaker, less than
# # 1X background activity.
# exp207:
# 	-mkdir data/exp207
# 	-rm data/exp207/*
# 	nice -19 python exp/exp207.py data/exp207/
#
# # exploring 205 from 5X to 9 with 5 as constant
# # i.e. very strong background
# exp208:
# 	-mkdir data/exp208
# 	-rm data/exp208/*
# 	nice -19 python exp/exp208.py data/exp208/
#
# # Repeat of 205 but with t = 5 to ensure
# # the slow oscillations effects aren't an artifact 
# # of few cycle counts
# exp209:
# 	-mkdir data/exp209
# 	-rm data/exp209/*
# 	nice -19 python exp/exp209.py data/exp209/
#
# # Repeat of Exp205 using the Fitzhugh-Ganumo neuron
# # (and a less dense I and xfactor sampling)
# exp210:
# 	-mkdir data/exp210
# 	-rm data/exp210/*
# 	nice -19 python exp/exp210.py data/exp210/
#
# # 210 doesn't look mutliplicative. Back to basics for fitz. No f, and
# # repeat the classic, then favoring re then ri
#
# # Classic Reyes for Ftiz
# exp211:
# 	-mkdir data/exp211
# 	-rm data/exp211/*
# 	nice -19 python exp/exp211.py data/exp211/
#
# # Classic for Fitzhugh favoring re
# exp212:
# 	-mkdir data/exp212
# 	-rm data/exp212/*
# 	nice -19 python exp/exp212.py data/exp212/
#
# # Classic for Fitzhugh favoring ri
# exp213:
# 	-mkdir data/exp213
# 	-rm data/exp213/*
# 	nice -19 python exp/exp213.py data/exp213/
#
# # HH neuron, classic testing (f=0)
# # Balanced, excess re then ri 
# # (this time all in one exp file)
# # fd2a6fd24db676d95eff772182f7206f4892ed51
# exp214:
# 	-mkdir data/exp214
# 	-rm data/exp214/*
# 	nice -19 python exp/exp214.py data/exp214/
#
# exp215:
# 	-mkdir data/exp215
# 	-rm data/exp215/*
# 	nice -19 python exp/exp215.py data/exp215/
#
# exp216:
# 	-mkdir data/exp216
# 	-rm data/exp216/*
# 	nice -19 python exp/exp216.py data/exp216/
#
# # HH and oscillation, from 1-60 Hz
# exp217:
# 	-mkdir data/exp217
# 	-rm data/exp217/*
# 	nice -19 python exp/exp217.py data/exp217/
#
# # HH with reduced variance by 10
# exp218:
# 	-mkdir data/exp218
# 	-rm data/exp218/*
# 	nice -19 python exp/exp218.py data/exp218/
#
# # HH with reduced variance by 2
# exp219:
# 	-mkdir data/exp219
# 	-rm data/exp219/*
# 	nice -19 python exp/exp219.py data/exp219/

# Retuned HH to have a Vm variance more 
# like the LIF neuron. Otherwise this is a rerun of 
# exp219/8

# exp220:
# 	-mkdir data/exp220
# 	-rm data/exp220/*
# 	nice -19 python exp/exp220.py data/exp220/

# Setting the baseline LIF data for zandt mass system
# 121e59944b063a90a8471a7060f3149aa166cd01
exp221:
	-mkdir data/exp221
	-rm data/exp221/*
	nice -19 python exp/exp221.py data/exp221/

# Rerun of Exp220 but with Is spanning a smaller
# range, 1-50 rather than 1-300. 
exp222:
	-mkdir data/exp222
	-rm data/exp222/*
	nice -19 python exp/exp222.py data/exp222/

# Rerun of 222 but halving the background 1x rate
# (i.e. how do these hold up to scale changes)
exp223:
	-mkdir data/exp223
	-rm data/exp223/*
	nice -19 python exp/exp223.py data/exp223/

# NEXT TO RUN
# Exploring several FI curves, with background and w_m
# and not
exp224:
	-mkdir data/exp224
	-rm data/exp224/*
	nice -19 python exp/exp224.py data/exp224/

# FI curve, 1X
exp225:
	-mkdir data/exp225
	-rm data/exp225/*
	nice -19 python exp/exp225.py data/exp225/

# Rerun of 204 (LIF) with constant 20 Hz background
# just so the bottom of the oscillation isn't OX
# which doesn't make much bio-sense.
exp226:
	-mkdir data/exp226
	-rm data/exp226/*
	nice -19 python exp/exp226.py data/exp226/

# =========================================================================
# gNNM experiments
# Generic E-I interaction (aka hippocampus)
exp300:
	-mkdir data/exp300
	-rm data/exp300/*
	nice -19 python exp/exp300.py data/exp300/

# Dense parameter sampling of exp300, added trials as well
exp301:
	-mkdir data/exp301
	-rm data/exp301/*
	nice -19 python exp/exp301.py data/exp301/ | tee data/exp301/log

# Param search
exp400:
	-mkdir data/exp400
	-rm data/exp400/*
	nice -19 python exp/exp400.py pars/pars_ei_f0.py data/exp400/ 

exp401:
	-mkdir data/exp401
	-rm data/exp401/*
	nice -19 python exp/exp400.py pars/pars_ei_f10.py data/exp401/ 
