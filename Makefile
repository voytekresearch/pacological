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


# Explore relation between k and firing rate
# for fixed abs rate and f
exp4:
	-mkdir data/exp4
	-rm data/exp4/*
	nice -19 python exp/exp4.py data/exp4/

