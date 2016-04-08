"""Usage: joinmass.py [-d] NAME FILES...

Join 'ys' files from a bluemass.py simulation.

    Arguments:
        NAME        Name of the result        
        FILES...    Files to join

    Options:
        -d          Delete original files

"""
from __future__ import division
from docopt import docopt
import numpy as np


args = docopt(__doc__, version='Alpha')
files = args['FILES']

# Loop, open, join
ys = []
times = []
for f in files:
    npz = np.load(f)
    ys.append(npz['ys'])
    times.append(npz['times'])

# Sort ys by times
ys = np.vstack(ys)
times = np.concatenate(times)

reordered = times.argsort()
ys = ys[reordered, :]
times = times[reordered]

# Grab the index from the last file
# (they should be idenitcal)
idxs = npz['idxs'].item()

# Save joined
np.savez(
    args['NAME'], 
    ys=ys,
    times=times,
    idxs=idxs
)

# Delete originals?
if args['-d']:
    [os.remove(f) for f in files]

