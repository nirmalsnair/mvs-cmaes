"""
Middlebury Mview dataset_par.txt to Standard-n34 dataset_P.mat format conversion
"""

import sys, os.path as path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
import scipy.io as spio
from modules import func


# Dataset path
data_path = 'data'
dataset = 'dinoRing'

# Input and output file paths
rpath = path.join(data_path, dataset, dataset + '_par.txt')
wpath = path.join(data_path, dataset, dataset + '_P.mat')

# Convert camera matrices from Middlebury format to standard format used in this codebase
P = func.mview2standard(rpath)

# Save the converted camera matrices to a .mat file
print('Writing converted camera matrices to:', wpath)
spio.savemat(wpath, dict(P=P))
