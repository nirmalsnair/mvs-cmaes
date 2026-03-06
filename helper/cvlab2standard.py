"""
Convert camera matrices from EPFL CVLab format to standard format used in this codebase.

The CVLab dataset can be downloaded from:
https://documents.epfl.ch/groups/c/cv/cvlab-unit/www/data/multiview/denseMVS.html

The script assumes that the camera matrices are stored in files with names ending in
'camera' in the specified folder. The converted camera matrices are saved in a .mat file
with the same name as the dataset, containing a variable 'P' which is a list of camera
matrices in the standard format.
"""

import sys, os.path as path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
import scipy.io as spio
from modules import func


# Dataset
data_path = 'data'
dataset = 'Leuven'

# Read and write paths
r_path = path.join(data_path, dataset)
w_path = path.join(data_path, dataset, dataset + '_P.mat')

# Convert camera matrices from CVLab format to standard format and save as .mat file
P = func.cvlab2standard(r_path)
spio.savemat(w_path, dict(P=P))
