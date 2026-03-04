'''
Create and save foreground masks for an image dataset.

This script uses SLIC superpixel segmentation to create foreground masks for
each image in the dataset. If SLIC fails to segment the image into 2 regions,
it falls back to using Chan-Vese segmentation. The resulting binary masks are
then eroded to refine the foreground regions before being saved to a new directory.
'''

import numpy as np
import os
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage.segmentation import slic
from skimage.segmentation import chan_vese
from skimage.segmentation import mark_boundaries

import sys, os.path as path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
import modules.func as func


# Dataset
data_path = 'data'
dataset = 'dinoRing'
new_dir = 'foreground_mask'

# Erosion kernel size (must be odd for symmetry)
ekern_size = 5


# Load images
im_dir = path.join(data_path, dataset, 'images')
I, file_list = func.load_images_from_folder(im_dir, mode=0, is_return_file_list=True)
n_images = len(I)

# Create new folder to write foreground masks
new_dir_path = path.join(data_path, dataset, new_dir)
if not path.exists(new_dir_path):
    os.makedirs(new_dir_path)
else:
    raise Exception('Folder already exists!')

# Disable matplotlib warning about opening too many windows
plt.rcParams.update({'figure.max_open_warning': 0})

# Create foreground masks using SLIC superpixel segmentation,
# with fallback to Chan-Vese if SLIC fails to segment into 2 regions.
print('Creating foreground masks for {} images..'.format(n_images))
M = []
for i in tqdm(range(n_images)):
    segments = slic(I[i], n_segments=2, compactness=0.00001, max_iter=50)
    n_out_seg = len(np.unique(segments))

    if (n_out_seg != 2):
        segments = chan_vese(I[i], max_iter=50)         # Returns bool
        tqdm.write("Applying Chan-Vese on Image {} since SLIC returned {} segments"
                   .format(i, n_out_seg))
        plt.figure("Image {}, {} Chan-Vese segments".format(i, len(segments)))
        plt.imshow(mark_boundaries(I[i], segments, color=(1,0,1)))
    else:
        plt.figure("Image {}, {} SLIC segments".format(i, len(segments)))
        plt.imshow(mark_boundaries(I[i], segments))
    M.append(segments)
M = np.asarray(M, dtype=np.uint8) * 255                 # 0/255 binary masks

# Erode foreground masks
kernel = np.ones((ekern_size, ekern_size), np.uint8)
print('Eroding foreground masks for {} images..'.format(n_images))
for i in tqdm(range(n_images)):
    M[i] = cv2.erode(M[i], kernel)
    plt.figure("Image {}, {} Chan-Vese segments".format(i, len(segments)))
    plt.imshow(mark_boundaries(I[i], M[i], color=(0,1,1)))

# Write foreground masks to new directory
print('Writing {} foreground masks to {}'.format(n_images, new_dir_path))
for i in range(n_images):
    file_name = path.join(new_dir_path, file_list[i])
    status = cv2.imwrite(file_name, M[i])
    if (status):
        print('Image {} write success'.format(i))
    else:
        raise Exception('Failed to write image {}!'.format(i))
