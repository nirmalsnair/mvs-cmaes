"""
Interactive patch inspection tool for visualizing the effect of depth adjustments
on patch similarity across views.

This script allows users to interactively explore how changes in the depth value
of a 3D point affect the similarity of image patches across multiple views. Users
can adjust the 3D point index, depth value, and patch size using sliders, and the
corresponding patches and their normalized cross-correlation (NCC) scores will be
displayed for each view.

Keyboard shortcuts:
    8 and 9: decrement/increment slider 1 (3D point)
    5 and 6: decrement/increment slider 2 (depth value)
    2 and 3: decrement/increment slider 3 (patch size)
    Ctrl: Large adjustment modifier
    Alt: Very large adjustment modifier
    WIN key: Fine adjustment modifier (for depth value only)
"""

import numpy as np
import scipy.io as spio
import matplotlib.pyplot as plt
import sys, os.path as path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from matplotlib.widgets import Slider
import modules.func as func


# Dataset
data_path = 'data'
dataset = 'dinoRing'

# Default parameters
point_idx = 0
patch_size = 5

# Slider step sizes
v1 = 1                  # For 3D point index
v2 = 0.001              # For depth value
v3 = 2                  # For patch size

# Load data
im_dir = path.join(data_path, dataset, 'images')
I = func.load_images_from_folder(im_dir, 0)
n_images = len(I)

f_path = path.join(data_path, dataset + '_dmaps.mat')
mat_file = spio.loadmat(f_path)
FA = mat_file['FA']
PA = mat_file['PA']
TA = mat_file['TA']
FA = np.squeeze(FA)
TA = np.squeeze(TA)

fpath = path.join(data_path, dataset + '_TPXE.mat')
mat_file = spio.loadmat(fpath)
P = mat_file['P']
n_pts = FA.shape[0]

# Set up plotting grid based on number of images
if (n_images == 48):        # dinoRing
    n_rows = 4
    n_cols = 12
elif (n_images == 7):       # Leuven
    n_rows = 2
    n_cols = 4
else:
    raise ValueError('No n_rows n_cols set for plotting')

fig, axes = plt.subplots(n_rows, n_cols)
axes = axes.ravel()
[axi.set_axis_off() for axi in axes]
plt.subplots_adjust(top=0.95, bottom=0.15, left=0.05, right=0.95)
plt.rcParams['axes.titlesize'] = 'small'
plt.rcParams['image.cmap'] = 'gray'

# Slider position, size, label, range, format, stepsize
ax1 = fig.add_axes([0.10, 0.11, 0.65, 0.02])
ax2 = fig.add_axes([0.10, 0.08, 0.65, 0.02])
ax3 = fig.add_axes([0.10, 0.05, 0.65, 0.02])
slider1 = Slider(ax1, '3D point', 0, n_pts, valinit=0, valfmt='%d / '+str(n_pts), valstep=1)
slider2 = Slider(ax2, 'Depth value', -5, 5, valinit=0, valfmt='%0.4f')
slider3 = Slider(ax3, 'Patch size', 5, 135, valinit=patch_size, valfmt='%d', valstep=2)

axim = []
for i in range(n_images):
    p0 = np.zeros((patch_size, patch_size), dtype=np.uint8)
    axi = axes[i].imshow(p0, vmin=0, vmax=255)
    axim.append(axi)
ptext_1 = plt.text(155, 3.2, '')
ptext_2 = plt.text(155, 1.7, '')

# Change 3D point
def update_1(val):
    val1 = int(val)
    val2 = TA[val1]
    slider2.vline.set_xdata(val2)                   # Change valinit marker line
    slider2.set_val(val2)                           # Triggers update on slider2

# Change depth value
def update_2(val):
    val1 = int(slider1.val)
    val2 = val
    val3 = int(slider3.val)
    update(val1, val2, val3)

# Change patch size
def update_3(val):
    val1 = int(slider1.val)
    val2 = slider2.val
    val3 = int(val)
    update(val1, val2, val3)

def update(val1, val2, val3):
    imx0, x, y = PA[val1]
    point = np.array((x,y,1)).reshape((3,1))
    rpos, rdir = func.camera_ray(P[imx0], point)
    X = rpos + val2 * rdir
    X = np.vstack((X, 1))
    p0 = func.get_patch(I[imx0], x, y, val3)

    for i in range(n_images):
        xi = func.cam_project(P[i], X)
        pi = func.get_patch(I[i], xi[0,0], xi[1,0], val3)
        axim[i].set_data(pi)
        ncc = func.zncc(p0, pi)
        axes[i].set_title(str(ncc), zorder=0)
        axes[i].title.set_backgroundcolor('1.0')

    axes[imx0].title.set_backgroundcolor('0.7')                 # Highlight reference view
    ptext_1.set_text('f = ' + str(round(FA[val1], 4)))          # Fitness value
    ptext_2.set_text(str([imx0, x, y]))                         # Reference point coordinates
    fig.canvas.draw()

# Connect sliders to update functions
slider1.on_changed(update_1)
slider2.on_changed(update_2)
slider3.on_changed(update_3)

# Keyboard shortcuts for slider adjustments
def on_key_press(event):
    key = event.key

    # 3D point index adjustments
    if (key == '8'):
        slider1.set_val(slider1.val - v1)
    elif (key == '9'):
        slider1.set_val(slider1.val + v1)
    elif (key == 'ctrl+8'):                     # Large decrement
        slider1.set_val(slider1.val - v1*5)
    elif (key == 'ctrl+9'):                     # Large increment
        slider1.set_val(slider1.val + v1*5)
    elif (key == 'alt+8'):                      # Very large decrement
        slider1.set_val(slider1.val - v1*50)
    elif (key == 'alt+9'):                      # Very large increment
        slider1.set_val(slider1.val + v1*50)
    
    # Depth value adjustments
    elif (key == '5'):
        slider2.set_val(slider2.val - v2)
    elif (key == '6'):
        slider2.set_val(slider2.val + v2)
    elif (key == 'super+5'):                    # Fine decrement
        slider2.set_val(slider2.val - v2/10)
    elif (key == 'super+6'):                    # Fine increment
        slider2.set_val(slider2.val + v2/10)
    elif (key == 'ctrl+5'):                     # Large decrement
        slider2.set_val(slider2.val - v2*5)
    elif (key == 'ctrl+6'):                     # Large increment
        slider2.set_val(slider2.val + v2*5)
    elif (key == 'alt+5'):                      # Very large decrement
        slider2.set_val(slider2.val - v2*10)
    elif (key == 'alt+6'):                      # Very large increment
        slider2.set_val(slider2.val + v2*10)
    
    # Patch size adjustments
    elif (key == '2'):
        slider3.set_val(slider3.val - v3)
    elif (key == '3'):
        slider3.set_val(slider3.val + v3)
    elif (key == 'ctrl+2'):                     # Large decrement
        slider3.set_val(slider3.val - v3*5)
    elif (key == 'ctrl+3'):                     # Large increment
        slider3.set_val(slider3.val + v3*5)
    elif (key == 'alt+2'):                      # Very large decrement
        slider3.set_val(slider3.val - v3*10)
    elif (key == 'alt+3'):                      # Very large increment
        slider3.set_val(slider3.val + v3*10)

# Connect key press event to handler
fig.canvas.mpl_connect('key_press_event', on_key_press)

# Initial update to display the first point
slider1.set_val(point_idx)
slider3.set_val(patch_size)
plt.show()
