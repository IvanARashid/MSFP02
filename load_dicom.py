# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 11:42:48 2022

@author: Ivan
"""

import numpy as np
import pydicom
import os
import matplotlib.pyplot as plt

def load_dicom(path, dir=False):
    """
    Parameters
    ----------
    path : string
        Path to the directory containing the dicom files.
    dir : boolean, optional
        Set to true if multiframe dicom files are used. If the each slice is a separate file, use False. The default is False.

    Returns
    -------
    volume : list
        List of loaded dicom files.
    image_volume : list
        List of loaded images (numpy arrays).

    """
    if dir:
        file = os.listdir(path)
        full_path = os.path.join(path,file[0])
        volume = pydicom.dcmread(full_path)
        type(volume)
        image_volume = volume.pixel_array
    else:
        
        # Loads a folder of dicom files
        volume = []
        files = os.listdir(path)
        for file in files:
            full_path = os.path.join(path, file)
            volume.append(pydicom.dcmread(full_path))
            #volume.append(pydicom.dcmread(full_path).pixel_array)
        image_volume = np.array([slice.pixel_array for slice in volume])
    return volume, image_volume

def __arrow_navigation__(event, z, Z):
    if event.key == "up":
        z = min(z + 1, Z - 1)
    elif event.key == 'down':
        z = max(z - 1, 0)
    elif event.key == 'right':
        z = min(z + 10, Z - 1)
    elif event.key == 'left':
        z = max(z - 10, 0)
    elif event.key == 'pagedown':
        z = min(z + 50, Z + 1)
    elif event.key == 'pageup':
        z = max(z - 50, 0)
    return z

def view_volume(vol, figure_num=1, cmap='gray', vmin=None, vmax=None):
    """
    Shows volumetric data for interactive inspection.
    Left/Right keys :   ± 10 projections
    Up/Down keys:       ± 1 projection
    Page Up/Down keys:  ± 50 projections
    Should work in Spyder. In PyCharm, change the plotting backend (see test script for details).
    """

    def update_drawing():
        ax.images[0].set_array(vol[z])
        ax.set_title('slice {}/{}'.format(z, vol.shape[0]))
        fig.canvas.draw()

    def key_press(event):
        nonlocal z
        z = __arrow_navigation__(event, z, Z)
        update_drawing()

    Z = vol.shape[0]
    z = (Z - 1) // 2
    fig, ax = plt.subplots(num=figure_num, dpi=200)
    if vmin is None:
        vmin = np.min(vol)
    if vmax is None:
        vmax = np.max(vol)
    ax.imshow(vol[z], cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title('slice {}/{}'.format(z, vol.shape[0]))
    fig.canvas.mpl_connect('key_press_event', key_press)