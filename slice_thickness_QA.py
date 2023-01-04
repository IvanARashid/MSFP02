# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 13:23:51 2022

@author: Ivan
"""
# %%

import numpy as np
import matplotlib.pyplot as plt
import pydicom
import os
from lmfit.models import StepModel  # import Model from LMFIT library
from lmfit.models import GaussianModel

SMALL_SIZE = 16
MEDIUM_SIZE = 20
BIGGER_SIZE = 24

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# %%


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
        volume = [pydicom.dcmread(full_path)]
        type(volume)
        image_volume = volume[0].pixel_array
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


def fit_erf(line):
    """
    Apply a step fit to a line profile (numpy array).
    The centre is guessed automatically from the data.
    Returns the best fit and the found fitting parameters.
    """
    model = StepModel(form='erf')

    # guess the centre as coordinates of mean value of curve
    centre_guess = np.abs(line - np.mean(np.unique(line))).argmin()
    pars = model.guess(line, x=np.arange(line.shape[0]), center=centre_guess)

    out = model.fit(line, pars, x=range(len(line)))

    print(out.fit_report())
    return out.best_fit, out.values

def fit_gaussian(lsf):
    """
    Apply a Gaussian fit to a projected slice profile (numpy array).
    Parameters are guessed automtically from input data.
    Returns the best fit and the found fitting parameters.
    """
    model = GaussianModel()
    params = model.make_params(center=np.argmax(lsf), sigma=5, amplitude=np.amax(lsf))
    result = model.fit(lsf, params, x=np.linspace(0, len(lsf), len(lsf)))

    print(result.fit_report())

    return result.best_fit, result.values


def add_padding(data, position='after', amount=5):
    """ Add padding BEFORE or AFTER data (list or 1D numpy array). """
    if position == 'after':
        data = np.pad(data, (0, amount), 'edge')
    elif position == 'before':
        data = np.pad(data, (amount, 0), 'edge')
    return data

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


# %% Preview the slice

path = r"C:\Users\Ivan\Documents\MSFP02 klinisk praktik\MR\MR bilder\swoop_tom^Phan_20220902_PhantomTest"
folder = r"0007_T2_(AXI__Fast)"
path = os.path.join(path, folder)

volume, image_volume = load_dicom(path, dir=True)

view_volume(image_volume)

# Read the dicom header and find the pixel spacing
print(volume)

# %% Settings, CHANGE THESE!!!

pixel_spacing = volume[0][0x0028,0x0030].value[0] #0.286458 # mm, from dicom header
slice = 9 # The slice to be used

# Thickness of the line profile to be extracted. The values will be averaged
thickness = 10

# Line profile position
x_start = 60
x_stop = x_start + thickness
y_start = 40
y_stop = 100

# Kernel size for smoothing of the line profile
kernel_size = 10

# Angle of the wedges in the phantom. 11.3 degrees for the Siemens multipurpose phantom
angle = 11.3*np.pi/180

# %% Extract The ERF (NEMA MS 5-2018 Figure 2-3)

ERF = image_volume[slice][y_start:y_stop,x_start:x_stop]

ERF = np.average(ERF, axis=1)
ERF = [int(i) for i in ERF]

# Smooth the data
kernel = np.ones(kernel_size) / kernel_size
data_convolved = np.convolve(ERF, kernel, mode='same')
ERF = data_convolved        
        
# Add padding
"""
ERF = add_padding(ERF, position="before", amount=200)
ERF = add_padding(ERF, position="after", amount=200)
"""

# Flip the ERF to get a positive derivative
if ERF[10] > ERF[-10]:
    ERF = np.flip(ERF)

# Fit error function to ERF
line_fit, fit_values = fit_erf(ERF)
sigma_esf = fit_values['sigma']  # characteristic width of edge in pixels
fwhm_esf = sigma_esf * 2.35
print('ESF: Sigma = {}, FWHM = {} pixels'.format(sigma_esf, fwhm_esf))

# Differentiate the ERF to get the projected slice profile
dx = 1
lsf = np.diff(ERF)/dx # Differentiate data points
lsf_from_fit = np.diff(line_fit)/dx # Differentiate the fitted error function

# Fit a Gaussian to the projected slice profiles to get the FWHM
lsf_fit, fit_values = fit_gaussian(lsf)
sigma_lsf = fit_values['sigma']
fwhm_lsf = fit_values['fwhm']

lsf_fit2, fit_values2 = fit_gaussian(lsf_from_fit)
sigma_lsf2 = fit_values2['sigma']
fwhm_lsf2 = fit_values2['fwhm']

# Determine and print the slice thickness
print("")
print("### RESULTS ###")
print("Slice thickness from data: {} mm".format(fwhm_lsf*pixel_spacing*np.tan(angle)))
print("Slice thickness from curve fit: {} mm".format(fwhm_lsf2*pixel_spacing*np.tan(angle)))

# %% Plots

# Makes the line profile visible on the image
image_volume[slice][y_start:y_stop,x_start:x_stop+thickness] = np.max(image_volume[slice])

fig, axs = plt.subplots(1,3)
axs[0].imshow(image_volume[slice], cmap="gray")
axs[0].set_title("Extracted line profile")
axs[0].axis("off")


axs[1].plot(ERF[10:-10], ls="", marker=".", color="black")
#axs[1].plot(line_fit[10:-10], '--')
axs[1].set_title("Line profile")
axs[1].set_xlabel("Pixels")

axs[2].plot(lsf[10:-10], ls="", marker=".", color="black")
axs[2].plot(lsf_fit[10:-10], '--', label="Fit to data: {:.1f} mm".format(fwhm_lsf*pixel_spacing*np.tan(angle)))
#axs[2].plot(lsf_fit2[10:-10], '--',label="Fit to error function: {:.1f} mm".format(fwhm_lsf2*pixel_spacing*np.tan(angle)))
axs[2].set_title("Differentiated line profile")
axs[2].set_xlabel("Pixels")
axs[2].legend(loc=1)

plt.tight_layout()
