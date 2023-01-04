# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 14:19:47 2022

@author: Ivan Ahmed (reachable at ivan52x53@gmail.com)
"""

import numpy as np
import load_dicom
import os
import tqdm


path = r"C:\Users\Ivan\Documents\MSFP02 klinisk praktik\MR Lund\MR bilder\In vivo" # Windows
#path = r"/media/ivan/34F2B718F2B6DCF4/Users/Ivan/Documents/MSFP02 klinisk praktik/MR Lund/MR bilder/In vivo" # Linux
b1_folder = r"901_B1_DREAM_20220303_09.11.22"
t2_folder = r"701_multiTE_T2_04x04_SENSE2_20220303_08.53.50"

b1_path = os.path.join(path, b1_folder)
t2_path = os.path.join(path, t2_folder)

b1_volume, b1_image_volume = load_dicom.load_dicom(b1_path)
t2_volume, t2_image_volume = load_dicom.load_dicom(t2_path)

# %% Dicom standard method
# See the DICOM documentation for the matrix equation (doc for image orientation (patient), tag (0020, 0037))

def get_image_orientation(dicom_volume, slice=0):
    """
    Function that reads the ImageOrientationPatient tag in the dicom header and returns the corresponding image orientation (coronal, transversal, sagittal).

    Parameters
    ----------
    dicom_volume : TYPE
        A list of read dicom slices using the pydicom module.

    Returns
    -------
    cor : boolean
        True if the ImageOrientation tag has coronary directions. Else is False.
    sag : boolean
        True if the ImageOrientation tag has sagittal directions. Else is False.
    tra : boolean
        True if the ImageOrientation tag has transversal directions. Else is False.

    """
    cor, sag, tra = False, False, False # Define the variables
    
    # Round the vectors in case the image plane is slightly angled
    image_orientation = np.round(dicom_volume[slice][0x0020, 0x0037].value)
    image_orientation = [int(value) for value in image_orientation]
    
    # Check the direction of the image. The conditions in the if statement correspond to the vectors that represent the different directions.
    if image_orientation == [1, 0, 0, 0, 0, -1]:
        cor = True
    elif image_orientation == [0, 1, 0, 0, 0, -1]:
        sag = True
    elif image_orientation == [1, 0, 0, 0, 1, 0]:
        tra = True
        
    return cor, sag, tra

def get_pixel_map(dicom_volume, slice=0):
    """
    Creates a pixel map, an array where the elements are the coordinates of that particular pixel in a coordinate system.

    Parameters
    ----------
    dicom_volume : list
        List of dicom slices read using the pydicom module.
    slice : int, optional
        The slice to be used. Should be a value that can be used as an index for the dicom_volume list. The default is 0.

    Returns
    -------
    pixel_map : numpy array
        An array where the elements are the coordinates of that particular pixel in a coordinate system..

    """
    image_shape = t2_volume[slice].pixel_array.shape # Get the dimensions of the image
    pixel_map = np.empty(image_shape, dtype="O") # Create the empty pixel map, that will be filled with data type "object" (in our case, lists)
    
    cor, sag, tra = get_image_orientation(dicom_volume) # Get the orientation of the image to know how to position the different coordinates (x, y, slice)
    
    # Create a list of tuples with all the indices
    indices = []
    for i in range(image_shape[0]):
        for j in range(image_shape[1]):
            if cor:
                indices.append([i,slice,j])
            elif sag:
                indices.append([slice, i, j])
            elif tra:
                indices.append([i, j, slice])
    
    # Set the fill the index map with the indices in the correct positions
    for index in indices:
        if cor:
            pixel_map[index[0], index[2]] = index
        elif sag:
            pixel_map[index[1], index[2]] = index
        elif tra:
            pixel_map[index[0], index[1]] = index
    
    return pixel_map

def transform_pixel_map(pixel_map, dicom_volume, inverse=False, position_slice=0):
    """
    Transforms the coordinates in a pixel map to the patient coordinate system.

    Parameters
    ----------
    pixel_map : numpy array
        Array where the elements are the coordinates of the pixels in a coordinate system.
    dicom_volume : list
        List of dicom slices read using the pydicom module.
    inverse : boolean, optional
        Whether to transform to or from the patient coordinate system. Using the value True will transform from the patient system to the image volume system. The default is False, which transforms from the image volume system to the patient coordinate system.
    position_slice : int, optional
        The slice to be used for the ImagePositionPatient tag. Should be a slice which contains the upper left corner voxel of the entire imaging volume. I.e. slice 0. This variable is more important when the slices are not ordered correctly in the dicom_volume list. Should be a value that can be used as an index for the dicom_volume list. The default is 0, which should be fine for most applications.

    Returns
    -------
    transformed_pixel_map : numpy array
        A pixel map where the indices are transformed to another coordinate system.

    """
    # Store the original shape of the image and flatten the array
    original_shape = pixel_map.shape
    pixel_map_flattened = pixel_map.flatten()
    
    # Loop over the array and transform the coordinates one by one using the coordinate_transform_to_patient_system function
    for i in tqdm.tqdm(range(len(pixel_map_flattened))):
        new_pixel_coordinate = coordinate_transformation_to_patient_system(pixel_map_flattened[i], dicom_volume, position_slice=position_slice, inverse=inverse)
        pixel_map_flattened[i] = new_pixel_coordinate
        
    transformed_pixel_map = np.reshape(pixel_map_flattened, original_shape) # Un-flatten the image by reshaping it to its original shape
    
    return transformed_pixel_map

def coordinate_transformation_to_patient_system(initial_coordinates, dicom_volume, position_slice=0, inverse=False):
    """
    Transforms the coordinates of a pixel/voxel in an imaging volume to the patient coordinate system. Can also do the reverse transformation.

    Parameters
    ----------
    initial_coordinates : list
        List of the coordinates to be transformed. E.g. [slice, y, x] for a sagittal image plane
    dicom_volume : list
        List of image slices read using the pydicom module.
    position_slice : int, optional
        The slice to be used for the ImagePositionPatient tag. Should be a slice which contains the upper left corner voxel of the entire imaging volume. I.e. slice 0. This variable is more important when the slices are not ordered correctly in the dicom_volume list. Should be a value that can be used as an index for the dicom_volume list. The default is 0, which should be fine for most applications.
    inverse : boolean, optional
        Whether to transform to or from the patient coordinate system. Using the value True will transform from the patient system to the image volume system. The default is False, which transforms from the image volume system to the patient coordinate system.

    Returns
    -------
    transformed_coordinates : list
        List containing the transformed coordinates.

    """
    # Determine the image orientation
    cor, sag, tra = get_image_orientation(dicom_volume)
    
    i, j, k = initial_coordinates # Pixel coordinates
    pixel_spacing1, pixel_spacing2 = dicom_volume[0][0x0028, 0x0030].value # Pixel spacings
    slice_thickness = dicom_volume[0][0x0018, 0x0050].value # Slice thickness

    pixel_vector = np.array([i, j, k, 1]) # The vector to be transformed

    # Build the transformation matrix
    image_orientation = dicom_volume[position_slice][0x0020, 0x0037].value # Get the vectors that make the imaging plane
    
    # Create the directional vectors that will represent the base vectors of the current image coordinate system relative to the patient system
    row_dir_cos_vector = np.array([image_orientation[0], image_orientation[1], image_orientation[2], 0]) # Note that an extra 0 is added in the end. Check the documentation for the ImageOrientationPatient tag for the transformation equation
    col_dir_cos_vector = np.array([image_orientation[3], image_orientation[4], image_orientation[5], 0])
    slice_dir_cos_vector = np.array(np.cross(np.array([image_orientation[0], image_orientation[1], image_orientation[2]]),
                                         np.array([image_orientation[3], image_orientation[4], image_orientation[5]])))
    slice_dir_cos_vector = np.append(slice_dir_cos_vector, 0) # Note that slice_dir_cos_vector is the cross product of row_dir_cos_vector and col_dir_cos_vector
    
    s_vector = np.array(list(dicom_volume[position_slice][0x0020, 0x0032].value) + [1]) # Vector with the ImagePositionPatient information
    
    # Arrange the vectors in the correct order depending on the image orientation
    # This is the most confusing part, how the coordinate systems relate to each other
    # If the code does not produce the intended results, the problem could be here.
    # Try to think about how the coordinate systems are defined and how they relate to each other.
    if cor:
        M = np.array([row_dir_cos_vector*pixel_spacing1,
                      slice_dir_cos_vector*slice_thickness,
                      -col_dir_cos_vector*pixel_spacing2, # If not happy with the results, try to remove the minus sign
                      s_vector])
    elif sag:
        M = np.array([slice_dir_cos_vector*slice_thickness,
                      row_dir_cos_vector*pixel_spacing1,
                      col_dir_cos_vector*pixel_spacing2, # If not happy with the results, try to remove the minus sign
                      s_vector])
    elif tra:
        M = np.array([row_dir_cos_vector*pixel_spacing1,
                      col_dir_cos_vector*pixel_spacing2,
                      -slice_dir_cos_vector*slice_thickness, # If not happy with the results, try to remove the minus sign
                      s_vector])
    
    # Whether to transform to or from the patient system
    if inverse:
        transformed_coordinates = np.matmul(np.linalg.inv(M.T), pixel_vector)
    else:
        transformed_coordinates = np.matmul(M.T, pixel_vector)
    
    # The transformation produces a 4 element vector. We are only interested in a 3 element vector, hence skip the last element
    transformed_coordinates = list(transformed_coordinates[:-1])
    
    return transformed_coordinates


slice = 0
pixel_map = get_pixel_map(t2_volume, slice=slice) # Create a pixel map
pixel_map_b1 = get_pixel_map(b1_volume, slice=slice)
transformed_pixel_map = transform_pixel_map(pixel_map, t2_volume, position_slice=slice) # Transform the pixel map to the patient system
b1_pixel_map = transform_pixel_map(transformed_pixel_map, b1_volume, inverse=True) # Use the b1 dicom information and revert the transformation to go from the patient system to the b1 imaging system
# You now have a pixel map of an image in the t2-map, and now where to retrieve the pixels from the b1-map for correction of t2 values.

# %% Manipulate the b1 pixel map

# Can be useful to shift through slices in combined volumes

for voxel in b1_pixel_map.flatten():
    voxel[0] += 1
    
    

# %% interpolate b1 image volume

from scipy.interpolate import interpn
import time

start_time = time.time()

points0 = [i for i in range(0, b1_image_volume.shape[0])]
points1 = [i for i in range(0, b1_image_volume.shape[1])]
points2 = [i for i in range(0, b1_image_volume.shape[2])]
points = (points0, points1, points2) # Tuple av koordinaterna från b1_image_volume.shape, som behövs för interpn-funktionen

# Har försökt använda multiprocessing, ingen tidsvinst. Bättre att bara köra med en process.
b1_image_volume_interpolated = np.array([interpn(points, b1_image_volume, voxel, bounds_error=False, fill_value=0) for voxel in tqdm.tqdm(b1_pixel_map.flatten())])


print("")
print("--- {} seconds ---".format(time.time() - start_time))

# %% Save interpolated array to file

np.savetxt("/media/ivan/34F2B718F2B6DCF4/Users/Ivan/Documents/MSFP02 klinisk praktik/MR Lund/Projekt koordinater/foo_plus_p1.csv", b1_image_volume_interpolated, delimiter=",")

# %% Read interpolated array from file

b1_image_volume_interpolated = np.loadtxt("/media/ivan/34F2B718F2B6DCF4/Users/Ivan/Documents/MSFP02 klinisk praktik/MR Lund/Projekt koordinater/foo_plus.csv", delimiter=",")

# %% Reshape and show image from interpolated array

import matplotlib.pyplot as plt

interpolated_b1 = b1_image_volume_interpolated.reshape(384,384)
fig, ax = plt.subplots()
ax.imshow(interpolated_b1, cmap="gray")
plt.axis("off")
fig.tight_layout()
fig.savefig("/media/ivan/34F2B718F2B6DCF4/Users/Ivan/Documents/MSFP02 klinisk praktik/MR Lund/Projekt koordinater/interpolated_slice.tiff")

 # %% Alternative way to get the angled slice, without interpolation

import matplotlib.pyplot as plt

# Create a matrix with the same shape as the t2 slice
image_shape = t2_volume[0].pixel_array.shape
new_image = np.empty(image_shape)
new_image_flattened = new_image.flatten()
b1_pixel_map_flattened = b1_pixel_map.flatten()

# Loop over the b1_pixel_map and extract values from the b1 image and place them in the new image.
for i in tqdm.tqdm(range(len(new_image_flattened))):
    indices = [int(value) for value in b1_pixel_map_flattened[i]]
    if all(value < 240 for value in indices):
        new_image_flattened[i] = b1_image_volume[indices[0], indices[1], indices[2]]
new_image = new_image_flattened.reshape(image_shape)

fig, ax = plt.subplots()
ax.imshow(new_image, cmap="gray")
ax.axis("off")
fig.tight_layout()
