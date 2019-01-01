# -*- encoding: utf-8 -*-

"""
    Digitre
    ~~~~~~~
    A simple Machine Learning application to recognize handwritten digits.

    digitre_preprocessing.py includes functionality to preprocess base64-encoded
    images of handwritten digits and get them ready for classification.

    :copyright: (c) 2017 by Luis Vale Silva.
    :license: MIT, see LICENSE for more details.
"""

__author__ = "Luis Vale Silva"
__status__ = "Development"

import time
import base64
from io import BytesIO
from PIL import Image
import numpy as np
from scipy import ndimage

def print_elapsed_time(t0):
    """
    Given a start time (time.time() object), computes and prints elapsed time.

    Parameters
    ----------
    t0: output of time.time()
        Start time to compute elapsed time from (no default)
    Returns
    ----------
    Elapsed time as string
    """
    elapsed_time = time.time() - t0

    if elapsed_time < 60:
        print("{:2.1f} sec.".format(elapsed_time))
    elif 60 < elapsed_time < 3600:
        print("{:2.1f} min.".format(elapsed_time / 60))
    else:
        print("{:2.1f} hr.".format(elapsed_time / 3600))

def b64_str_to_np(base64_str):
    """
    Get digit drawn by user as base64 image and convert to numpy array.

    Parameters
    ----------
    base64_str: string
        String of base64-encoded image of user drawing in html canvas.
    Returns
    -------
    Image as 2D numpy ndarray
    """
    base64_str = str(base64_str)
    if "base64" in base64_str:
        _, base64_str = base64_str.split(',')

    buf = BytesIO()
    buf.write(base64.b64decode(base64_str))
    buf.seek(0)
    pimg = Image.open(buf)
    img = np.array(pimg)

    # Keep only 4th value in 3rd dimension (first 3 are all zeros)
    return img[:, :, 3]


def crop_img(img_ndarray):
    """
    Crop white space around digit.

    Parameters
    ----------
    img_ndarray: 2D numpy ndarray, shape=(var, var) (determined by canvas size)
        Image to crop (drawn by user).
    Returns
    -------
    Cropped image as 2D numpy ndarray
    """
    # Length of zero pixel values for rows and columns
    # Across rows
    first_row = np.nonzero(img_ndarray)[0].min()
    last_row = np.nonzero(img_ndarray)[0].max()
    middle_row = np.mean([last_row, first_row])
    # Across cols
    first_col = np.nonzero(img_ndarray)[1].min()
    last_col = np.nonzero(img_ndarray)[1].max()
    middle_col = np.mean([last_col, first_col])

    # Crop by longest non-zero to make sure all is kept
    row_length = last_row - first_row
    col_length = last_col - first_col
    length = max(row_length, col_length)
    # Minimum size of 28x28
    length = max(length, 28)

    # Get half length to add to middle point (add some padding: 1px)
    half_length = (length / 2) + 1

    # Make sure even the shorter dimension is centered
    first_row = int(middle_row - half_length)
    last_row = int(middle_row + half_length)
    first_col = int(middle_col - half_length)
    last_col = int(middle_col + half_length)

    # Crop image
    return img_ndarray[first_row:last_row, first_col:last_col]


def center_img(img_ndarray):
    """
    Center digit on center of mass of the pixels.

    Parameters
    ----------
    img_ndarray: 2D numpy ndarray, shape=(var, var) (determined by digit size)
        Image to center (drawn by user).
    Returns
    -------
    Centered image as 2D numpy ndarray
    """
    # Compute center of mass and frame center
    com = ndimage.measurements.center_of_mass(img_ndarray)
    center = len(img_ndarray) / 2

    # Center by adding lines of zeros matching diff between
    # center of mass and frame center
    row_diff = int(com[0] - center)
    col_diff = int(com[1] - center)

    rows = np.zeros((abs(row_diff), img_ndarray.shape[1]))
    if row_diff > 0:
        img_ndarray = np.vstack((img_ndarray, rows))
    elif row_diff < 0:
        img_ndarray = np.vstack((rows, img_ndarray))

    cols = np.zeros((img_ndarray.shape[0], abs(col_diff)))
    if col_diff > 0:
        img_ndarray = np.hstack((img_ndarray, cols))
    elif col_diff < 0:
        img_ndarray = np.hstack((cols, img_ndarray))

    # Make image square again (add zero rows to the smaller dimension)
    dim_diff = img_ndarray.shape[0] - img_ndarray.shape[1]
    half_A = half_B = abs(int(dim_diff / 2))
    
    if dim_diff % 2 != 0:
        half_B += 1

    # Add half to each side (to keep center of mass)
    # Handle dim_diff == 1
    if half_A == 0:  # 1 line off from exactly centered
        if dim_diff > 0:
            half_B = np.zeros((img_ndarray.shape[0], half_B))
            img_ndarray = np.hstack((half_B, img_ndarray))
        else:
            half_B = np.zeros((half_B, img_ndarray.shape[1]))
            img_ndarray = np.vstack((half_B, img_ndarray))

    elif dim_diff > 0:
        half_A = np.zeros((img_ndarray.shape[0], half_A))
        half_B = np.zeros((img_ndarray.shape[0], half_B))
        img_ndarray = np.hstack((img_ndarray, half_A))
        img_ndarray = np.hstack((half_B, img_ndarray))
    else:
        half_A = np.zeros((half_A, img_ndarray.shape[1]))
        half_B = np.zeros((half_B, img_ndarray.shape[1]))
        img_ndarray = np.vstack((img_ndarray, half_A))
        img_ndarray = np.vstack((half_B, img_ndarray))
    # Add padding all around (15px of zeros)
    return np.lib.pad(img_ndarray, 15, 'constant', constant_values=(0))

def resize_img(img_ndarray):
    """
    Resize digit image to 28x28.

    Parameters
    ----------
    img_ndarray: 2D numpy ndarray, shape=(var, var) (depending on drawing)
        Image to resize.
    Returns
    -------
    Resized image as 2D numpy ndarray (28x28)
    """
    img = Image.fromarray(img_ndarray)
    img.thumbnail((28, 28), Image.ANTIALIAS)

    return np.array(img)


def min_max_scaler(img_ndarray, final_range=(0, 1)):
    """
    Scale and transform feature values to given range.

    Parameters
    ----------
    img_ndarray: 2D numpy ndarray, shape=(28, 28)
        Image to scale and transform
    final_range: tuple (min, max), default=(0, 1)
        Desired range of transformed data
    Returns
    -------
    Scaled and transformed image as 2D numpy ndarray
    """
    px_min = final_range[0]
    px_max = final_range[1]

    # Hard code pixel value range
    img_std = img_ndarray / 255
    return img_std * (px_max - px_min) + px_min

def reshape_array(img_ndarray):
    """
    Reshape image array for classifier.

    Parameters
    ----------
    img_ndarray: 2D numpy ndarray, shape=(28, 28)
        Image array to reshape.
    Returns
    -------
    Reshaped image numpy ndarray
    """
    digit = np.reshape(img_ndarray, (-1, 28, 28, 1), order='C')
    return digit
