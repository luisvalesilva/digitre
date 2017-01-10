# -*- encoding: utf-8 -*-

"""
    Digitre
    ~~~~~~~
    A simple Machine Learning application to recognize handwritten digits.

    digitre_preprocessing.py includes functionality to preprocess base64-encoded
    images of handwritten digits and get them ready for classification.

    :copyright: (c) 2016 by Luis Vale Silva.
    :license: MIT, see LICENSE for more details.
"""

__author__ = "Luis Vale Silva"
__status__ = "Development"

import time
import base64
from io import BytesIO
from PIL import Image
import numpy as np

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
    Image as 3D numpy ndarray
    """

    if "base64" in base64_str:
        _, base64_str = base64_str.split(',')

    buf = BytesIO()
    buf.write(base64.b64decode(base64_str))
    pimg = Image.open(buf)
    img = np.array(pimg)

    # Keep only 4th pixel value in 3rd dimension (first 3 are all zeros)
    return img[:, :, 3]

def crop_img(img_ndarray):
    """
    Crop white space around digit.

    Parameters
    ----------
    img_ndarray: 3D numpy ndarray, shape=[var, var, 1] (depending on canvas size)
        Image to crop (drawn by user).
    Returns
    -------
    Cropped image as 3D numpy ndarray
    """
    # Length of zero pixel values for rows and columns
    # Across rows
    first_row = np.nonzero(img_ndarray)[0].min()
    last_row = np.nonzero(img_ndarray)[0].max()
    middle_row = int(np.mean([last_row, first_row]))
    # Across cols
    first_col = np.nonzero(img_ndarray)[1].min()
    last_col = np.nonzero(img_ndarray)[1].max()
    middle_col = int(np.mean([last_col, first_col]))

    # Crop by longest non-zero to make sure all is kept
    row_length = last_row - first_row
    col_length = last_col - first_col
    length = max(row_length, col_length)
    # Minimum size of 28x28
    length = max(length, 28)

    # Get half length to add to middle point (add some padding: 1px)
    half_length = int(length / 2) + 1

    # Make sure even the shorter dimension is centered
    first_row = middle_row - half_length
    last_row = middle_row + half_length
    first_col = middle_col - half_length
    last_col = middle_col + half_length

    # Crop image
    img_ndarray = img_ndarray[first_row:last_row, first_col:last_col]
    # Add padding (15px of zeros)
    return np.lib.pad(img_ndarray, 15, 'constant', constant_values=(0))


def resize_img(img_ndarray):
    """
    Resize digit image to 28x28.

    Parameters
    ----------
    img_ndarray: 3D numpy ndarray, shape=[var, var, 1] (depending on drawing)
        Image to resize.
    Returns
    -------
    Resized image as 3D numpy ndarray
    """
    img = Image.fromarray(img_ndarray)
    img.thumbnail((28, 28), Image.ANTIALIAS)

    return np.array(img)


def min_max_scaler(img_ndarray, final_range=(0, 1)):
    """
    Scale and transform feature values to given range.

    Parameters
    ----------
    img_ndarray: 3D numpy ndarray, shape=[28, 28, 1]
        Image to scale and transform.
    final_range: tuple (min, max), default=(0, 1)
        Desired range of transformed data.
    Returns
    -------
    Scaled and transformed image as 3D numpy ndarray
    """
    px_min = final_range[0]
    px_max = final_range[1]

    #img_std = (img_ndarray - img_ndarray.min(axis=0)) / (img_ndarray.max(axis=0) - img_ndarray.min(axis=0))
    # Hard code pixel vlue range
    img_std = img_ndarray / 255
    return img_std * (px_max - px_min) + px_min
