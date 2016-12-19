#!/usr/bin/env python3
# coding:utf-8

"""
    Digitre helpers
    ~~~~~~~~~~~~~~~
    Digitre helper functions for digit image preprocessing
    :copyright: (c) 2016 by Luis Vale Silva.
    :license: MIT, see LICENSE for more details.
"""

__author__ = "Luis Vale Silva"
__status__ = "Development"

import base64
from io import BytesIO
from PIL import Image
import numpy as np

def b64_str_to_np(base64_str):
    """Get digit drawn by user as base64 image and convert to numpy array"""

    if "base64" in base64_str:
        _, base64_str = base64_str.split(',')

    buf = BytesIO()
    buf.write(base64.b64decode(base64_str))
    pimg = Image.open(buf)
    img = np.array(pimg)

    # Keep only 4th pixel value in 3rd dimension (first 3 are all zeros)
    return img[:, :, 3]


def crop_img(img_ndarray):
    """Crop white space around digit"""
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
    # (add some padding: 1px)
    first = min(first_row, first_col) - 1
    last = max(last_row, last_col) + 1
    length = last - first

    # Minimum size of 28x28
    length = max(length, 28)

    half_length = int(length / 2)

    # Make sure even the shorter dimension is centered
    first_row = middle_row - half_length
    last_row = middle_row + half_length
    first_col = middle_col - half_length
    last_col = middle_col + half_length

    # Crop image
    img_ndarray[first_row:last_row, first_col:last_col]
    # Add padding (15px of zeros)
    return np.lib.pad(img_ndarray, 15, 'constant', constant_values=(0))


def resize_img(img_ndarray):
    """Resize digit to 28x28"""
    img = Image.fromarray(img_ndarray)
    img.thumbnail((28, 28), Image.ANTIALIAS)

    return np.array(img)


def min_max_scaler(img_ndarray, final_range=(0, 1)):
    """Scale features to given range"""
    min = final_range[0]
    max = final_range[1]

    img_std = (img_ndarray - img_ndarray.min(axis=0)) / (img_ndarray.max(axis=0) - img_ndarray.min(axis=0))
    return img_std * (max - min) + min

def flatten_img(img_ndarray):
    """Flatten digit image as numpy ndarray to 1D"""
    return img_ndarray.flatten()
