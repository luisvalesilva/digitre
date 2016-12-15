#!/usr/bin/env python
# coding:utf-8

"""
    Digitre
    ~~~~~~~
    A simple application...
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

    print('Dimensions of numpy ndarray: ', img.shape)

    # Keep only 4th pixel value in 3rd dimension (first 3 are all zeros)
    return img[:,:,3]

def lowest_non_zero(x):
    """Determine index of first non-zero value"""
    return np.nonzero(x)[0][0]


def crop_and_resize(img_ndarray):
    """Crop white space around digit and resize to 28x28"""
    # Length of zero pixel values for rows and columns
    # Across rows
    first_row = np.nonzero(img_ndarray)[0].min()
    last_row = np.nonzero(img_ndarray)[0].max()
    
    # Across cols
    first_col = np.nonzero(img_ndarray)[1].min()
    last_col = np.nonzero(img_ndarray)[1].max()

    # Crop by longest non-zero to make sure all is kept
    first = min(first_row, first_col)
    last = max(last_row, last_col)

    print(last-first)
    # Minimum size of 28x28
    if last-first < 28:
        if first + 28 < 200 :
            last = first + 28
        else:
            first = last - 28

    # Crop image
    img_ndarray = img_ndarray[first:last, first:last]

    return img_ndarray.thumbnail((28, 28), Image.ANTIALIAS)



def flatten_img(img_ndarray):
    """Flatten digit image as numpy ndarray to 1D"""

    return img_ndarray.flatten()