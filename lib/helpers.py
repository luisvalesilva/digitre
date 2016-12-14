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
    _, base64_str = base64_str.split(',')
    buf = BytesIO()
    buf.write(base64.b64decode(base64_str))
    pimg = Image.open(buf)
    img = np.array(pimg)

    print('Dimesnions of numpy ndarray: ', img.shape)
    return(img)








def main():
    """
    Main function
    """
    # ...
    


if __name__ == '__main__':
    main()
