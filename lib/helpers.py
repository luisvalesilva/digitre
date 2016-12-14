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
from PIL import Image
import cv2
from StringIO import StringIO
import numpy as np

def b64_str_to_np(base64_str):
    """Get digit drawn by user as base64 image and convert to numpy array"""
    _, base64_str = base64_str.split(',')
    
    buf = StringIO()
    sbuf.write(base64.base64decode(base64_str))
    pimg = Image.open(sbuf)
    img = np.array(pimg)


    #img = base64.b64decode(base64_str)
    #img = np.asarray(img, dtype="float64")
    #i = io.BytesIO(i)
    #q = np.frombuffer(r, dtype=np.float64)
    print(img)
    return(img)








def main():
    """
    Main function
    """
    # ...
    


if __name__ == '__main__':
    main()
