# -*- encoding: utf-8 -*-

"""
    Digitre
    ~~~~~~~
    A simple Machine Learning application to recognize handwritten digits.

    digitre_classifier.py includes a class with functionality to preprocess base64-encoded
    handwritten digit images and classify the digit in the image.

    :copyright: (c) 2016 by Luis Vale Silva.
    :license: MIT, see LICENSE for more details.
"""

__author__ = "Luis Vale Silva"
__status__ = "Development"

import os
import digitre_preprocessing as prep
import digitre_model

class Classifier(object):
    """
    Given base64-encoded image, transforms it to the appropriate format and predicts
    digit class.

    Classifier prepares base64-encoded image of handwritten digit (hopefully...) from
    html canvas by:
        . Converting it to numpy 3D array
        . Cropping it to square of minimum size around drawing (no smaller than of 28 x 28)
        . Resizing it to 28 x 28
        . MinMax transforming pixel values between 0 and 255 to values between 0 and 1
    It then uses pre-trained machine learning model to predict digit class, with the
    output being the probability distribution over the 10 classes (0 to 9).

    Parameters
    ----------
    file_name: str, default='cnn.tflearn'
        File name of pre-trained TFLearn model
    copy : boolean, optional, default True
        Set to False to perform inplace row normalization and avoid a
        copy (if the input is already a numpy array).
    """

    def __init__(self, file_name='cnn.tflearn'):
        cwd = os.path.dirname(__file__)
        # Load the model
        self.model = digitre_model.build()
        self.model.load(os.path.join(cwd, file_name))


    def preprocess(self, digit_image):
        """
        Get digit drawn by user as base64 image and convert it to numpy array.

        Parameters
        ----------
        digit_image: string
            String of base64-encoded image of user drawing in html canvas.
        Returns
        -------
        Processed image as numpy 3D array ready for classification
        """
        digit = prep.b64_str_to_np(digit_image)
        digit = prep.crop_img(digit)
        digit = prep.center_img(digit)
        digit = prep.resize_img(digit)
        digit = prep.min_max_scaler(digit, final_range=(0, 1))
        digit = prep.reshape_array(digit)
        return digit

    def classify(self, preprocessed_image):
        """
        Get digit drawn by user as base64 image and convert to numpy array.

        Parameters
        ----------
        preprocessed_image: 4D numpy ndarray, shape=(1, 28, 28, 1)
            Image array to classify.
        Returns
        -------
        Image as numpy 3D array
        """
        return self.model.predict(preprocessed_image)
