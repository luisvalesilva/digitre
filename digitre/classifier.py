# -*- encoding: utf-8 -*-

"""
    Digitre
    ~~~~~~~
    A simple Machine Learning application to recognize handwritten digits.

    :copyright: (c) 2016 by Luis Vale Silva.
    :license: MIT, see LICENSE for more details.
"""

__author__ = "Luis Vale Silva"
__status__ = "Development"

import base64
from io import BytesIO
from PIL import Image
import numpy as np

class Classifier(object):
    """Given base64-encoded image, transforms it to appropriate format and predicts digit class.

    Classifier prepares base64-encoded image of digit (hopefully...) handwritten in html canvas by:
        . Converting it to numpy 3D array
        . Cropping it to minimum square size around drawing (minimum of 28 x 28)
        . Resizing it to 28 x 28
        . MinMax transforming 0 to 255 pixel values to 0 to 1
    It then uses pre-trained CNN model to predict digit class (0 to 9)

    Parameters
    ----------
    feature_range : tuple (min, max), default=(0, 1)
        Desired range of transformed data.
    copy : boolean, optional, default True
        Set to False to perform inplace row normalization and avoid a
        copy (if the input is already a numpy array).
    
    Attributes
    ----------
    min_ : ndarray, shape (n_features,)
        Per feature adjustment for minimum.
    scale_ : ndarray, shape (n_features,)
        Per feature relative scaling of the data.
        .. versionadded:: 0.17
           *scale_* attribute.
    data_min_ : ndarray, shape (n_features,)
        Per feature minimum seen in the data
        .. versionadded:: 0.17
           *data_min_* instead of deprecated *data_min*.
    data_max_ : ndarray, shape (n_features,)
        Per feature maximum seen in the data
        .. versionadded:: 0.17
           *data_max_* instead of deprecated *data_max*.
    data_range_ : ndarray, shape (n_features,)
        Per feature range ``(data_max_ - data_min_)`` seen in the data
        .. versionadded:: 0.17
           *data_range_* instead of deprecated *data_range*.
    """

    def __init__(self, model_pickle, tfidf_pickle, stopwords_pickle):
        # Load the model, the text vectorizer, and the stopwords
        self.model = pickle.load(open(model_pickle))
        self.tfidf = pickle.load(open(tfidf_pickle))
        self.stopwords = pickle.load(open(stopwords_pickle))

        # Label dictionary for nice categories
        self.label_dict = {0: "Arts", 1: "Business", 2: "Food", 3: "Health", 4: "NY", 5: "Politics", 6: "RealEstate", 7: "Science", \
             8: "Sports", 9: "Style", 10: "Tech", 11: "Travel", 12: "US", 13: "World"}
        # Label dictionary for categories as they need to be put into the NYT Top Stories API
        self.label_dict_NYT = {0: "arts", 1: "business", 2: "dining", 3: "health", 4: "nyregion", 5: "politics", 6: "realestate", 7: "science", \
             8: "sports", 9: "fashion", 10: "technology", 11: "travel", 12: "national", 13: "world"}

        # Set up the Twitter access
        auth = tweepy.OAuthHandler(TW_CON_SECRET_KEY, TW_CON_SECRET)
        auth.set_access_token(TW_TOKEN_KEY, TW_TOKEN)

        # Set up the Twitter API
        self.api = tweepy.API(auth)

        # Helper list of all single alphabetic letters
        self.singleletters = [chr(i) for i in range(97,123)] + [chr(i).upper() for i in range(97,123)]


    def b64_str_to_np(self, base64_str):
        """Get digit drawn by user as base64 image and convert to numpy array.
        
        Parameters
        ----------
        base64_str: string
            String of base64-encoded image of user drawing in html canvas.
        Returns
        -------
        Image as numpy 3D array
        """

        if "base64" in base64_str:
            _, base64_str = base64_str.split(',')

        buf = BytesIO()
        buf.write(base64.b64decode(base64_str))
        pimg = Image.open(buf)
        img = np.array(pimg)

        # Keep only 4th pixel value in 3rd dimension (first 3 are all zeros)
        return img[:, :, 3]

    def crop_img(self, img_ndarray):
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
        img_ndarray = np.lib.pad(img_ndarray, 15, 'constant', constant_values=(0))

        return img_ndarray


    def resize_img(self, img_ndarray):
        """Resize digit to 28x28"""
        img = Image.fromarray(img_ndarray)
        img.thumbnail((28, 28), Image.ANTIALIAS)

        return np.array(img)


    def min_max_scaler(self, img_ndarray, final_range=(0, 1)):
        """Scale features to given range"""
        px_min = final_range[0]
        px_max = final_range[1]
        img_std = img_ndarray / 255 # Hard code pixel value range

        return img_std * (px_max - px_min) + px_min
