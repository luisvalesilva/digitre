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
    Given base64-encoded image, transforms it to appropriate format and predicts
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
    """

    def __init__(self, file_name='cnn.tflearn'):

        cwd = os.path.dirname(__file__)

        # Define and load the model
        self.model = model.build()
        self.model.load(os.path.join(cwd, file_name))




        # Load the model, the text vectorizer, and the stopwords
        self.model = pickle.load(open(model_pickle))
        self.tfidf = pickle.load(open(tfidf_pickle))

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


    def preprocess_image(self, base64_str):
        """
        Get digit drawn by user as base64 image and convert it to numpy array.

        Parameters
        ----------
        base64_str: string
            String of base64-encoded image of user drawing in html canvas.
        Returns
        -------
        Image as numpy 3D array
        """

        pass

    def classify_image(self, blabla):
        """
        Get digit drawn by user as base64 image and convert to numpy array.

        Parameters
        ----------
        base64_str: string
            String of base64-encoded image of user drawing in html canvas.
        Returns
        -------
        Image as numpy 3D array
        """

        pass
