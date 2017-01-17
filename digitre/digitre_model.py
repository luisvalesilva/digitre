# -*- encoding: utf-8 -*-

"""
    Digitre
    ~~~~~~~
    A simple Machine Learning application to recognize handwritten digits.

    digitre_model.py includes functionality to define and fit the handwritten digit
    classifier.

    Note
    ----
    Throws an error with latest stable TensorFlow version (0.12):
        "ValueError: No variables to save".

    Needed to revert TF back to version 0.11 for this to work:
        $ pip uninstall protobuf
        $ pip uninstall tensorflow

        # Ubuntu/Linux 64-bit, CPU only, Python 3.5:
        $ export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.11.0rc1-cp35-cp35m-linux_x86_64.whl
        # Mac OS X, CPU only, Python 3.4 or 3.5:
        $ export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-0.11.0rc1-py3-none-any.whl

        $ pip install $TF_BINARY_URL


    :copyright: (c) 2016 by Luis Vale Silva.
    :license: MIT, see LICENSE for more details.
"""

__author__ = "Luis Vale Silva"
__status__ = "Development"

import os
import sys
import time
import numpy as np

import digitre_preprocessing as prep

from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
import tflearn.datasets.mnist as mnist
from tflearn.models.dnn import DNN

def load_data():
    """
    Get MNIST training data distributed with TFLearn.

    Returns
    -------
    Tuple containing four object:
        X: Training data
        Y: Training labels
        testX: Test data
        testY: Test labels

    """
    import tflearn.datasets.mnist as mnist

    X, Y, testX, testY = mnist.load_data(one_hot=True)
    X = X.reshape([-1, 28, 28, 1])
    testX = testX.reshape([-1, 28, 28, 1])

    return X, Y, testX, testY

def build():
    """
    Build classification model.

    Returns
    -------
    Defined machine learning model
    """
    ### Build CNN model
    cnn = input_data(shape=[None, 28, 28, 1], name='input')
    # 1st convolutional layer:
    # Convolution computing 32 features for each 5x5 patch
    # Stride of one and zero-padded convolutions (defaults)
    cnn = conv_2d(cnn, 32, 5, activation='relu', regularizer="L2")
    # Max pooling over 2x2 blocks
    cnn = max_pool_2d(cnn, 2)
    cnn = local_response_normalization(cnn)
    # 2nd convolutional layer:
    # Convolution computing 64 features for each 5x5 patch
    cnn = conv_2d(cnn, 64, 5, activation='relu', regularizer="L2")
    cnn = max_pool_2d(cnn, 2)
    cnn = local_response_normalization(cnn)
    # Fully connected layer
    cnn = fully_connected(cnn, 1024, activation='relu')
    cnn = dropout(cnn, 0.5)
    cnn = fully_connected(cnn, 10, activation='softmax')
    cnn = regression(cnn, optimizer='adam', learning_rate=0.01,
                     loss='categorical_crossentropy', name='target')

    # Define model
    model = DNN(cnn, tensorboard_verbose=0)

    return model

def fit(model, X, Y, testX, testY, n_epoch=20, run_id='cnn_mnist'):
    """
    Fit (train) classification model.

    Parameters
    ----------
    model: TFLearn model
        Built and defined model.
    X: numpy ndarray, shape=(55000, 28, 28, 1)
        Training data
    Y: numpy ndarray, shape=(55000, 10)
        Training labels
    testX: numpy ndarray, shape=(10000, 28, 28, 1)
        Test data
    testY: numpy ndarray, shape=(10000, 10)
        Test labels
    n_epoch: int, default=20
        Number of training epochs
    run_id: str, default='cnn_mnist'
        ID to assign the current run
    """
    model.fit({'input': X}, {'target': Y}, n_epoch=n_epoch,
              validation_set=({'input': testX}, {'target': testY}),
              snapshot_step=100, show_metric=True, run_id=run_id)

def save(model, file_name='cnn.tflearn'):
    """
    Save trained classification model in current woring directory.

    Parameters
    ----------
    model: TFLearn model
        Trained model
    file_name: string, default='cnn.tflearn'
        Name to assign model file written to disk
    """
    cwd = os.path.dirname(__file__)
    model.save(os.path.join(cwd, file_name))


def main():
    t0 = time.time()

    print('-------------------------------------')
    print('               DIGITRE')
    print('         Training classifier')
    print('-------------------------------------')
    print()

    print('... Loading training and test data')
    X, Y, testX, testY = load_data()

    if args.all_data:
        print('... Merging train and test data for training on all data')
        X = np.concatenate((X, testX), axis=0)
        Y = np.concatenate((Y, testY), axis=0)

    print('... Building model')
    model = build()

    print('... Training model')
    if not args.n_epoch:
        args.n_epoch = 20
    t1 = time.time()
    fit(model, X, Y, testX, testY, n_epoch=int(args.n_epoch))
    print('-----')
    print('Completed training in')
    prep.print_elapsed_time(t1)
    print('-----')

    print('... Saving trained model as "', args.file_name, '"')
    save(model, file_name=args.file_name)

    print()
    print()
    print('--')
    print('Completed in')
    prep.print_elapsed_time(t0)
    print('Bye!')
    print('-------------------------------------')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train handwritten digit classifier.')
    parser.add_argument('-f', '--file_name', help=('name of model file written to disk'),
                        required=True)
    parser.add_argument('-a', '--all_data', action='store_true',
                        help=('whether to train on all 65,000 samples'),
                        default=False)
    parser.add_argument('-e', '--n_epoch', help=('number of training epochs'))
    parser.add_argument('-v', '--version', action='version', version='%(prog)s 0.1')

    # Also print help if no arguments are provided
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    main()
