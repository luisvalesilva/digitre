# -*- coding: utf-8 -*-
"""
    Digitre
    ~~~~~~~
    A simple application...
    :copyright: (c) 2016 by Luis Vale Silva.
    :license: MIT, see LICENSE for more details.
"""

__author__ = "Luis Vale Silva"
__status__ = "Development"


from flask import Flask, render_template, request, jsonify
import os
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
import numpy as np
import prep_digit_image as prep


app = Flask(__name__)

cwd = os.path.dirname(__file__)

# Prepare pretrained classifier
cnn = input_data(shape=[None, 28, 28, 1], name='input')
cnn = conv_2d(cnn, 32, 5, activation='relu', regularizer="L2")
cnn = max_pool_2d(cnn, 2)
cnn = local_response_normalization(cnn)
cnn = conv_2d(cnn, 64, 5, activation='relu', regularizer="L2")
cnn = max_pool_2d(cnn, 2)
cnn = local_response_normalization(cnn)
cnn = fully_connected(cnn, 1024, activation='relu')
cnn = dropout(cnn, 0.5)
cnn = fully_connected(cnn, 10, activation='softmax')
cnn = regression(cnn, optimizer='adam', learning_rate=0.01,
                 loss='categorical_crossentropy', name='target')

# Load the tflearn pre-trained model
model = tflearn.DNN(cnn, tensorboard_verbose=0)
model.load(os.path.join(cwd, 'cnn.tflearn'))


def preprocess_digit_image(base64_str):
    digit = prep.b64_str_to_np(base64_str)
    digit = prep.crop_img(digit)
    digit = prep.resize_img(digit)
    digit = prep.min_max_scaler(digit, final_range=(0, 1))
    return np.reshape(digit, (1, 28, 28, 1), order='C')

def classify(digit):
    return model.predict(digit)


@app.route('/_get_digit')
def get_digit():
    """Get digit drawn by user as base64 image and recognize it"""
    digit = request.args.get('digit', 0, type=str)
    digit = preprocess_digit_image(digit)
    prediction = classify(digit)[0]
    max_idx = np.argmax(prediction)
    max_val = prediction[max_idx]
    prob = np.around(max_val, 3) * 100
    prediction = '{} ({} % probability)'.format(str(max_idx), str(prob))
    return jsonify(result=prediction)

@app.route('/')
def index():
    """Render landing page"""
    labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    values = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
    return render_template('index.html', values=values, labels=labels)

@app.route('/about')
def about():
    return render_template('about.html')


if __name__ == '__main__':
    app.run(debug=True)
