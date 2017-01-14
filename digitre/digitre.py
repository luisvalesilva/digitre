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


from flask import Flask, render_template, request, jsonify
import digitre_classifier
import numpy as np

app = Flask(__name__)

# Instantiate Classifier
model = digitre_classifier.Classifier(file_name='cnn.tflearn')

@app.route('/_get_digit', methods=['POST'])
def get_digit():
    """Get digit drawn by user as base64 image and recognize it"""
    digit = request.get_json(force=True).get('digit', '')
    #digit = request.args.get('digit', 0, type=str)
    digit = model.preprocess(digit)
    prediction = model.classify(digit)[0]
    # Get class with highest probability
    max_idx = np.argmax(prediction)
    max_val = prediction[max_idx]
    prob = np.around(max_val, 3) * 100
    # Build output strings
    if prob < 70:
        prediction = 'Huh...'
        probability = 'You call that a digit?'
    else:
        prediction = '{}'.format(str(max_idx))
        probability = '({}% probability)'.format(str(prob))

    return jsonify(result=prediction, probability=probability)

@app.route('/')
def index():
    """Render landing page"""
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
