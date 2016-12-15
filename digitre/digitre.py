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
import prep_digit_image as prep

app = Flask(__name__)


@app.route('/_get_digit')
def get_digit():
    """Get digit drawn by user as base64 image and convert to numpy array"""
    digit = request.args.get('digit', 0, type=str)
    prep.b64_str_to_np(digit)
    return jsonify(result=digit)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')


if __name__ == '__main__':
    app.run(debug=True)
