# digitre

**`Digitre`** is a machine learning application to recognize handwritten digits. **`Digitre`** is written in Python using the Flask web framework.

The front end shows an html canvas where you are asked to draw a digit. **`Digitre`** recognizes the handwritten digit using Machine Learning (ML) and outputs its prediction.

The ML model is a Convolutional Neural Network (CNN) trained on the [MNIST](http://yann.lecun.com/exdb/mnist/) dataset using the [TFLearn](http://tflearn.org/) software library (a high level abstraction of Google's [TensorFlow](https://www.tensorflow.org/)).

Give it a try at [digitre.technology](http://digitre.technology).

## Try it locally

Steps to download the source code and run Flask's development server locally.

1. Clone the repo and go inside
```shell
git clone https://github.com/luisvalesilva/digitre.git
cd digitre/
```

2. Create and activate a virtual environment
```shell
virtualenv venv
source venv/bin/activate
```

Due to some incompatibilities, I am avoiding the latest version of tensorflow (at the time of development; version 0.12) and using version 0.11 instead. The frozen state of environment packages in the `requirements.txt` file defaults to installation of tensorflow 0.11 for Linux (Ubuntu/Linux 64-bit, CPU only, Python 3.5).

If you're on Mac OS X, change the line referring to tensorflow at this point, in order to install the distribution for Mac OS X (CPU only, Python 3.4 or 3.5):
 ```shell
sed -i '12s/.*/https:\/\/storage.googleapis.com\/tensorflow\/mac\/cpu\/tensorflow-0.11.0rc1-py3-none-any.whl/' requirements.txt
```  

3. Install requirements (listed in frozen state of environment packages)
```shell
pip install -r requirements.txt
```

4. Run app locally with Flask's development server
```shell
python digitre/digitre.py
```
Go to address `http://127.0.0.1:5000/` on your web browser to use the app.


## License

This project is licensed under the terms of the MIT license. See [LICENSE](LICENSE) file for details.
