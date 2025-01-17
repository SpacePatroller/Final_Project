from __future__ import division, print_function

# import necessary libraries
from flask import Flask, jsonify, render_template, redirect, url_for, request
from flask_sqlalchemy import SQLAlchemy

from werkzeug.utils import secure_filename
import sqlalchemy
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import Session
from sqlalchemy import create_engine

import sys
import os
import glob
import re
import pandas as pd
import numpy as np

import keras
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array
from keras.applications.xception import (
    Xception, preprocess_input, decode_predictions)
from keras import backend as K


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

model = None
graph = None

def load_model1():
    global model
    global graph
    model = keras.models.load_model("models/skin_model_delta.h5")
    graph = K.get_session().graph


load_model1()


def prepare_image(img):
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    # return the processed image
    return img


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    diagnosis = []
    # data = {"success": False}
    if request.method == 'POST':
        print(request)

        if request.files.get('file'):
            # read the file
            file = request.files['file']

            # create a path to the uploads folder
            basepath = os.path.dirname(__file__)
            filepath = os.path.join(
                basepath, 'uploads', secure_filename(file.filename))

            # Save the file to the uploads folder
            file.save(filepath)

            # Load the saved image using Keras and resize it 
            image_size = (75, 100)
            im = keras.preprocessing.image.load_img(filepath,
                                                    target_size=image_size,
                                                    grayscale=False)

            # preprocess the image and prepare it for classification
            image = prepare_image(im)

            global graph
            with graph.as_default():

                labels = ['Melanocytic nevi (Benign)', 'Melanoma (Malignant)', 'Benign keratosis-like lesions (Benign)', 'Basal cell carcinoma (Malignant)',
	                          'Actinic keratoses (Malignant)', 'Vascular lesions (Benign)', 'Dermatofibroma (Benign)']

                labels = tuple(labels)

                global preds
                preds = model.predict(image)

                # convert preds array to list
                preds = preds.tolist()

                # convert list of lists to one list for rounding to work
                flat_preds = [item for sublist in preds for item in sublist]

                updated_preds = list(
                    map(lambda x: (round(x*100, 3)), flat_preds))

                dictionary = dict(zip(labels, updated_preds))

                # create a function which returns the value of a dictionary

                def keyfunction(k):
                    return dictionary[k]

            # sort by dictionary by the values and print top 3 {key, value} pairs
            for key in sorted(dictionary, key=keyfunction, reverse=True)[:3]:

                if dictionary[key] > 0:
                    diagnosis.append([key, str(dictionary[key]) + "%"])
                    
            print(diagnosis)
            return jsonify(diagnosis)

    return jsonify(diagnosis)


if __name__ == "__main__":
    app.run(port=5002, debug=False, threaded=False)

    # Serve the app with gevent
    # http_server = WSGIServer(('', 5000), app)
    # http_server.serve_forever()
