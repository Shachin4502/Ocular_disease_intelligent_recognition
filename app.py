from __future__ import division, print_function

# coding=utf-8
import os

import numpy as np
# Flask utils
from flask import Flask, request, render_template
# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from tensorflow.keras.preprocessing import image

from werkzeug.utils import secure_filename

# Define a flask app
app = Flask(__name__, template_folder='template')

# Load your trained model
model = load_model('odir.h5')
model.make_predict_function()
print('Model loaded. Check http://127.0.0.1:5000/')


def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224, 3))
    x = np.expand_dims(img, axis=0)
    img_data = preprocess_input(x)
    prediction = np.argmax(model.predict(img_data), axis=1)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!

    prediction = np.argmax(model.predict(x), axis=1)

    return prediction


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction

        prediction = model_predict(file_path, model)
        ind = ['cataract', 'diabetic_retinopathy', 'glaucoma', 'normal']
        result = str(ind[prediction[0]])

        return result
    return None


if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True)
