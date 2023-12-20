import pickle

from flask import Flask
from flask import request
from flask import render_template
from PIL import Image
import numpy as np
from keras.preprocessing.image import img_to_array
import tensorflow as tf

model_file = 'model_final.bin'

model = tf.keras.models.load_model('model_final')

app = Flask('image_classification', template_folder='templates')

@app.route('/')
def home():
    return render_template('home.html')

class_names = ["buildings", "forest", "glacier", "mountain", "sea", "street"]

@app.route('/predict', methods=['POST'])
def predict():
    image = request.files['image']
    image_X = Image.open(image)
    image_X = tf.image.resize(image_X, [150, 150])
    X = img_to_array(image_X)
    X = np.expand_dims(X, axis=0)
    probabilities = model.predict(X)
    predicted_class = class_names[(np.argmax(probabilities, axis=1))[0]]



    return render_template('predict.html', predicted_class=predicted_class)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)