import pickle

from flask import Flask
from flask import request
from flask import render_template
import pandas as pd

model_file = 'model_regressor.bin'

with open(model_file, 'rb') as f_in:
    dv, transformer, regressor = pickle.load(f_in)

app = Flask('energy_use',  template_folder='templates')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.files['csv']
    df = pd.read_csv(data)
    X = dv.transform([df])
    X_norm = transformer.transform(X)
    y_pred = regressor.predict(X_norm)

    return render_template('predict.html', predicted_value=y_pred)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)

