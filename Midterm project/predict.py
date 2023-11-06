import pickle

from flask import Flask
from flask import request
from flask import jsonify

model_file = 'model_forest.bin'

with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

app = Flask('raintomorrow')

@app.route('/predict', methods=['POST'])
def predict():
    day = request.get_json()

    X = dv.transform([day])
    y_pred = model.predict_proba(X)[0,1]
    raintomorrow = y_pred >= 0.5
    result = {
        'rain_tomorrow_porbability': y_pred,
        'rain_tomorrow': bool(raintomorrow)
    }
    return jsonify(y_pred)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)