import numpy as np
import pandas as pd
import pickle
from flask import Flask, request, jsonify


app = Flask('Premium insurance prediction')

LOCAL_MODEL_PATH = 'models/xgb_model.picke'

@app.route('/predict', methods=['POST'])
def predict():
    with open (LOCAL_MODEL_PATH, 'rb') as f_in:
        loaded_model = pickle.load(f_in)
    try:
        input_data = request.get_json()
        df = pd.DataFrame(input_data, index=[0])
        df['customer_feedback'] = np.where(df['customer_feedback'] == "Good", 1, 0)
        df = df[['credit_score', 'annual_income', 'health_score', 'customer_feedback']]
        prediction = loaded_model.predict(df)[0]
        return jsonify({'prediction': str(prediction)})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/home', methods=['GET'])
def home():
    return jsonify({'status': 'ok'}), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
