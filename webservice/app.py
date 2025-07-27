from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
import os
import boto3
from botocore.exceptions import NoCredentialsError

app = Flask('Premium insurance prediction')


def load_model_from_s3(bucket_name, s3_key, local_path):
    s3 = boto3.client(
        's3',
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
    )

    try:
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        s3.download_file(bucket_name, s3_key, local_path)
        print(f"Model downloaded from S3: s3://{bucket_name}/{s3_key}")
    except NoCredentialsError:
        print("AWS credentials not found.")
        raise
    except Exception as e:
        print(f"Failed to download model: {e}")
        raise


# Load model from S3
S3_BUCKET = os.getenv("AWS_BUCKET_NAME")
S3_KEY = os.getenv("S3_MODEL_KEY") 
LOCAL_MODEL_PATH = "model/pipeline_baseline.pkl"

load_model_from_s3(S3_BUCKET, S3_KEY, LOCAL_MODEL_PATH)
loaded_pipeline = joblib.load(LOCAL_MODEL_PATH)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = request.get_json()
        df = pd.DataFrame(input_data, index=[0])
        df['customer_feedback'] = np.where(df['customer_feedback'] == "Good", 1, 0)

        df = df[['credit_score', 'annual_income', 'health_score', 'customer_feedback']]
        prediction = loaded_pipeline.predict(df)[0]

        return jsonify({'prediction': str(prediction)})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/home', methods=['GET'])
def home():
    return jsonify({'status': 'ok'}), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
