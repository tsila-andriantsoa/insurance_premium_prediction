
# Premium Insurance Prediction API

This project is an end to end machine learning project for practing MLOPS Zoomcamp by DataTalksClub.

## Project Structure

```
.
├── config                 # Configuration folder for experiment tracking
├── data                   # Prepared data for training model
├── notebook               # Exploration
├── orchestration          # Orchestration
├── webservice             # Model deployment
├── app.py                 # Flask app
├── Dockerfile             # Docker configuration
├── requirements.txt       # Python dependencies
├── README.md              # Documentation

```

## Features

- Training machine learning model to predict premium insurance
- Use MLFlow for experiment tracking
- Use prefect for pipeline and save model into S3 bucket.
- Loads trained ML model from S3 using `boto3`
- Dockerized for local or cloud deployment
- Simple REST API with `/predict` endpoints

## Prerequisites

- Building python environment
- Installing required packages using requirements.txt
- An **AWS S3 bucket** and AWS credentials with permission to read from S3

```env
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_BUCKET_NAME=your_s3_bucket_name
S3_MODEL_KEY=models/pipeline_baseline.pkl
```

## Step

# data preparation
src/data_preparation.py
# model training and experimentation
src/train.py

## experiment tracking
mlflow ui \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns
