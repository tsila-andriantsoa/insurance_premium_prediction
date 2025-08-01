
# Premium Insurance Prediction

The objectif of this project is to provide an end to end machine learning project to predict the Insurance Premium Prediction from the dataset [https://www.kaggle.com/competitions/playground-series-s4e12/data](Regression with an Insurance Dataset).
and practice Machine Learning Operations from the amazing courses of MLOPS Zoomcamp by DataTalksClub. 

## Dataset

The dataset is available on [Kaggle](https://www.kaggle.com/competitions/playground-series-s4e12/overview)

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
- An **AWS S3 bucket** and AWS credentials with permission to read from S3 (ACCESS_KEY_ID, SECRET_ACCESS_KEY, BUCKET_NAME, MODEL_KEY)

## Step

### data preparation
src/data_preparation.py
### model training and experimentation
src/train.py

### experiment tracking
mlflow ui \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns

### get prediction and save submission data
src/predict.py

### build prefect pipeline
Start prefect server
prefect server start

Run prefect orchestration
python orchestration/orchestration.py --process True


### Evidently
- create docker-compose file

- build docker container
docker-compose up

- test runing service (adminer, grafana)

- model monitoring



### Deploy model using Docker
Build docker image
docker build -t predict-app .

Run docker image
docker run -d -p 5000:5000 predict-app

Do prediction
python webservice/predict_test.py

## Best pratices
