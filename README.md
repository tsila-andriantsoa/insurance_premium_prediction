
# Premium Insurance Prediction

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
- An **AWS S3 bucket** and AWS credentials with permission to read from S3 (ACCESS_KEY_ID, SECRET_ACCESS_KEY, BUCKET_NAME, MODEL_KEY)

