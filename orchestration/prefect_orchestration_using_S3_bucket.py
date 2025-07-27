import os
from prefect import flow, task
import numpy as np
import pandas as pd
from pathlib import Path
import xgboost as xgb
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import train_test_split
import mlflow

# Load AWS credentials and S3 config from env variables
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_BUCKET_NAME = os.getenv("AWS_BUCKET_NAME", "insurance-premium-bucket")

# Set env vars for boto3 and MLflow
os.environ["AWS_ACCESS_KEY_ID"] = AWS_ACCESS_KEY_ID
os.environ["AWS_SECRET_ACCESS_KEY"] = AWS_SECRET_ACCESS_KEY

# MLflow setup
mlflow.set_tracking_uri('sqlite:///mlflow.db')  # Local DB file
mlflow.set_experiment(
    experiment_name='insurance-premium-experiment',
    artifact_location=f"s3://{AWS_BUCKET_NAME}/mlflow-artifacts"
)

models_folder = Path('models')
models_folder.mkdir(exist_ok=True)


@task
def read_data(data_path):
    return pd.read_csv(data_path)


@task
def prepare_data(df):
    X = df.drop(['premium_amount'], axis=1)
    y = df['premium_amount'].values
    return X, y


@task
def rmsle(y_true, y_pred):
    y_pred = np.maximum(0, y_pred)
    return np.sqrt(mean_squared_log_error(y_true, y_pred))


@task
def train_model(X_train, y_train, X_val, y_val):
    with mlflow.start_run() as run:
        train = xgb.DMatrix(X_train, label=y_train)
        valid = xgb.DMatrix(X_val, label=y_val)

        best_parameters = {
            'max_depth': 6,
            'learning_rate': 0.1967,
            'min_child_weight': 1.8356,
            'objective': 'reg:squarederror',
            'seed': 42
        }

        mlflow.log_params(best_parameters)

        booster = xgb.train(
            params=best_parameters,
            dtrain=train,
            num_boost_round=30,
            evals=[(valid, 'validation')],
            early_stopping_rounds=10
        )

        y_pred = booster.predict(valid)
        rmsle_score = rmsle.fn(y_val, y_pred)
        mlflow.log_metric("rmsle", rmsle_score)

        mlflow.xgboost.log_model(booster, artifact_path="models_mlflow")

        return run.info.run_id


@flow(name="Insurance Premium Model Training")
def training_pipeline():
    train_path = "../data/prepared/df_training.csv"
    val_path = "../data/prepared/df_validation.csv"

    df_train = read_data(train_path)
    df_val = read_data(val_path)

    X_train, y_train = prepare_data(df_train)
    X_val, y_val = prepare_data(df_val)

    run_id = train_model(X_train, y_train, X_val, y_val)
    print(f"MLflow run_id: {run_id}")
    return run_id


if __name__ == "__main__":
    training_pipeline()
