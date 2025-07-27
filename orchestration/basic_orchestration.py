import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error

import pickle

import xgboost as xgb
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope

pd.set_option('display.max_columns', 100)


import mlflow
mlflow.set_tracking_uri('sqlite:////workspaces/mlflow.db')

mlflow.set_experiment('insurance-premium-experiment')

models_folder = Path('models')
models_folder.mkdir(exist_ok=True)

def read_data(data_path):
    df_training = pd.read_csv(data_path)
    return df_training

def prepare_data(df):
    X = df.drop(['premium_amount'], axis = 1)
    y = df['premium_amount'].values
    return X, y

def rmsle(y_true, y_pred):
    y_pred = np.maximum(0, y_pred)
    return np.sqrt(mean_squared_log_error(y_true, y_pred))


def train_model(X_train, y_train, X_val, y_val):
    with mlflow.start_run() as run:
        train = xgb.DMatrix(X_train, label=y_train)
        valid = xgb.DMatrix(X_val, label=y_val)
        
        best_parameters = {
            'max_depth': 6,
            'learning_rate': 0.1967,
            'min_child_weight': 1.8356,
            'objective': 'reg:linear',
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
        rmsle_score = rmsle(y_val, y_pred)
        mlflow.log_metric("rmsle", rmsle_score)
        mlflow.xgboost.log_model(booster, artifact_path="models_mlflow")

        return run.info.run_id


def run():
    train_path = "../data/prepared/df_training.csv"
    validation_path = "../data/prepared/df_validation.csv"    

    df_train = read_data(train_path)
    df_validation = read_data(validation_path)

    X_train, y_train = prepare_data(df_train)
    X_validation, y_validation = prepare_data(df_validation)

    run_id = train_model(X_train, y_train, X_validation, y_validation)
    print(f"MLflow run_id: {run_id}")
    return run_id


if __name__ == "__main__":

    run_id = run()
    print('run_id', run_id)