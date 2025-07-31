import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error
from sklearn.preprocessing import StandardScaler

import xgboost as xgb
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope

import os

RAW_DATA_PATH = "data/raw/"
PREPARED_DATA_PATH = "data/prepared/"


# Define evaluation metrics
def rmsle(y_true, y_pred):
    # Clip predictions to avoid log(0)
    y_pred = np.maximum(0, y_pred)
    return np.sqrt(mean_squared_log_error(y_true, y_pred))

def train():

    df_train = pd.read_parquet(PREPARED_DATA_PATH + 'df_training.parquet')
    df_validation = pd.read_parquet(PREPARED_DATA_PATH+ 'df_validation.parquet')

    X_train_part = df_train.drop(columns = ['premium_amount'])
    y_train_part = df_train[['premium_amount']]

    X_validation_part = df_validation.drop(columns = ['premium_amount'])
    y_validation_part = df_validation[['premium_amount']]

    train = xgb.DMatrix(X_train_part, label=y_train_part)
    valid = xgb.DMatrix(X_validation_part, label=y_validation_part)

    import mlflow
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment('insurance-premium-experiment')

    params = {
        'max_depth': scope.int(hp.quniform('max_depth', 4, 50, 1)),
        'learning_rate': hp.loguniform('learning_rate', -3, 0),
        'min_child_weight': hp.loguniform('min_child_weight', -1, 3),
        'objective': 'reg:linear',
        'seed': 42
    }

    with mlflow.start_run():
        mlflow.set_tag("model", "xgboost")
        mlflow.log_params(params)        
        booster = xgb.train(
            params=params,
            dtrain=train,
            num_boost_round=1000,
            evals=[(valid, 'validation')],
            early_stopping_rounds=10
        )
        y_pred = booster.predict(valid)
        rmse = rmsle(y_validation_part, y_pred)
        mlflow.log_metric("rmse", rmse)
        mlflow.sklearn.log_model(booster, "xgb_model")

best_result = fmin(
    fn=objective,
    space=search_space,
    algo=tpe.suggest,
    max_evals=10,
    trials=Trials()
)

best_parameters = {
    'max_depth': 6,
    'learning_rate': 0.1967,
    'min_child_weight': 1.8356,
    'objective': 'reg:linear',
    'seed': 42
}

