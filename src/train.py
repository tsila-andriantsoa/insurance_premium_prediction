import os
import pandas as pd

import pickle

import xgboost as xgb
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope

import mlflow

from utils import rmsle, prepare_xgb_data

RAW_DATA_PATH = "data/raw/"
PREPARED_DATA_PATH = "data/prepared/"
MODEL_PATH = "models/"
os.makedirs(MODEL_PATH, exist_ok=True)

def load_data():
    df_train = pd.read_parquet(PREPARED_DATA_PATH + 'df_training.parquet')
    df_validation = pd.read_parquet(PREPARED_DATA_PATH+ 'df_validation.parquet')
    print("data loaded")
    return df_train, df_validation

def train_models_and_log_experiments(df_train, df_validation):
    _, _, _, y_validation_part, train, valid = prepare_xgb_data(df_train, df_validation)
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment('insurance-premium-experiment')
    def objective(params):
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
        return {
            'loss': rmse, 'status': STATUS_OK}
    params = {
        'max_depth': scope.int(hp.quniform('max_depth', 4, 50, 1)),
        'learning_rate': hp.loguniform('learning_rate', -3, 0),
        'min_child_weight': hp.loguniform('min_child_weight', -1, 3),
        'objective': 'reg:linear',
        'seed': 42,
    }    
    fmin(
        fn=objective,
        space=params,
        algo=tpe.suggest,
        max_evals=10,
        trials=Trials(),
    )

def train_best_model(df_train, df_validation):
    _, _, _, y_validation_part, train, valid = prepare_xgb_data(df_train, df_validation)
    best_parameters = {
        'max_depth': 6,
        'learning_rate': 0.1967,
        'min_child_weight': 1.8356,
        'objective': 'reg:linear',
        'seed': 42,
    }
    booster = xgb.train(
        params=best_parameters,
        dtrain=train,
        num_boost_round=1000,
        evals=[(valid, 'validation')],
        early_stopping_rounds=10,
    )
    y_pred = booster.predict(valid)
    rmse = rmsle(y_validation_part, y_pred)
    mlflow.log_metric("rmse", rmse)
    mlflow.xgboost.log_model(booster, artifact_path="models_mlflow")
    print("Best model trained")
    return booster

def save_model(model):
    with open(MODEL_PATH + 'xgb_model.pickle', 'wb') as f_out:
        pickle.dump(model, f_out)
    print("Best model saved")

if __name__ == "__main__":
    train, validation = load_data()
    train_models_and_log_experiments(train, validation)
    best_model = train_best_model(train, validation)
    save_model(best_model)