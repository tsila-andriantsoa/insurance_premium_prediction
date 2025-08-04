import sys
import os
import pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

import xgboost as xgb
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope

from prefect import flow, task

import mlflow

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from utils import rmsle

mlflow.set_tracking_uri('sqlite:////workspaces/mlflow.db')
mlflow.set_experiment('insurance-premium-experiment')

RAW_DATA_PATH = "data/raw/"
PREPARED_DATA_PATH = "data/prepared/"
os.makedirs(PREPARED_DATA_PATH, exist_ok=True)
MODEL_PATH = "models/"
os.makedirs(MODEL_PATH, exist_ok=True)
RESULT_DATA_PATH = "data/result/"
os.makedirs(RESULT_DATA_PATH, exist_ok=True)
BEST_MODEL = MODEL_PATH + 'xgb_model.pickle'

@task(log_prints=True)
def load_raw_data():
    train = pd.read_parquet(RAW_DATA_PATH + 'train.parquet')
    test = pd.read_parquet(RAW_DATA_PATH + 'test.parquet')
    print("raw data loaded")
    return train, test

@task(log_prints=True)
def preprocess_data(df):
    # rename columns    
    df.columns = [str.lower(col).replace(' ','_') for col in df.columns]
    # remove null value
    df.dropna(subset='premium_amount',inplace=True)
    # remove unused columns
    df.drop(columns=['id'], inplace = True)
    df.drop(columns = ['previous_claims', 'occupation'], inplace = True)
    print("train data preprocessed")
    return df

@task(log_prints=True)
def prepared_train_validation_data(df):
    numerical_features = df.select_dtypes(include = 'float').columns.tolist()
    numerical_features = [n for n in numerical_features if n != 'premium_amount' if n!= 'previous_claims']
    categorical_features = df.select_dtypes(include = 'object').columns.tolist()
    categorical_features = [n for n in categorical_features if n != 'policy_start_date' if n!= 'occupation']
    # Preprocessing for numerical data
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')), 
        ('scaler', StandardScaler()),
    ])
    # Preprocessing for categorical data
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),  
        ('onehot', OneHotEncoder(drop = 'first', handle_unknown='ignore')),
    ])
    # Combine preprocessors in a ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features),
        ])
    pipeline = Pipeline([
        ('preprocessor',  preprocessor),
    ])
    X_train = df.drop(columns = ['premium_amount'])
    Y_train = df['premium_amount']
    pipeline.fit(X_train, Y_train)

    X_train_transformed = pipeline.named_steps['preprocessor'].transform(X_train)
    numerical_transformed_columns = pipeline.named_steps['preprocessor'].transformers_[0][2]
    categorical_transformed_columns = pipeline.named_steps['preprocessor'].transformers_[1][1].get_feature_names_out().tolist()
    all_columns = [numerical_transformed_columns + categorical_transformed_columns]
    all_columns = all_columns[0]
    all_columns = [str.lower(col).replace('\'','').replace(' ','_') for col in all_columns]

    df_X_full_train_transformed = pd.DataFrame(
        X_train_transformed,
        columns = all_columns,
    )

    # select best columns
    top_features = ['credit_score', 'customer_feedback_good', 'annual_income', 'health_score']
    X_train_part, X_validation_part, y_train_part, y_validation_part = train_test_split(df_X_full_train_transformed[top_features], Y_train, test_size=0.2, random_state=42)
    # save data
    df_training = X_train_part.copy()
    df_training['premium_amount'] = y_train_part.values  # Or just y_training_part if the index aligns
    df_validation = X_validation_part.copy()
    df_validation['premium_amount'] = y_validation_part.values  # Or just y_training_part if the index aligns
    df_training.to_parquet(PREPARED_DATA_PATH + 'df_training.parquet', index=False)
    df_validation.to_parquet(PREPARED_DATA_PATH + 'df_validation.parquet', index=False)
    print('train data and validation data saved into path: ', PREPARED_DATA_PATH)
    return df_training, df_validation

@task(log_prints=True)
def train_models_and_log_experiments(df_train, df_validation):
    X_train_part = df_train.drop(columns = ['premium_amount'])
    y_train_part = df_train[['premium_amount']]
    X_validation_part = df_validation.drop(columns = ['premium_amount'])
    y_validation_part = df_validation[['premium_amount']]
    train = xgb.DMatrix(X_train_part, label=y_train_part)
    valid = xgb.DMatrix(X_validation_part, label=y_validation_part)
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
            'loss': rmse, 
            'status': STATUS_OK,
        }
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
        trials=Trials()
    )

@task(log_prints=True)
def train_best_model(df_train, df_validation):
    X_train_part = df_train.drop(columns = ['premium_amount'])
    y_train_part = df_train[['premium_amount']]
    X_validation_part = df_validation.drop(columns = ['premium_amount'])
    y_validation_part = df_validation[['premium_amount']]
    train = xgb.DMatrix(X_train_part, label=y_train_part)
    valid = xgb.DMatrix(X_validation_part, label=y_validation_part)
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

@task(log_prints=True)
def save_model(model):
    with open(MODEL_PATH + 'xgb_model.pickle', 'wb') as f_out:
        pickle.dump(model, f_out)
    print("Best model saved")

@task(log_prints=True)
def load_test_data():
    df_test = pd.read_parquet(RAW_DATA_PATH + 'test.parquet')
    print("test data loaded")
    return df_test

@task(log_prints=True)
def prepare_test_data(df):
    # rename columns    
    df.columns = [str.lower(col).replace(' ','_') for col in df.columns]
    # create feature
    df['customer_feedback_good'] = np.where(df['customer_feedback'] == "Good", 1, 0)
    df = df[['credit_score', 'customer_feedback_good','annual_income', 'health_score']]
    return df

@task(log_prints=True)    
def predict(df):

    X_matrix = xgb.DMatrix(df)
    with open(BEST_MODEL, 'rb') as f_in:
        best_model = pickle.load(f_in)
    predictions = best_model.predict(X_matrix)
    print("prediction done!")
    return predictions

@task(log_prints=True)
def save_predictions(df_test, predictions):
    prediction = pd.DataFrame({
        'id': df_test['id'].values,
        'premium_amount': predictions
    })
    prediction.to_csv(RESULT_DATA_PATH + 'submission.csv', index = False, mode='w')
    print("prediction save at", RESULT_DATA_PATH)
    
@flow(name="Insurance Premium pipeline")
def my_pipeline():
    df_train, df_validation = load_raw_data()
    df_train_preprocessed = preprocess_data(df_train)
    df_training, df_validation = prepared_train_validation_data(df_train_preprocessed) 
    # train_models_and_log_experiments(df_training, df_validation)
    best_model = train_best_model(df_training, df_validation)
    save_model(best_model)
    df_test = load_test_data()
    df_test_prepared = prepare_test_data(df_test)
    predictions = predict(df_test_prepared)
    save_predictions(df_test, predictions)

if __name__ == "__main__":
    my_pipeline()

