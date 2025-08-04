import pandas as pd
from sklearn import datasets
    
from evidently import Dataset
from evidently import DataDefinition
from evidently import Report
from evidently.presets import DataDriftPreset, DataSummaryPreset

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from utils import rmsle

import pickle
import xgboost as xgb


PREPARED_TRAIN_DATA_PATH = 'data/prepared/df_training.parquet'
PREPARED_VALIDATION_DATA_PATH = 'data/prepared/df_validation.parquet'
MODEL_PATH = 'models/xgb_model.pickle'



# data labelling
target = "premium_amount"
numerical_features = ["credit_score", "customer_feedback_good", "annual_income", "health_score"]

def prepare_data():
    train = pd.read_parquet(PREPARED_TRAIN_DATA_PATH)
    validation = pd.read_parquet(PREPARED_VALIDATION_DATA_PATH)

    with open (MODEL_PATH, 'rb') as f_in:
        loaded_model = pickle.load(f_in)

    X_train = train[numerical_features]
    y_train = train[target].values

    X_validation = validation[numerical_features]
    y_validation = validation[target].values

    xgb_train = xgb.DMatrix(X_train, label=y_train)
    xgb_valid = xgb.DMatrix(X_validation, label=y_validation)

    train_preds = loaded_model.predict(xgb_train)
    train['prediction'] = train_preds

    validation_preds = loaded_model.predict(xgb_valid)
    validation['prediction'] = validation_preds

    # Map the column types
    schema = DataDefinition(
        numerical_columns=numerical_features,
    )

    # Create Evidently Datasets to work with
    eval_data_1 = Dataset.from_pandas(
        pd.DataFrame(train),
        data_definition=schema
    )

    eval_data_2 = Dataset.from_pandas(
        pd.DataFrame(validation),
        data_definition=schema
    )
    print("data prepared for monitoring")
    return eval_data_1, eval_data_2

# Get report
def generate_report(data1, data2):

    report = Report([
        DataSummaryPreset(),
        DataDriftPreset(),

    ])

    my_eval = report.run(data1, data2)
    # save the report
    my_eval.save_html("evidently_report.html")
    print("evidently reporting generated!")

if __name__ == "__main__":
    data_eval_1, data_eval_2 = prepare_data()
    generate_report(data_eval_1, data_eval_2)