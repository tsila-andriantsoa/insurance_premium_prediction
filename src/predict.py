
import pandas as pd
import pickle
import numpy as np
import xgboost as xgb
from utils import *

RAW_DATA_PATH = "data/raw/"
PREPARED_DATA_PATH = "data/prepared/"
BEST_MODEL = "models/xgb_model.pickle"
RESULT_DATA_PATH = "data/result/"

def load_test_data():
    df_test = pd.read_parquet(RAW_DATA_PATH + 'test.parquet')
    print("test data loaded")
    return df_test

def prepare_test_data(df):
    # rename columns    
    df.columns = [str.lower(col).replace(' ','_') for col in df.columns]
    # create feature
    df['customer_feedback_good'] = np.where(df['customer_feedback'] == "Good", 1, 0)
    df = df[['credit_score', 'customer_feedback_good','annual_income', 'health_score']]
    return df
    
def predict(df):

    X_matrix = xgb.DMatrix(df)

    with open(BEST_MODEL, 'rb') as f_in:
        best_model = pickle.load(f_in)

    predictions = best_model.predict(X_matrix)
    print("prediction done!")
    return predictions

def save_predictions(df_test, predictions):
    prediction = pd.DataFrame({
        'id': df_test['id'].values,
        'premium_amount': predictions
    })
    prediction.to_csv(RESULT_DATA_PATH + 'submission.csv', index = False, mode='x')
    print("prediction save at", RESULT_DATA_PATH)
    


if __name__ == "__main__":
    df_test = load_test_data()
    df_test_prepared = prepare_test_data(df_test)
    predictions = predict(df_test_prepared)
    save_predictions(df_test, predictions)
