from sklearn.metrics import mean_squared_log_error
import numpy as np
import xgboost as xgb

def say_hello():
    print("hello")

# Define evaluation metrics
def rmsle(y_true, y_pred):
    # Clip predictions to avoid log(0)
    y_pred = np.maximum(0, y_pred)
    return np.sqrt(mean_squared_log_error(y_true, y_pred))

# Prepare data for xgboost
def prepare_xgb_data(df_train, df_validation):
    X_train_part = df_train.drop(columns = ['premium_amount'])
    y_train_part = df_train[['premium_amount']]
    X_validation_part = df_validation.drop(columns = ['premium_amount'])
    y_validation_part = df_validation[['premium_amount']]
    xgb_train = xgb.DMatrix(X_train_part, label=y_train_part)
    xgb_valid = xgb.DMatrix(X_validation_part, label=y_validation_part)
    return X_train_part, y_train_part, X_validation_part, y_validation_part, xgb_train, xgb_valid