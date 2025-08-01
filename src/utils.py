from sklearn.metrics import mean_squared_log_error
import numpy as np
# Define evaluation metrics
def rmsle(y_true, y_pred):
    # Clip predictions to avoid log(0)
    y_pred = np.maximum(0, y_pred)
    return np.sqrt(mean_squared_log_error(y_true, y_pred))