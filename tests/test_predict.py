import os
import sys

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)  

from src import predict


def test_model_file_exists():
    assert os.path.exists(
        predict.BEST_MODEL
    ), f"Model file not found at {predict.BEST_MODEL}"

def test_load_test_data():
    data = predict.load_test_data()
    assert not data.empty, "Test data is empty"


def test_prepared_test_data():
    df_test = predict.load_test_data()
    test_columns = list(predict.prepare_test_data(df_test).columns)
    expected_columns = ['credit_score', 'customer_feedback_good', 'annual_income', 'health_score']
    assert test_columns == expected_columns, f"Expected columns {expected_columns}, but got {test_columns}"


def test_get_prediction():
    assert os.path.exists(
        f"{predict.RESULT_DATA_PATH}submission.csv"
    ), f"Prediction results found at {predict.BEST_MODEL}"