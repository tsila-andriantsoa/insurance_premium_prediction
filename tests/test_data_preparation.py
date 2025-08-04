import os
import sys

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)  
from src import data_preparation

def test_load_raw_data():
    assert os.path.exists(
        data_preparation.RAW_DATA_PATH
    ), f"Raw data not found at {data_preparation.RAW_DATA_PATH}"

def test_prepare_data_for_training():
    assert os.path.exists(
        data_preparation.PREPARED_DATA_PATH
    ), f"Processed data not found at {data_preparation.PREPARED_DATA_PATH}"
