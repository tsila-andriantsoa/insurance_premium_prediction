import os
import sys

# Allow code between imports by ignoring E402 for this line
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)  # noqa: E402
from src import data_preparation, predict, train


def test_raw_data_exists():
    assert os.path.exists(
        data_preparation.RAW_DATA_PATH
    ), f"Raw data not found at {data_preparation.RAW_DATA_PATH}"


