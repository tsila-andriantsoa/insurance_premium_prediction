import os
import sys

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)  
from src import train

def trained_model_exists():
    assert os.path.exists(
        f"{train.MODEL_PATH}/xbg_model.pickle"
    ), f"Trained model not found at {train.MODEL_PATH}"
