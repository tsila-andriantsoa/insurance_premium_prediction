# insurance_premium_prediction
Repository for MLOPS Zoomcamp capstone project

# A. Experiment tracking

## 1. Prepare environment

- Create environment 

``` conda create --name myenv python=3.10 ```

- List existing environment

``` conda env list ```

- Activate environment

``` conda activate myenv ```

- Install package

 ``` pip install -r requirements.txt ```

## 2. Run MLFlow

``` mlflow ui --backend-store-uri sqlite:///mlflow/db ```

## 3. Set tracking URI

``` mlflow.set_tracking_uri('sqlite:///mlflow/db') ```

## 4. Set experiment

``` mlflow.set_experiment('nyc-taxi-experiment') ```

# 5. Run experiment

``` with mlflow.start_run():
    mlflow.set_tag("developper", "Tsila")
    mlflow.log_param("train-data-path", "../data/train.csv")
    mlflow.log_param("validation-data-path", "data/green_tripdata_2025-02.parquet")
    mlflow.log_param("alpha",alpha) ```

    