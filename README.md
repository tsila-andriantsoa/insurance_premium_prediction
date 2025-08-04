
# Premium Insurance Prediction

## Problem description

Insurance companies rely on accurate risk assessment to determine the premium charged to each customer. However, with diverse and often non-linear relationships between customer attributes (like credit score, income, health condition, etc.) and claim risks, it becomes challenging to set a fair and profitable premium.

This project tackles the challenge of predicting the insurance premium a customer should be charged using historical data from a Kaggle competition. By building a robust machine learning pipeline, we aim to:
* Help insurance companies improve their pricing strategy
* Ensure customers are charged fairly based on their risk profile
* Demonstrate a full MLOps pipeline from data to deployment using tools like MLflow, Prefect, Docker, and AWS S3

The project is also aligned to pratice course from MLOps Zoomcamp by DataTalksClub.

## Dataset

The dataset is available on [Kaggle](https://www.kaggle.com/competitions/playground-series-s4e12/overview)

| Column Name             | Type         | Description |
|-------------------------|--------------|-------------|
| **age**                 | Numeric      | Age of the policyholder in years. Likely influences health risk and insurance pricing. |
| **gender**              | Categorical  | Gender of the policyholder, typically `"Male"` or `"Female"`. May correlate with health or lifestyle behaviors. |
| **annual_income**       | Numeric      | The yearly income of the policyholder. Higher income may be associated with higher coverage and premium levels. |
| **marital_status**      | Categorical  | Indicates whether the policyholder is `"Single"`, `"Married"`, `"Divorced"`, etc. Can affect claim probability and policy type. |
| **number_of_dependents**| Numeric      | Number of dependents (e.g., children or family members). May influence policy size and premium. |
| **education_level**     | Categorical  | Highest level of education achieved (e.g., `"High School"`, `"Bachelor"`, `"PhD"`). Can be a proxy for socio-economic status. |
| **occupation**          | Categorical  | Job title or employment category of the policyholder. May relate to risk level (e.g., hazardous jobs). |
| **health_score**        | Numeric      | A synthetic score measuring overall health. Higher scores likely indicate better health and lower risk. |
| **location**            | Categorical  | Geographic region or ZIP-level code where the policyholder resides. Can impact risk profile and premiums. |
| **policy_type**         | Categorical  | Type of insurance policy, such as `"Basic"`, `"Standard"`, `"Premium"`, etc. Determines level of coverage. |
| **previous_claims**     | Numeric      | Number of prior insurance claims. Indicates claim history; higher values may reflect higher risk. |
| **vehicle_age**         | Numeric      | Age of the insured vehicle in years. Relevant for auto insurance policies. Older vehicles may have different premium profiles. |
| **credit_score**        | Numeric      | Creditworthiness of the policyholder. Often used in underwriting to assess reliability and risk. |
| **insurance_duration**  | Numeric      | Number of months or years the policy has been active. Longer durations may indicate loyalty and lower churn. |
| **policy_start_date**   | Date / String| The date when the insurance policy began. Useful for deriving features like policy age, seasonality, etc. |
| **customer_feedback**   | Categorical / Text | Feedback score or qualitative feedback from the customer. May reflect satisfaction or service experience. |
| **smoking_status**      | Categorical  | Indicates if the policyholder is a smoker or not. A key health risk factor affecting premiums. |
| **exercise_frequency**  | Categorical / Numeric | Frequency of exercise (e.g., `"Daily"`, `"Rarely"`, `"Never"`). Often used as a lifestyle indicator. |
| **property_type**       | Categorical  | Type of insured property (e.g., `"House"`, `"Apartment"`, `"Condo"`). Relevant for home insurance policies. |
| **premium_amount**      | Numeric (Target) | The target variable — total premium amount charged for the policy. This is what the model aims to predict. |


## Project Structure

```
.
├── config                 # Configuration folder for experiment tracking
├── data                   # Data
├──├── raw                 # Raw
├── notebook               # Exploration
├── orchestration          # Orchestration
├── webservice             # Model deployment
├── Dockerfile             # Docker configuration
├── requirements.txt       # Python dependencies
├── README.md              # Documentation

```

## Features

- Training machine learning model to predict premium insurance
- Use MLFlow for experiment tracking
- Use prefect for pipeline and save model into S3 bucket.
- Loads trained ML model from S3 using `boto3`
- Dockerized for local or cloud deployment
- Simple REST API with `/predict` endpoints

## Prerequisites

- Building python environment
- Installing required packages using requirements.txt
- An **AWS S3 bucket** and AWS credentials with permission to read from S3 (ACCESS_KEY_ID, SECRET_ACCESS_KEY, BUCKET_NAME, MODEL_KEY)

## Step

### data preparation
src/data_preparation.py
### model training and experimentation
src/train.py

### experiment tracking
mlflow ui \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns

### get prediction and save submission data
src/predict.py

### build prefect pipeline
Start prefect server
prefect server start

Run prefect orchestration
python orchestration/orchestration.py --process True


### Projet deployment
- create docker-compose file

- build docker container
docker-compose up

- test runing service (adminer, grafana)

- model monitoring



### Deploy model using Docker
Build docker image
docker build -t predict-app .

Run docker image
docker run -d -p 5000:5000 predict-app

Do prediction
python webservice/predict_test.py


### Monitoring with Evidently

## Best pratices
python linting : pylint --recussive=y .
