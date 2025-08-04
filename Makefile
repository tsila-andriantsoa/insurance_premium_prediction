# Makefile for Premium Insurance Prediction Project

# Variables
PYTHON=python
PREFECT_SERVER=prefect server start
MLFLOW_UI=mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns

# Default target
.PHONY: help
help:
	@echo "Makefile Commands:"
	@echo "  make install         - Install dependencies"
	@echo "  make train           - Train the model"
	@echo "  make predict         - Generate predictions"
	@echo "  make test            - Run unit tests"
	@echo "  make lint            - Run Python linter"
	@echo "  make mlflow-ui       - Start MLflow UI"
	@echo "  make prefect-start   - Start Prefect server"
	@echo "  make prefect-run     - Run Prefect flow"
	@echo "  make docker-build    - Build Docker image"
	@echo "  make docker-run      - Run Docker container"
	@echo "  make monitor         - Run Evidently monitoring script"

# Setup & install
install:
	pip install -r requirements.txt

# Training and prediction
prepare data:
	$(PYTHON) src/data_preparation.py

train:
	$(PYTHON) src/train.py

predict:
	$(PYTHON) src/predict.py

# Testing
test:
	pytest tests/

# Linting
lint:
	pylint --recursive=y .

# MLflow UI
mlflow-ui:
	$(MLFLOW_UI)

# Prefect
prefect-start:
	$(PREFECT_SERVER)

prefect-run:
	$(PYTHON) orchestration/orchestration.py --process True

# Docker
docker-build:
	docker build -t predict-app .

docker-run:
	docker run -d -p 5000:5000 predict-app

docker-test:
	curl -X POST http://localhost:5000/predict \
	     -H "Content-Type: application/json" \
	     -d '{"credit_score": 700, "customer_feedback_good": 1, "annual_income": 60000, "health_score": 85}'

# Monitoring
monitor:
	$(PYTHON) monitoring/evidently_basic_monitoring.py
