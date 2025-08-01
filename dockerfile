# Use a lightweight Python base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy requirements.txt and install dependencies
COPY ["requirements.txt", "./"]

# Install package from requirements.txt
RUN pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt

# Copy application code and model
COPY ["webservice/predict_test.py", "./"]

# Expose the Flask app's port
EXPOSE 5000

# Start the Flask application
CMD ["waitress-serve", "--host=0.0.0.0", "--port=5000", "app:app"]