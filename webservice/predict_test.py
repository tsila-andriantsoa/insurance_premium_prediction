import requests

url = "http://localhost:5000/predict"

data = {
    "credit_score": "459.0",
    "annual_income" : "19900",
    "health_score" : "36.177549",
    "customer_feedback" : "poor"
}

response = requests.post(url, json=data).json()
print(response)
