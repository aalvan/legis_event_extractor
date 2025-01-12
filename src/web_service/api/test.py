import requests

api_url = 'http://localhost:8000/predict'

data = {}
response = requests.post(api_url, json=data)

if response.status_code == 200:
    print(f"Prediction result: {response.json()}")
else:
    print(f"Failed to get prediction, Status code: {response.status_code}, Error: {response.text}")