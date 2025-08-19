import requests

url = "http://127.0.0.1:5000/predict_sequence"  # Correct endpoint
payload = {"sequence": [5, 7, 9, 11, 13]}  # Example sequence
response = requests.post(url, json=payload)

print(response.json())
