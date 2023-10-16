import requests

url = 'http://localhost:9697/predict'


client = {
    "job": "retired", 
    "duration": 445, 
    "poutcome": "success"
    }

response = requests.post(url, json=client).json()

print(client)
print(response)