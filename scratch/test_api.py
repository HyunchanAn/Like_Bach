import requests
import json

url = "http://localhost:8000/api/generate"
data = {
    "subject_notes": [{"pitch": 60, "duration": 1.0, "offset": 0.0, "voice": 1}],
    "target_measures": 4,
    "temperature": 0.8,
    "refine_iters": 3
}

try:
    response = requests.post(url, json=data)
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
except Exception as e:
    print(f"Error: {e}")
