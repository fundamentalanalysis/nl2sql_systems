"""Test script for streaming endpoint."""
import requests
import json

API_BASE_URL = "http://localhost:8000"

question = "How many records are in the database?"

print(f"Testing streaming endpoint with question: {question}\n")
print("=" * 80)

try:
    response = requests.post(
        f"{API_BASE_URL}/query/stream",
        json={"question": question},
        headers={"Accept": "text/event-stream"},
        stream=True,
        timeout=300
    )
    
    print(f"Response status: {response.status_code}\n")
    
    if response.status_code == 200:
        print("Receiving events:\n")
        for line in response.iter_lines():
            if line:
                line_str = line.decode('utf-8')
                if line_str.startswith('data: '):
                    event_data = line_str[6:]
                    event = json.loads(event_data)
                    print(f"Event: {event.get('type')} - {event}")
    else:
        print(f"Error: {response.text}")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
