import requests
import json

# Test if the streaming endpoint is reachable
url = "http://127.0.0.1:8000/query/stream"

# Get a token first (simulate login)
print("=" * 80)
print("TESTING /query/stream ENDPOINT")
print("=" * 80)

# First, let's check if we need authentication
headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer test-token-12345"  # Dummy token for testing
}

data = {
    "question": "Test query"
}

try:
    print(f"\nSending POST request to {url}")
    print(f"Data: {data}")
    
    response = requests.post(url, json=data, headers=headers, stream=True, timeout=10)
    
    print(f"\nResponse Status Code: {response.status_code}")
    print(f"Response Headers: {dict(response.headers)}")
    
    if response.status_code == 401:
        print("\n❌ AUTHENTICATION FAILED")
        print("The request is being rejected due to invalid/missing JWT token.")
        print("This is why you're not seeing logs - the request never reaches the agent.")
        print("\nResponse body:")
        print(response.text)
    elif response.status_code == 200:
        print("\n✅ Request successful! Streaming response:")
        for line in response.iter_lines():
            if line:
                print(line.decode('utf-8'))
    else:
        print(f"\n❌ Unexpected status code: {response.status_code}")
        print(response.text)
        
except requests.exceptions.ConnectionError:
    print("\n❌ Cannot connect to backend on port 8000")
except Exception as e:
    print(f"\n❌ Error: {e}")

print("=" * 80)
