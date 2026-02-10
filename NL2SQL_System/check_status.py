import requests
import json

print("=" * 80)
print("CHECKING BACKEND STATUS")
print("=" * 80)

try:
    response = requests.get("http://127.0.0.1:8000/health", timeout=5)
    health = response.json()
    
    print(f"\n✅ Backend is RUNNING on port 8000")
    print(f"\nHealth Status:")
    print(f"  Overall Status: {health.get('status')}")
    print(f"  Database Connected: {health.get('database_connected')}")
    print(f"  Agent Ready: {health.get('agent_ready')}")
    print(f"  Redis Connected: {health.get('redis_connected')}")
    
    if not health.get('agent_ready'):
        print(f"\n❌ PROBLEM: Agent is NOT READY")
        print(f"   This means the agent failed to initialize during startup.")
        print(f"   Check the backend terminal for error messages during startup.")
    
except requests.exceptions.ConnectionError:
    print(f"\n❌ Backend is NOT RUNNING on port 8000")
    print(f"   Cannot connect to http://127.0.0.1:8000")
except Exception as e:
    print(f"\n❌ Error checking status: {e}")

print("=" * 80)
