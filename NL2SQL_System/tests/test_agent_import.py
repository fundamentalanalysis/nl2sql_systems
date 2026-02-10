import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

try:
    from agent.agent import get_agent
    print("Successfully imported get_agent")
    agent = get_agent()
    print("Successfully initialized agent")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
