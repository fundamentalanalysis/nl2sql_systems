
import os
import sys
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage
from pydantic import SecretStr

# Load environment variables
load_dotenv()

def test_llm_connection():
    print("🚀 Testing Azure OpenAI Connection...", flush=True)
    
    endpoint = os.getenv("AZURE_AI_ENDPOINT")
    api_key = os.getenv("AZURE_AI_API_KEY")
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
    api_version = os.getenv("AZURE_AI_API_VERSION")
    
    print(f"   Endpoint: {endpoint}", flush=True)
    if api_key:
        print(f"   API Key found: {api_key[:5]}...", flush=True)
    else:
        print("   ❌ API Key NOT found in environment variables", flush=True)
        
    print(f"   Deployment: {deployment}", flush=True)
    
    try:
        llm = AzureChatOpenAI(
            azure_endpoint=endpoint,
            openai_api_key=SecretStr(api_key) if api_key else None,
            azure_deployment=deployment,
            api_version=api_version,
            temperature=0
        )
        
        print("   Sending test message...", flush=True)
        response = llm.invoke([HumanMessage(content="Hello, are you working?")])
        
        print(f"✅ LLM Response: {response.content}", flush=True)
        return True
    except Exception as e:
        print(f"❌ LLM Connection Failed: {e}", flush=True)
        return False

if __name__ == "__main__":
    test_llm_connection()
