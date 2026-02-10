"""Quick diagnostic script to check environment variables."""
import os
from dotenv import load_dotenv

load_dotenv()

print("=" * 60)
print("ENVIRONMENT VARIABLES CHECK")
print("=" * 60)

# Check API configuration
api_host = os.getenv("API_HOST", "localhost")
api_port = os.getenv("API_PORT", "8000")
print(f"\nAPI Configuration:")
print(f"  API_HOST: {api_host} {'(default - NOT SET IN .env)' if api_host == 'localhost' else '(from .env)'}")
print(f"  API_PORT: {api_port} {'(default - NOT SET IN .env)' if api_port == '8000' else '(from .env)'}")
print(f"  API_BASE_URL: http://{api_host}:{api_port}")

# Check MySQL configuration
mysql_host = os.getenv("MYSQL_HOST", "localhost")
mysql_port = os.getenv("MYSQL_PORT", "3306")
mysql_database = os.getenv("MYSQL_DATABASE", "")
print(f"\nMySQL Configuration:")
print(f"  MYSQL_HOST: {mysql_host}")
print(f"  MYSQL_PORT: {mysql_port}")
print(f"  MYSQL_DATABASE: {mysql_database}")

# Check if .env file exists
import pathlib
env_path = pathlib.Path(".env")
print(f"\n.env file exists: {env_path.exists()}")
if env_path.exists():
    print(f".env file location: {env_path.absolute()}")

print("\n" + "=" * 60)
print("WHAT TO DO:")
print("=" * 60)
if api_host == "localhost":
    print("\n⚠️  API_HOST is not set in .env file!")
    print("   Add this line to your .env file:")
    print(f"   API_HOST={mysql_host}  # Use the same as MYSQL_HOST")
    
if api_port == "8000":
    print("\n⚠️  API_PORT is not set in .env file!")
    print("   Add this line to your .env file:")
    print("   API_PORT=8000")

print("\n✅ After updating .env, restart the Streamlit app")
print("=" * 60)
